import json
import shutil
import threading
from pathlib import Path

from ..models import SessionState, CandidateResult, CurvePoint
from ..curve import load_curve_csv, write_curve_csv, mv_to_uv, mhz_to_khz, _build_vlock_curve
from ..session import save_session
from .. import flightlog, faultmap
from ..anchors import (
    plateau_starts, pick_anchors, search_safe_gain, anchor_ceiling, proven_gains_from_flightlog,
    build_curve, SAFE_MARGIN_STEPS,
)
from ...hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from ...hardware.point_lock import PointLockError
from .autoplan import _AUTO_PLAN_MODES, _AUTO_PLAN_FINAL_SECONDS, _run_auto_plan_suite
from .confident import evaluate_candidate_confident
from .recovery import revert_to_last_good, _check_for_manual_recovery


_HARD_FAULT_PREFIXES = ("APPLY_FAILED", "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT",
                        "HARDWARE_ERROR_EVENTS", "GPU_FAULT_EVENT")
_ANCHOR_VERIFY_RETRIES = 3
_ANCHOR_START_MARGIN_STEPS = 2  # back off the found gains by this many steps before verifying
_ANCHOR_VERIFY_PASSES = 2
_ANCHOR_UTIL_STEP = 10
_ANCHOR_MIN_UTIL_PCT = 35
_DEFAULT_ANCHOR_CAP_MHZ = 150  # conservative ceiling when neither the profile nor --max-gain-mhz sets one
_ANCHOR_FAIL_SETTLE_SECONDS = 25  # let the driver/GPU recover after a failed point before loading it again


def run_anchor_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    """Find the max stable gain at a few locked anchors, interpolate, then verify unlocked."""
    if state.active_candidate_label.startswith("anchor_verify"):
        raise RuntimeError(f"Previous run ended during {state.active_candidate_label}; refusing automatic retry")
    if state.active_candidate_label:
        # The crashed point is already blocked in fault_map.json, so the search resumes below it.
        print(f"Previous run ended during {state.active_candidate_label}; that point is blocked, resuming below it.")
        state.active_candidate_label = ""
        save_session(state)
    if state.vlock_phase in ("failed", "done"):
        raise RuntimeError(f"Anchor session already {state.vlock_phase}; start a new session")
    if state.doloming_mode == "simple" and not state.doloming_modes:
        # Only a self-checking workload turns an unstable point into a clean error instead of a hang.
        state.doloming_mode = "canary"
        print("Stress workload: canary (verifies its own results).")
    state.point_lock = True
    out_dir = Path(state.out_dir)
    stock = load_curve_csv(Path(state.stock_curve_csv))
    step_khz = mhz_to_khz(state.step_mhz)
    cap_khz = mhz_to_khz(state.anchor_max_gain_mhz or _DEFAULT_ANCHOR_CAP_MHZ)
    results_path = out_dir / "anchors.json"
    if state.point_util_pct == 0:
        state.point_util_pct = 60
    gains: dict = {}
    retry: set = set()
    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        gains = {int(k): int(v) for k, v in saved["gains"].items()}
        retry = {int(i) for i in saved.get("retry", [])}
        # Anchors skipped or limited by missing coverage are retried on resume.
        for i in list(gains):
            if i in retry or gains[i] < 0:
                del gains[i]
        retry = set()

    def save_results() -> None:
        results_path.write_text(json.dumps({"gains": gains, "retry": sorted(retry)}), encoding="utf-8")

    starts = plateau_starts(stock, mv_to_uv(state.bin_min_mv), mv_to_uv(state.bin_max_mv))
    anchors = pick_anchors(starts)
    if not anchors:
        raise ValueError("No tunable curve bins between --bin-min-mv and --bin-max-mv")
    print(f"\n=== VoltVandal - anchor mode ===\n  Anchors: "
          + ", ".join(f"{stock[i].voltage_uv / 1000:g} mV" for i in anchors))

    if state.anchor_finish:
        for idx, g in proven_gains_from_flightlog(out_dir / "flight.jsonl", stock, anchors).items():
            if g > gains.get(idx, 0):
                gains[idx] = g
        save_results()
        print("Finish mode: no new search; using proven gains only.")
    print(f"  Gain ceiling: +{cap_khz // 1000} MHz, single {state.step_mhz} MHz steps, "
          f"{SAFE_MARGIN_STEPS * state.step_mhz} MHz safety margin.")

    cand_csv = out_dir / "candidate.csv"
    for idx in anchors:
        if state.anchor_finish and idx not in gains:
            print(f"Anchor {stock[idx].voltage_uv / 1000:g} mV has no proven gain: left out of the fit.")
            continue
        if idx in gains:
            print(f"Anchor {stock[idx].voltage_uv / 1000:g} mV already done: "
                  + ("skipped" if gains[idx] < 0 else f"+{gains[idx] // 1000} MHz"))
            continue
        v_uv, stock_f = stock[idx].voltage_uv, stock[idx].freq_khz
        inconclusive: list = []
        hard_fault: list = []

        def passes(gain_khz: int) -> bool:
            _check_for_manual_recovery(state, f"anchor_{idx}", manual_recovery_event)
            freq = stock_f + gain_khz
            if faultmap.is_blocked(out_dir, v_uv, freq):
                print(f"\nSKIPPED anchor_{v_uv // 1000}mv_{freq // 1000}mhz: this point (or a lower voltage) "
                      f"already hard-faulted (see {faultmap._FILE}).")
                hard_fault.append("BLOCKED_BY_FAULT_MAP")
                return False
            write_curve_csv(cand_csv, _build_vlock_curve(stock, idx, v_uv, freq, 0))
            label = f"anchor_{v_uv // 1000}mv_{freq // 1000}mhz"
            print(f"\n== {label} ==")
            state.active_candidate_label = label
            save_session(state)
            while True:
                try:
                    result = evaluate_candidate_confident(
                        state, cand_csv, label, interrupted_event, manual_recovery_event,
                        max_freq_mhz=freq // 1000, target_point=CurvePoint(v_uv, freq),
                    )
                except PointLockError as ex:
                    if not str(ex).startswith("INCONCLUSIVE"):
                        raise
                    result = CandidateResult(False, str(ex))
                    if state.point_util_pct - _ANCHOR_UTIL_STEP >= _ANCHOR_MIN_UTIL_PCT:
                        # Power cap sagged the clock off the target: retry with less load.
                        state.point_util_pct -= _ANCHOR_UTIL_STEP
                        print(f"Coverage inconclusive; retrying at {state.point_util_pct}% matrix/ray load.")
                        save_session(state)
                        try:
                            revert_to_last_good(state)
                        except Exception as rex:
                            print(f"WARNING: revert failed: {rex}")
                        continue
                break
            state.active_candidate_label = ""
            save_session(state)
            if result.ok:
                print("Result: PASS")
                return True
            if "INCONCLUSIVE" in result.reason:
                # e.g. power-capped: the point was not exercised, so nothing above is validated.
                inconclusive.append(result.reason)
                print(f"Result: INCONCLUSIVE | {result.reason}")
                try:
                    revert_to_last_good(state)
                except Exception as ex:
                    print(f"WARNING: revert failed: {ex}")
                return False
            print(f"Result: FAIL | {result.reason}")
            if result.reason.startswith(_HARD_FAULT_PREFIXES):
                hard_fault.append(result.reason)
            try:
                revert_to_last_good(state)
            except Exception as ex:
                print(f"WARNING: revert failed: {ex}")
            flightlog.log("revert_and_settle", label=state.active_candidate_label, settle_s=_ANCHOR_FAIL_SETTLE_SECONDS)
            print(f"Letting the GPU settle for {_ANCHOR_FAIL_SETTLE_SECONDS}s after the failure...")
            interrupted_event.wait(_ANCHOR_FAIL_SETTLE_SECONDS)
            return False

        proven = [g for g in gains.values() if g > 0]
        ceiling = anchor_ceiling(proven, cap_khz, step_khz)
        start = min(proven) // 2 if proven else 0
        gain = search_safe_gain(passes, step_khz, ceiling, start)
        if inconclusive:
            retry.add(idx)
        if inconclusive and gain == 0:
            gains[idx] = -1  # nothing validated here; excluded from the curve fit
            print(f"Anchor {v_uv / 1000:g} mV skipped: point could not be exercised under load.")
        else:
            gains[idx] = gain
            print(f"Anchor {v_uv / 1000:g} mV: max stable gain +{gain // 1000} MHz"
                  + (" (limited by inconclusive coverage above)" if inconclusive else ""))
        save_results()

    usable = {i: g for i, g in gains.items() if i in anchors and g >= 0}
    if not usable:
        state.vlock_phase = "failed"
        save_session(state)
        raise RuntimeError("No anchor could be validated; no curve built")

    margin_khz = step_khz * SAFE_MARGIN_STEPS
    for attempt in range(_ANCHOR_VERIFY_RETRIES + 1):
        curve = build_curve(stock, usable, margin_khz, step_khz)
        write_curve_csv(cand_csv, curve)
        label = f"anchor_verify_margin{margin_khz // 1000}mhz"
        print(f"\n== Verifying assembled curve (margin {margin_khz // 1000} MHz) ==")
        state.active_candidate_label = label
        save_session(state)
        for run_no in range(1, _ANCHOR_VERIFY_PASSES + 1):
            print(f"Verification pass {run_no} of {_ANCHOR_VERIFY_PASSES}")
            final = _run_auto_plan_suite(
                state, cand_csv, f"{label}_pass{run_no}", _AUTO_PLAN_FINAL_SECONDS,
                interrupted_event, manual_recovery_event, include_gpuburn=True,
                modes=state.auto_plan_modes or _AUTO_PLAN_MODES,
            )
            if not final.ok:
                break
        state.active_candidate_label = ""
        if final.ok:
            write_curve_csv(Path(state.last_good_curve_csv), curve)
            state.vlock_phase = "done"
            save_session(state)
            print("\n=== anchor tuning complete: assembled curve verified and saved ===")
            return
        print(f"Verification failed ({final.reason}).")
        margin_khz += step_khz
    nvapi_apply_curve_safe(state.gpu, Path(state.stock_curve_csv), timeout_seconds=12.0)
    shutil.copyfile(state.stock_curve_csv, state.last_good_curve_csv)
    state.vlock_phase = "failed"
    save_session(state)
    print("Curve never verified; stock curve restored.")
