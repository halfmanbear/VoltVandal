import json
import csv
import re
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .models import SessionState, CandidateResult, CurvePoint
from .utils import eprint, now_utc_iso, ensure_dir
from .curve import (
    load_curve_csv, write_curve_csv, mv_to_uv, mhz_to_khz,
    _build_vlock_curve, _build_vlock_phase2_curves, apply_offsets_to_bin
)
from .session import save_session, session_paths
from ..hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from ..hardware.monitor import NvmlMonitor
from ..stress.runner import run_doloming, run_gpuburn

_WARMUP_MIN_SECONDS = 30
_WARMUP_MAX_SECONDS = 60

def _doloming_mode_tag(mode: str) -> str:
    return mode.upper().replace("-", "_")

def _extract_summary_value(text: str, label: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else None

def _extract_first_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None

def _parse_doloming_stability(out_text: str, mode: str) -> Tuple[bool, Optional[str]]:
    mode_tag = _doloming_mode_tag(mode)
    summary_idx = out_text.rfind("Test Summary:")
    scan_text = out_text[summary_idx:] if summary_idx >= 0 else out_text
    if re.search(r"Error during (?:stress )?test:", scan_text, re.I):
        return False, f"DOLOMING_{mode_tag}_STRESS_ERROR"
    if re.search(r"cudaerror\w*|illegal memory access|device-side assert|unspecified launch failure|launch timeout|driver shutting down", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_CUDA_RUNTIME_ERROR"

    status = _extract_summary_value(scan_text, "Status")
    if status and re.search(r"\b(unstable|fail(?:ed|ure)?|error)\b", status, re.I):
        return False, f"DOLOMING_{mode_tag}_UNSTABLE_STATUS"
    if re.search(r"failed to fully stabilize", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_FAILED_TO_STABILIZE"
    return True, None

def evaluate_candidate(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
) -> CandidateResult:
    out_dir = Path(state.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _build_result(
        ok: bool,
        reason: str,
        monitor_obj: Optional[NvmlMonitor],
        stress_codes: Optional[dict] = None,
    ) -> CandidateResult:
        return CandidateResult(
            ok=ok,
            reason=reason,
            telemetry_max_temp_c=monitor_obj.max_temp if monitor_obj else None,
            telemetry_max_power_w=monitor_obj.max_power if monitor_obj else None,
            telemetry_any_throttle=monitor_obj.any_throttle if monitor_obj else None,
            stress_exit_codes=stress_codes,
            metrics=monitor_obj.metrics() if monitor_obj else None,
        )

    try:
        nvapi_apply_curve_safe(state.gpu, candidate_csv, timeout_seconds=12.0)
        interrupted_event.wait(timeout=2.0)
        if interrupted_event.is_set():
            raise KeyboardInterrupt("User pressed Ctrl+C")
    except Exception as ex:
        return _build_result(False, f"APPLY_FAILED: {ex}", None, None)

    monitor_log = logs_dir / "telemetry.csv"
    monitor: Optional[NvmlMonitor] = None
    abort_event = threading.Event()

    _modes_raw = (state.doloming_modes or "").strip()
    _multi_modes = [m.strip() for m in _modes_raw.split(",") if m.strip()] if _modes_raw else []
    _use_multi = bool(_multi_modes)
    _run_modes = _multi_modes if _use_multi else [state.doloming_mode]
    _secs_each = state.multi_stress_seconds if _use_multi else state.stress_seconds
    _expected_test_seconds = 0
    if state.doloming: _expected_test_seconds += _secs_each * len(_run_modes)
    if state.gpuburn: _expected_test_seconds += state.stress_seconds

    try:
        monitor = NvmlMonitor(
            gpu_index=state.gpu,
            poll_seconds=state.poll_seconds,
            temp_limit_c=state.temp_limit_c,
            hotspot_limit_c=state.hotspot_limit_c,
            hotspot_offset_c=state.hotspot_offset_c,
            power_limit_w=state.power_limit_w,
            abort_on_throttle=state.abort_on_throttle,
            log_csv=monitor_log,
            curve_csv=candidate_csv,
            stock_curve_csv=Path(state.stock_curve_csv),
            mode=state.mode,
            vlock_target_mv=state.vlock_target_mv,
            expected_test_seconds=_expected_test_seconds if _expected_test_seconds > 0 else None,
            live_display=state.live_display,
            use_nvapi_live=False,
        )
        monitor.start()
        abort_event = monitor.abort_event
    except Exception as e:
        eprint(f"Failed to start monitor: {e}")
        return _build_result(False, f"MONITOR_START_FAILED: {e}", None, None)

    stress_exit_codes = {}
    if state.doloming:
        _dolo_timeout = state.stress_timeout if state.stress_timeout is not None else max(_secs_each * 5, 300)
        for _mode in _run_modes:
            dololog = logs_dir / f"{candidate_label}_doloming_{_mode}.log"
            rc, out_text = run_doloming(
                state.doloming, state.gpu, _mode, _secs_each, None, dololog,
                abort_event, manual_recovery_event, interrupted_event,
                stress_timeout=_dolo_timeout, max_freq_mhz=max_freq_mhz,
            )
            _key = f"doloming_{_mode}" if _use_multi else "doloming"
            stress_exit_codes[_key] = rc
            if rc != 0:
                if monitor: monitor.stop()
                _reason = "MANUAL_RECOVERY_REQUESTED" if rc == 997 else f"DOLOMING_{_mode.upper().replace('-', '_')}_RC_{rc}"
                return _build_result(False, _reason, monitor, stress_exit_codes)
            _stable, _instability_reason = _parse_doloming_stability(out_text, _mode)
            if not _stable:
                if monitor: monitor.stop()
                return _build_result(False, _instability_reason, monitor, stress_exit_codes)

    if state.gpuburn:
        burnlog = logs_dir / f"{candidate_label}_gpuburn.log"
        _burn_timeout = state.stress_timeout if state.stress_timeout is not None else state.stress_seconds * 5
        rc, out_text, parsed_ok = run_gpuburn(
            state.gpuburn, state.stress_seconds, None, burnlog,
            abort_event, manual_recovery_event, interrupted_event,
            stress_timeout=_burn_timeout,
        )
        stress_exit_codes["gpuburn"] = rc
        if rc != 0 or not parsed_ok or re.search(r"\bnan\b|\bfailed\b|errors?\s*[:=]\s*[1-9]", out_text, re.I):
            if monitor: monitor.stop()
            _reason = f"GPUBURN_RC_{rc}" if rc != 0 else ("GPUBURN_ERRORS_DETECTED" if not parsed_ok else "GPUBURN_OUTPUT_ERROR_KEYWORD")
            return _build_result(False, _reason, monitor, stress_exit_codes)

    if monitor:
        if monitor.abort_event.is_set():
            monitor.stop()
            if monitor.driver_reset_detected:
                reason = "GPU_DRIVER_RESET_DETECTED"
            elif monitor.abort_reason:
                reason = f"MONITOR_ABORT_THRESHOLD:{monitor.abort_reason}"
            else:
                reason = "MONITOR_ABORT_THRESHOLD"
            return _build_result(False, reason, monitor, stress_exit_codes)
        monitor.stop()

    return _build_result(True, "PASS", monitor, stress_exit_codes)

def evaluate_candidate_confident(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
    passes_required: int = 1,
    max_runs: int = 1,
    warmup: bool = False,
) -> CandidateResult:
    all_results: List[CandidateResult] = []
    hard_fail_prefixes = ("APPLY_FAILED", "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT_THRESHOLD", "DOLOMING_", "GPUBURN_")

    def _is_hard_fail(reason: str) -> bool:
        return any(reason.startswith(p) for p in hard_fail_prefixes) or "CUDA_ERROR" in reason.upper()

    orig_stress = state.stress_seconds
    orig_multi = state.multi_stress_seconds
    try:
        if warmup:
            state.stress_seconds = max(
                _WARMUP_MIN_SECONDS,
                min(_WARMUP_MAX_SECONDS, max(1, orig_stress // 3)),
            )
            state.multi_stress_seconds = max(
                _WARMUP_MIN_SECONDS,
                min(_WARMUP_MAX_SECONDS, max(1, orig_multi // 2)),
            )
            w = evaluate_candidate(state, candidate_csv, f"{candidate_label}_warmup", interrupted_event, manual_recovery_event, max_freq_mhz)
            all_results.append(w)
            if not w.ok and _is_hard_fail(w.reason): return _merge_results(all_results, False, f"WARMUP_HARD_FAIL:{w.reason}")

        state.stress_seconds, state.multi_stress_seconds = orig_stress, orig_multi
        passes = 0
        fails = 0
        fail_reasons = []
        for i in range(1, max_runs + 1):
            r = evaluate_candidate(state, candidate_csv, f"{candidate_label}_run{i}", interrupted_event, manual_recovery_event, max_freq_mhz)
            all_results.append(r)
            if r.ok:
                passes += 1
                if passes >= passes_required: return _merge_results(all_results, True, f"PASS_{passes}OF{max_runs}")
            else:
                fails += 1
                fail_reasons.append(r.reason)
                if _is_hard_fail(r.reason) or fails > (max_runs - passes_required): return _merge_results(all_results, False, f"HARD_FAIL:{r.reason}" if _is_hard_fail(r.reason) else f"FAIL_{fails}OF{max_runs}:{fail_reasons[-1]}")
        return _merge_results(all_results, passes >= passes_required, f"PASS_{passes}OF{max_runs}" if passes >= passes_required else f"FAIL_{fails}OF{max_runs}:{fail_reasons[-1]}")
    finally:
        state.stress_seconds, state.multi_stress_seconds = orig_stress, orig_multi

def _merge_results(results: List[CandidateResult], ok: bool, reason: str) -> CandidateResult:
    temps = [r.telemetry_max_temp_c for r in results if r.telemetry_max_temp_c is not None]
    powers = [r.telemetry_max_power_w for r in results if r.telemetry_max_power_w is not None]
    throttles = [r.telemetry_any_throttle for r in results if r.telemetry_any_throttle is not None]
    codes = {}
    metrics = None
    for r in results:
        if r.stress_exit_codes:
            codes.update(r.stress_exit_codes)
        if r.metrics:
            metrics = r.metrics
    return CandidateResult(
        ok,
        reason,
        max(temps) if temps else None,
        max(powers) if powers else None,
        any(throttles) if throttles else None,
        codes or None,
        metrics,
    )


def _snap_down_to_stock_bin_khz(stock_points: List[CurvePoint], requested_khz: int) -> int:
    """
    Snap a requested frequency down to the nearest stock VF-bin frequency.

    If the request is below all known bins, clamp to the minimum stock bin.
    """
    bins = sorted({p.freq_khz for p in stock_points if p.freq_khz > 0})
    if not bins:
        return requested_khz
    lower_or_equal = [f for f in bins if f <= requested_khz]
    if lower_or_equal:
        return lower_or_equal[-1]
    return bins[0]


def _next_lower_stock_bin_khz(
    stock_points: List[CurvePoint],
    current_khz: int,
    min_khz: int,
) -> Optional[int]:
    """
    Return the next lower stock VF-bin frequency below current_khz, clamped to min_khz.
    """
    bins = sorted({p.freq_khz for p in stock_points if p.freq_khz > 0})
    lower_bins = [f for f in bins if min_khz <= f < current_khz]
    return lower_bins[-1] if lower_bins else None

def revert_to_last_good(state: SessionState) -> None:
    nvapi_apply_curve_safe(
        state.gpu,
        Path(state.last_good_curve_csv),
        timeout_seconds=12.0,
    )

def _check_for_manual_recovery(state: SessionState, label: str, manual_recovery_event: threading.Event) -> None:
    if not manual_recovery_event.is_set(): return
    manual_recovery_event.clear()
    eprint(f"\nManual recovery requested ({label}) — reverting...")
    try:
        revert_to_last_good(state)
        eprint("Revert complete.")
    except Exception as ex:
        eprint(f"WARNING: revert failed: {ex}")
    raise KeyboardInterrupt("Manual recovery hotkey")

def _mvscan_candidates_mvs(
    stock_points: List[CurvePoint],
    min_mv: int,
    max_mv: int,
) -> List[int]:
    lo = min(min_mv, max_mv)
    hi = max(min_mv, max_mv)
    bins = sorted(
        {
            int(round(p.voltage_uv / 1000.0))
            for p in stock_points
            if lo <= int(round(p.voltage_uv / 1000.0)) <= hi
        },
        reverse=True,
    )
    if not bins:
        raise ValueError(
            f"No stock VF bins found in range {lo}-{hi} mV. "
            "Adjust --bin-min-mv / --bin-max-mv."
        )
    return bins

def _build_mvscan_cap_curve(
    stock_points: List[CurvePoint],
    target_mv: int,
) -> Tuple[List[CurvePoint], int, int]:
    target_uv = mv_to_uv(target_mv)
    anchor_idx = min(range(len(stock_points)), key=lambda i: abs(stock_points[i].voltage_uv - target_uv))
    anchor_uv = stock_points[anchor_idx].voltage_uv
    anchor_freq_khz = stock_points[anchor_idx].freq_khz
    capped: List[CurvePoint] = []
    for p in stock_points:
        if p.voltage_uv >= anchor_uv:
            capped.append(CurvePoint(p.voltage_uv, anchor_freq_khz))
        else:
            capped.append(CurvePoint(p.voltage_uv, p.freq_khz))
    return capped, anchor_uv, anchor_freq_khz

def _mvscan_rank_key(row: dict, objective: str) -> Tuple[float, float, float, float, float]:
    p95 = float(row.get("p95_clock_mhz", 0.0) or 0.0)
    avg = float(row.get("avg_clock_mhz", 0.0) or 0.0)
    severe = float(row.get("throttle_severe_ratio_pct", 0.0) or 0.0)
    pwr = float(row.get("throttle_pwr_ratio_pct", 0.0) or 0.0)
    mv = float(row.get("target_mv", 0.0) or 0.0)
    if objective == "max-clock":
        return (p95, avg, -severe, -pwr, -mv)
    if objective == "min-cap":
        return (-severe, -pwr, p95, avg, -mv)
    score = p95 - (2.0 * severe) - (0.5 * pwr)
    return (score, p95, -severe, -pwr, -mv)

def _safe_metric(metrics: Optional[dict], key: str) -> float:
    if not metrics:
        return 0.0
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0

def _row_ok(row: dict) -> bool:
    v = row.get("ok")
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"

def run_mvscan_session(
    state: SessionState,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
) -> None:
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    last_good_csv = Path(state.last_good_curve_csv)
    candidates_mvs = _mvscan_candidates_mvs(stock_points, state.bin_min_mv, state.bin_max_mv)
    total = len(candidates_mvs)
    start_idx = max(0, min(state.current_step, total))

    print("\n=== VoltVandal - mvscan mode ===")
    print(
        f"  Objective: {state.mvscan_objective} | "
        f"Candidates: {total} bins (high->low) in {min(state.bin_min_mv, state.bin_max_mv)}-{max(state.bin_min_mv, state.bin_max_mv)} mV"
    )
    if start_idx > 0:
        print(f"  Resuming from candidate {start_idx + 1}/{total}")

    results_csv = out_dir / "mvscan_results.csv"
    steps_log = out_dir / "steps.jsonl"
    rows: List[dict] = []
    if start_idx > 0 and results_csv.exists():
        try:
            with results_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
    consecutive_failures = 0

    for idx in range(start_idx, total):
        target_mv = candidates_mvs[idx]
        label = f"mvscan_step{idx:03d}_{target_mv}mv"
        _check_for_manual_recovery(state, label, manual_recovery_event)
        if interrupted_event.is_set():
            eprint("\nInterrupted - reverting...")
            try:
                revert_to_last_good(state)
            except Exception as ex:
                eprint(f"WARNING: revert failed: {ex}")
            raise KeyboardInterrupt("User pressed Ctrl+C")

        curve_points, anchor_uv, anchor_freq_khz = _build_mvscan_cap_curve(stock_points, target_mv)
        candidate_csv = out_dir / f"{label}.csv"
        write_curve_csv(candidate_csv, curve_points)

        print(f"\n== {label} ==")
        print(f"  Cap: {anchor_uv//1000} mV | Plateau: {anchor_freq_khz//1000} MHz")
        result = evaluate_candidate_confident(
            state,
            candidate_csv,
            label,
            interrupted_event,
            manual_recovery_event,
            max_freq_mhz=anchor_freq_khz // 1000,
        )
        if result.ok:
            print("Result: PASS")
        else:
            print(f"Result: FAIL | {result.reason}")

        if result.ok:
            shutil.copyfile(candidate_csv, last_good_csv)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            revert_to_last_good(state)

        row = {
            "utc": now_utc_iso(),
            "step": idx,
            "target_mv": int(target_mv),
            "anchor_mv": int(anchor_uv // 1000),
            "plateau_mhz": int(anchor_freq_khz // 1000),
            "ok": bool(result.ok),
            "reason": result.reason,
            "avg_clock_mhz": _safe_metric(result.metrics, "avg_clock_mhz"),
            "p95_clock_mhz": _safe_metric(result.metrics, "p95_clock_mhz"),
            "max_clock_mhz": _safe_metric(result.metrics, "max_clock_mhz"),
            "throttle_any_ratio_pct": _safe_metric(result.metrics, "throttle_any_ratio_pct"),
            "throttle_pwr_ratio_pct": _safe_metric(result.metrics, "throttle_pwr_ratio_pct"),
            "throttle_severe_ratio_pct": _safe_metric(result.metrics, "throttle_severe_ratio_pct"),
            "sample_count": _safe_metric(result.metrics, "sample_count"),
            "telemetry_max_temp_c": result.telemetry_max_temp_c or "",
            "telemetry_max_power_w": result.telemetry_max_power_w or "",
        }
        rows.append(row)
        with steps_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(result), "utc": row["utc"], "label": label, "step": idx}) + "\n")

        state.current_step = idx + 1
        save_session(state)

        if not result.ok and "GPU_DRIVER_RESET_DETECTED" in result.reason:
            print("Stopping mvscan early: GPU driver reset detected.")
            break
        if consecutive_failures >= 3:
            print("Stopping mvscan early: 3 consecutive failures.")
            break

    fieldnames = [
        "utc",
        "step",
        "target_mv",
        "anchor_mv",
        "plateau_mhz",
        "ok",
        "reason",
        "avg_clock_mhz",
        "p95_clock_mhz",
        "max_clock_mhz",
        "throttle_any_ratio_pct",
        "throttle_pwr_ratio_pct",
        "throttle_severe_ratio_pct",
        "sample_count",
        "telemetry_max_temp_c",
        "telemetry_max_power_w",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    stable_rows = [r for r in rows if _row_ok(r)]
    if not stable_rows:
        print("\n=== mvscan complete: no stable candidates found (kept last good curve) ===")
        return

    best = max(stable_rows, key=lambda r: _mvscan_rank_key(r, state.mvscan_objective))
    best_idx = int(best["step"])
    best_mv = int(best["target_mv"])
    best_label = f"mvscan_step{best_idx:03d}_{best_mv}mv"
    best_csv = out_dir / f"{best_label}.csv"
    if best_csv.exists():
        shutil.copyfile(best_csv, last_good_csv)
        try:
            nvapi_apply_curve_safe(state.gpu, best_csv, timeout_seconds=12.0)
        except Exception:
            pass
        shutil.copyfile(best_csv, out_dir / "mvscan_best_curve.csv")

    print("\n=== mvscan complete ===")
    print(
        f"Best candidate: {best_mv} mV | "
        f"P95 {best['p95_clock_mhz']:.0f} MHz | "
        f"SevereCap {best['throttle_severe_ratio_pct']:.1f}% | "
        f"PwrCap {best['throttle_pwr_ratio_pct']:.1f}%"
    )
    print(f"Results saved: {results_csv}")

def run_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    last_good_csv = Path(state.last_good_curve_csv)

    print(f"Starting from step {state.current_step}/{state.max_steps} mode={state.mode}")
    while state.current_step < state.max_steps:
        _check_for_manual_recovery(state, "main tuning loop", manual_recovery_event)
        if interrupted_event.is_set():
            eprint("\nInterrupted — reverting...")
            try: revert_to_last_good(state)
            except Exception as ex: eprint(f"WARNING: revert failed: {ex}")
            raise KeyboardInterrupt("User pressed Ctrl+C")

        step = state.current_step + 1
        if state.mode == "uv": offset_mv, offset_mhz = -(state.step_mv * step), 0
        elif state.mode == "oc": offset_mv, offset_mhz = 0, state.step_mhz * step
        elif state.mode == "hybrid":
            phase = getattr(state, "hybrid_phase", "uv")
            if phase == "uv": offset_mv, offset_mhz = -(state.step_mv * step), 0
            else:
                offset_mv = getattr(state, "hybrid_locked_mv", 0)
                oc_step = step - getattr(state, "hybrid_oc_start_step", 0)
                offset_mhz = state.step_mhz * oc_step
        else: raise ValueError(f"Unknown mode: {state.mode}")

        label = f"step{step:03d}_mv{offset_mv}_mhz{offset_mhz}"
        candidate_csv = out_dir / "candidate.csv"
        points_candidate = apply_offsets_to_bin(stock_points, state.bin_min_mv, state.bin_max_mv, offset_mv, offset_mhz)
        write_curve_csv(candidate_csv, points_candidate)

        print(f"\n== Candidate {label} ==\n  Offset: {offset_mv} mV, {offset_mhz} MHz")
        result = evaluate_candidate(state, candidate_csv, label, interrupted_event, manual_recovery_event)
        if result.ok:
            print("Result: PASS")
        else:
            print(f"Result: FAIL | {result.reason}")
        _check_for_manual_recovery(state, label, manual_recovery_event)

        steps_log = out_dir / "steps.jsonl"
        with steps_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(result), "utc": now_utc_iso(), "label": label, "step": step}) + "\n")

        if result.ok:
            shutil.copyfile(candidate_csv, last_good_csv)
            state.current_step, state.current_offset_mv, state.current_offset_mhz = step, offset_mv, offset_mhz
            save_session(state)
        else:
            revert_to_last_good(state)
            if state.mode == "hybrid" and state.hybrid_phase == "uv":
                state.hybrid_phase, state.hybrid_locked_mv, state.hybrid_oc_start_step = "oc", state.current_offset_mv, step
                state.current_step = step - 1
                save_session(state)
                continue
            state.current_step = step - 1
            save_session(state)
            break

def run_vlock_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    # Ported from voltvandal.py with necessary adjustments
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    target_uv = mv_to_uv(state.vlock_target_mv)
    anchor_idx = min(range(len(stock_points)), key=lambda i: abs(stock_points[i].voltage_uv - target_uv))
    anchor_v_uv = stock_points[anchor_idx].voltage_uv
    anchor_stock_f_khz = stock_points[anchor_idx].freq_khz
    oc_base_freq_khz = state.vlock_oc_base_freq_khz or anchor_stock_f_khz
    requested_start_khz = mhz_to_khz(state.vlock_start_freq_mhz) if state.vlock_start_freq_mhz > 0 else oc_base_freq_khz
    oc_start_freq_khz = _snap_down_to_stock_bin_khz(stock_points, requested_start_khz)
    step_khz = mhz_to_khz(state.step_mhz)
    coarse_mult = 2
    if state.vlock_anchor_freq_khz <= 0:
        state.vlock_anchor_freq_khz = anchor_stock_f_khz

    print(f"\n=== VoltVandal — vlock mode ===\n  Anchor: {anchor_v_uv//1000} mV | Stock: {anchor_stock_f_khz//1000} MHz")
    if state.vlock_start_freq_mhz > 0:
        if oc_start_freq_khz != requested_start_khz:
            print(
                f"  Phase 1 start frequency adjust: {requested_start_khz//1000} MHz "
                f"-> {oc_start_freq_khz//1000} MHz "
                "(auto adjusted to closest round tuning bin)."
            )
        else:
            print(f"  Phase 1 start frequency override: {oc_start_freq_khz//1000} MHz")
    
    # Phase 1: OC search
    if state.vlock_phase == "oc":
        while state.vlock_phase == "oc":
            _check_for_manual_recovery(state, "vlock_p1", manual_recovery_event)
            step = state.current_step
            if step > state.max_steps:
                state.vlock_phase = "uv"
                state.vlock_uv_bin_idx = anchor_idx - 1
                state.current_step = 0
                state.vlock_last_fail_step = -1
                save_session(state)
                break

            cand_freq = oc_start_freq_khz + mhz_to_khz(state.step_mhz * step)
            cand_pts = _build_vlock_curve(stock_points, anchor_idx, anchor_v_uv, cand_freq, 0)
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, cand_pts)
            
            _p1_mode = "coarse" if state.vlock_last_fail_step < 0 else "fine"
            label = f"vlock_p1_step{step:03d}_{cand_freq//1000}mhz_{_p1_mode}"
            print(f"\n== {label} ==")
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, max_freq_mhz=cand_freq//1000)
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
            
            if result.ok:
                shutil.copyfile(cand_csv, Path(state.last_good_curve_csv))
                state.vlock_anchor_freq_khz = cand_freq
                if state.vlock_last_fail_step >= 0:
                    next_step = step + 1
                    if next_step >= state.vlock_last_fail_step:
                        state.vlock_phase = "uv"
                        state.vlock_uv_bin_idx = anchor_idx - 1
                        state.current_step = 0
                        state.vlock_last_fail_step = -1
                    else:
                        state.current_step = next_step
                else:
                    state.current_step = step + coarse_mult
                save_session(state)
            else:
                revert_to_last_good(state)
                if state.vlock_last_fail_step >= 0:
                    state.vlock_phase = "uv"
                    state.vlock_uv_bin_idx = anchor_idx - 1
                    state.current_step = 0
                    state.vlock_last_fail_step = -1
                    save_session(state)
                    break

                prev_coarse_step = max(0, step - coarse_mult)
                fine_start_step = prev_coarse_step + 1
                if fine_start_step >= step:
                    failed_freq_khz = cand_freq
                    lowered_start_khz = _next_lower_stock_bin_khz(
                        stock_points,
                        cand_freq,
                        anchor_stock_f_khz,
                    )
                    if lowered_start_khz is not None:
                        print(
                            f"Phase 1 start {cand_freq//1000} MHz failed immediately; "
                            f"lowering start to {lowered_start_khz//1000} MHz and retrying."
                        )
                        oc_start_freq_khz = lowered_start_khz
                        state.vlock_start_freq_mhz = lowered_start_khz // 1000
                        # Keep subsequent search below the original immediate-fail
                        # ceiling to avoid jumping to higher coarse points.
                        delta_khz = max(0, failed_freq_khz - lowered_start_khz)
                        step_at_failed = (delta_khz + step_khz - 1) // step_khz
                        state.vlock_last_fail_step = max(1, int(step_at_failed + 1))
                        state.current_step = 0
                        state.vlock_anchor_freq_khz = anchor_stock_f_khz
                        save_session(state)
                        continue
                    state.vlock_phase = "uv"
                    state.vlock_uv_bin_idx = anchor_idx - 1
                    state.current_step = 0
                    state.vlock_last_fail_step = -1
                    save_session(state)
                    break

                state.vlock_last_fail_step = step
                state.current_step = fine_start_step
                _lo_freq = oc_start_freq_khz + mhz_to_khz(state.step_mhz * prev_coarse_step)
                _hi_freq = cand_freq
                print(f"Refining Phase 1 between {_lo_freq//1000} and {_hi_freq//1000} MHz using {state.step_mhz} MHz steps.")
                save_session(state)
    
    # Phase 2: UV/OC Shift Sweep
    if state.vlock_phase == "uv":
        oc_gain = state.vlock_anchor_freq_khz - anchor_stock_f_khz
        uv_failed = False
        while state.vlock_uv_bin_idx >= 0:
            bin_idx = state.vlock_uv_bin_idx
            _check_for_manual_recovery(state, f"vlock_p2_bin{bin_idx}", manual_recovery_event)
            
            last_good_pts = load_curve_csv(Path(state.last_good_curve_csv))
            test_pts, save_pts = _build_vlock_phase2_curves(stock_points, last_good_pts, bin_idx, anchor_idx, anchor_v_uv, state.vlock_anchor_freq_khz, oc_gain)
            
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, test_pts)
            
            label = f"vlock_p2_bin{bin_idx:03d}_{stock_points[bin_idx].voltage_uv//1000}mv"
            print(f"\n== {label} ==")
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event)
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
            
            if result.ok:
                write_curve_csv(Path(state.last_good_curve_csv), save_pts)
            else:
                revert_to_last_good(state)
                uv_failed = True
                save_session(state)
                break
            
            state.vlock_uv_bin_idx -= 1
            save_session(state)

        if not uv_failed and state.vlock_uv_bin_idx < 0:
            state.vlock_phase = "done"
            save_session(state)
            print("\n=== vlock tuning complete ===")
        elif uv_failed:
            print("\n=== vlock tuning stopped on failure (reverted to last good curve) ===")
