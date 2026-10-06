import json
import re
import csv
import shutil
import sys
import threading
import time
from contextlib import nullcontext
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
from .planner import lower_bin_from_metrics
from . import flightlog, faultmap
from .anchors import (
    plateau_starts, pick_anchors, search_safe_gain, anchor_ceiling, proven_gains_from_flightlog,
    build_curve, SAFE_MARGIN_STEPS,
)
from ..hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from ..hardware.monitor import NvmlMonitor
from ..hardware.point_lock import (
    PointLockError, temporary_point_lock, inspect_point_lock, check_monitor_identity,
)
from ..hardware.events import hardware_errors_since, FaultWatch
from ..stress.runner import run_doloming, run_gpuburn, terminate_all_active_processes

_WARMUP_MIN_SECONDS = 30
_WARMUP_MAX_SECONDS = 60
_AUTO_PLAN_MODES = "simple,matrix,ray,frequency-max"
_AUTO_PLAN_PROBE_SECONDS = 20
_AUTO_PLAN_FINAL_SECONDS = 120
_POST_APPLY_QUIET_SECONDS = 3.0


def _run_auto_plan_suite(
    state: SessionState,
    curve_csv: Path,
    label: str,
    seconds_per_mode: int,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    include_gpuburn: bool = False,
    modes: str = _AUTO_PLAN_MODES,
) -> CandidateResult:
    original = (
        state.doloming_mode, state.doloming_modes, state.stress_seconds,
        state.multi_stress_seconds, state.gpuburn,
    )
    # Probe/final suites measure natural loaded voltage, so they run unlocked.
    original_point_lock = state.point_lock
    state.point_lock = False
    try:
        selected = [mode.strip() for mode in modes.split(",") if mode.strip()]
        if len(selected) > 1:
            state.doloming_modes = ",".join(selected)
            state.multi_stress_seconds = seconds_per_mode
        else:
            state.doloming_mode = selected[0]
            state.doloming_modes = ""
            state.stress_seconds = seconds_per_mode
        state.gpuburn = original[4] if include_gpuburn else None
        if include_gpuburn:
            state.stress_seconds = max(state.stress_seconds, seconds_per_mode)
        return evaluate_candidate(
            state, curve_csv, label, interrupted_event, manual_recovery_event
        )
    finally:
        state.point_lock = original_point_lock
        (state.doloming_mode, state.doloming_modes, state.stress_seconds,
         state.multi_stress_seconds, state.gpuburn) = original


def _update_auto_plan_bound(
    state: SessionState,
    metrics: Optional[dict],
    stock_points: List[CurvePoint],
    anchor_idx: int,
) -> bool:
    bound = lower_bin_from_metrics(metrics, stock_points, anchor_idx)
    if bound is None:
        state.auto_plan_fallback_full = True
        state.auto_plan_min_bin_idx = 0
        print("Measured loaded-voltage coverage unavailable; using the full lower-bin sweep.")
        return False
    if state.auto_plan_min_bin_idx < 0 or bound < state.auto_plan_min_bin_idx:
        state.auto_plan_min_bin_idx = bound
        print(f"Automatic sweep now extends down to {stock_points[bound].voltage_uv / 1000:.2f} mV.")
        return True
    return False

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
    if re.search(r"cuda_?error\w*|illegal memory access|device-side assert|unspecified launch failure|launch timeout|driver shutting down", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_CUDA_RUNTIME_ERROR"
    if re.search(r"Error during (?:stress )?test:", scan_text, re.I):
        return False, f"DOLOMING_{mode_tag}_STRESS_ERROR"

    status = _extract_summary_value(scan_text, "Status")
    if status and re.search(r"\b(unstable|fail(?:ed|ure)?|error)\b", status, re.I):
        return False, f"DOLOMING_{mode_tag}_UNSTABLE_STATUS"
    if re.search(r"failed to fully stabilize", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_FAILED_TO_STABILIZE"
    return True, None

def effective_point(points: List[CurvePoint], target: CurvePoint) -> CurvePoint:
    """Resolve a target bin to the lowest-voltage bin sharing its clock.

    The driver runs a flat stretch of the curve at its first (lowest-voltage)
    bin, so higher bins on the plateau can never be observed on their own.
    """
    same = [p.voltage_uv for p in points
            if p.freq_khz == target.freq_khz and p.voltage_uv <= target.voltage_uv]
    return CurvePoint(min(same), target.freq_khz) if same else target

_active_watch: Optional[FaultWatch] = None
_HARD_FAULT_PREFIXES = ("APPLY_FAILED", "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT",
                        "HARDWARE_ERROR_EVENTS", "GPU_FAULT_EVENT")


def _emergency_stop(state: SessionState, label: str, faults: List[str]) -> None:
    """GPU fault event seen: kill the load and drop back to the last good curve at once."""
    flightlog.log("gpu_fault_event", label=label, events=faults)
    eprint(f"\nGPU FAULT EVENT ({', '.join(faults)}) - stopping load and reverting.")
    terminate_all_active_processes()
    try:
        revert_to_last_good(state)
        flightlog.log("emergency_revert_done", label=label)
    except Exception as ex:
        flightlog.log("emergency_revert_failed", label=label, error=str(ex))
        eprint(f"WARNING: emergency revert failed: {ex}")


def evaluate_candidate(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
    target_point: Optional[CurvePoint] = None,
) -> CandidateResult:
    """Run one candidate under a GPU-fault watchdog with a write-ahead crash journal."""
    global _active_watch
    out_dir = Path(state.out_dir)
    if target_point is not None:
        faultmap.mark_in_flight(out_dir, candidate_label, target_point.voltage_uv, target_point.freq_khz)
    watch = FaultWatch(lambda faults: _emergency_stop(state, candidate_label, faults)).start()
    _active_watch = watch
    try:
        result = _evaluate_candidate_inner(
            state, candidate_csv, candidate_label, interrupted_event,
            manual_recovery_event, max_freq_mhz, target_point)
    finally:
        watch.stop()
        _active_watch = None
        if target_point is not None:
            if watch.faults:
                faultmap.record_fault(out_dir, target_point.voltage_uv, target_point.freq_khz)
            faultmap.clear_in_flight(out_dir)
    if watch.faults:
        result.ok = False
        result.reason = f"GPU_FAULT_EVENT: {', '.join(watch.faults)}"
    return result


def _evaluate_candidate_inner(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
    target_point: Optional[CurvePoint] = None,
) -> CandidateResult:
    targeted = bool(state.point_lock and target_point is not None)
    journal = Path(state.out_dir) / "point_lock_recovery.json"
    if journal.exists():
        raise PointLockError(f"Recover the previous point lock before tuning: {journal}")
    if state.point_lock and target_point is None:
        raise PointLockError("Point-lock testing requires an explicit target curve point")
    if target_point is not None:
        curve_points = load_curve_csv(candidate_csv)
        if targeted and target_point not in curve_points:
            raise PointLockError("Target point does not match the candidate curve")
        target_point = effective_point(curve_points, target_point)
    if targeted:
        info = inspect_point_lock(state.gpu)
        check_monitor_identity(state.gpu, info["bus"])
    started_local = datetime.now()
    try:
        changed = [(p.voltage_uv, p.freq_khz) for p in load_curve_csv(candidate_csv)]
    except OSError:
        changed = []
    flightlog.log("candidate_begin", label=candidate_label, target=target_point and (target_point.voltage_uv, target_point.freq_khz),
                  max_freq_mhz=max_freq_mhz, util_pct=state.point_util_pct, curve=changed)
    try:
        flightlog.log("curve_apply_start", label=candidate_label)
        nvapi_apply_curve_safe(state.gpu, candidate_csv, timeout_seconds=12.0)
        flightlog.log("curve_apply_done", label=candidate_label)
        interrupted_event.wait(timeout=2.0)
        if interrupted_event.is_set():
            raise KeyboardInterrupt("User pressed Ctrl+C")
        if _active_watch is not None and _active_watch.tripped.wait(timeout=_POST_APPLY_QUIET_SECONDS):
            return CandidateResult(False, "GPU_FAULT_EVENT_AFTER_APPLY")
    except Exception as ex:
        flightlog.log("curve_apply_failed", label=candidate_label, error=str(ex))
        if targeted:
            raise PointLockError(f"Point test curve application failed: {ex}") from ex
        return CandidateResult(False, f"APPLY_FAILED: {ex}")

    monitors = []
    lock = (temporary_point_lock(state.gpu, target_point.voltage_uv, journal)
            if targeted else nullcontext(None))
    with lock as bus:
        flightlog.log("stress_begin", label=candidate_label, point_lock_bus=bus)
        try:
            result = _evaluate_applied_candidate(
                state, candidate_csv, candidate_label, interrupted_event,
                manual_recovery_event, max_freq_mhz, target_point,
                bus, monitors,
            )
        finally:
            for monitor in monitors:
                monitor.stop()
    flightlog.log("candidate_result", label=candidate_label, ok=result.ok, reason=result.reason)
    if result.ok:
        # Silent instability: WHEA / driver / TDR events logged during a "passing" test.
        events = hardware_errors_since(started_local)
        if events:
            result.ok = False
            result.reason = f"HARDWARE_ERROR_EVENTS: {', '.join(sorted(set(events)))}"
    return result


def _evaluate_applied_candidate(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int,
    target_point: Optional[CurvePoint],
    point_lock_bus: Optional[int],
    monitors: list,
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

    point_options = {}
    if target_point is not None:
        distances = [abs(p.voltage_uv - target_point.voltage_uv) / 1000.0
                     for p in load_curve_csv(candidate_csv)
                     if p.voltage_uv != target_point.voltage_uv]
        point_options = dict(
            point_lock_bus=point_lock_bus,
            point_voltage_mv=target_point.voltage_uv / 1000.0,
            point_clock_mhz=target_point.freq_khz / 1000.0,
            point_voltage_tolerance_mv=min(10.0, min(distances) * 0.49) if distances else 3.0,
        )
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
            measure_voltage=True,
            **point_options,
        )
        monitors.append(monitor)
        monitor.start()
        abort_event = monitor.abort_event
    except Exception as e:
        eprint(f"Failed to start monitor: {e}")
        if target_point is not None:
            raise PointLockError(f"Point test monitor unavailable: {e}") from e
        return _build_result(False, f"MONITOR_START_FAILED: {e}", None, None)

    stress_exit_codes = {}
    def monitor_abort_result() -> Optional[CandidateResult]:
        if not monitor.abort_event.is_set():
            return None
        if monitor.driver_reset_detected:
            reason = "GPU_DRIVER_RESET_DETECTED"
        elif monitor.abort_reason:
            reason = f"MONITOR_ABORT_THRESHOLD:{monitor.abort_reason}"
        else:
            reason = "MONITOR_ABORT_UNKNOWN"
        return _build_result(False, reason, monitor, stress_exit_codes)

    def check_point_coverage(marker, workload) -> Optional[CandidateResult]:
        if target_point is None:
            return
        coverage = monitor.point_coverage(marker)
        with (logs_dir / f"{candidate_label}_point.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"workload": workload, **coverage}) + "\n")
        if coverage["matched_samples"] < 3 or coverage["coverage_pct"] < 80.0:
            # An unexercised point is not a stability boundary. Stop without
            # advancing the search/checkpoint or promoting the candidate.
            if point_lock_bus is not None:
                raise PointLockError(f"INCONCLUSIVE_POINT_COVERAGE: {workload}: {coverage}")
            return _build_result(
                False,
                f"INCONCLUSIVE_BIN_COVERAGE: {workload}: "
                f"{coverage['matched_samples']}/{coverage['loaded_samples']} loaded samples "
                f"at {coverage['target_voltage_mv']:.2f} mV / "
                f"{coverage['target_clock_mhz']:.0f} MHz",
                monitor,
                stress_exit_codes,
            )
        return None

    if state.doloming:
        _dolo_timeout = state.stress_timeout if state.stress_timeout is not None else max(_secs_each * 5, 300)
        for _mode in _run_modes:
            marker = monitor.point_marker() if target_point else None
            dololog = logs_dir / f"{candidate_label}_doloming_{_mode}.log"
            monitor.arm_collapse_check()
            try:
                rc, out_text = run_doloming(
                    state.doloming, state.gpu, _mode, _secs_each, None, dololog,
                    abort_event, manual_recovery_event, interrupted_event,
                    stress_timeout=_dolo_timeout, max_freq_mhz=max_freq_mhz,
                    util_pct=state.point_util_pct if target_point is not None else 0,
                )
            finally:
                monitor.disarm_collapse_check()
            _key = f"doloming_{_mode}" if _use_multi else "doloming"
            stress_exit_codes[_key] = rc
            _stable, _instability_reason = _parse_doloming_stability(out_text, _mode)
            if _instability_reason and _instability_reason.endswith("_CUDA_RUNTIME_ERROR"):
                return _build_result(False, _instability_reason, monitor, stress_exit_codes)
            aborted = monitor_abort_result()
            if aborted is not None:
                return aborted
            if rc == 999:
                return _build_result(False, "MONITOR_ABORT_UNKNOWN", monitor, stress_exit_codes)
            if point_lock_bus is not None and (rc in (996, 998) or monitor.abort_reason == "POINT_VOLTAGE_UNAVAILABLE"):
                raise PointLockError(f"INCONCLUSIVE_POINT_TEST: {_mode}, rc={rc}, {monitor.abort_reason}")
            if rc != 0:
                _reason = "MANUAL_RECOVERY_REQUESTED" if rc == 997 else f"DOLOMING_{_mode.upper().replace('-', '_')}_RC_{rc}"
                return _build_result(False, _reason, monitor, stress_exit_codes)
            if not _stable:
                return _build_result(False, _instability_reason, monitor, stress_exit_codes)
            uncovered = check_point_coverage(marker, _mode)
            if uncovered is not None:
                return uncovered

    if state.gpuburn:
        marker = monitor.point_marker() if target_point else None
        burnlog = logs_dir / f"{candidate_label}_gpuburn.log"
        _burn_timeout = state.stress_timeout if state.stress_timeout is not None else state.stress_seconds * 5
        monitor.arm_collapse_check()
        try:
            rc, out_text, parsed_ok = run_gpuburn(
                state.gpuburn, state.stress_seconds, None, burnlog,
                abort_event, manual_recovery_event, interrupted_event,
                stress_timeout=_burn_timeout,
            )
        finally:
            monitor.disarm_collapse_check()
        stress_exit_codes["gpuburn"] = rc
        aborted = monitor_abort_result()
        if aborted is not None:
            return aborted
        if rc == 999:
            return _build_result(False, "MONITOR_ABORT_UNKNOWN", monitor, stress_exit_codes)
        if point_lock_bus is not None and (rc in (996, 998) or monitor.abort_reason == "POINT_VOLTAGE_UNAVAILABLE"):
            raise PointLockError(f"INCONCLUSIVE_POINT_TEST: gpuburn, rc={rc}, {monitor.abort_reason}")
        if rc != 0 or not parsed_ok or re.search(r"\bnan\b|\bfailed\b|errors?\s*[:=]\s*[1-9][0-9]*", out_text, re.I):
            _reason = f"GPUBURN_RC_{rc}" if rc != 0 else ("GPUBURN_ERRORS_DETECTED" if not parsed_ok else "GPUBURN_OUTPUT_ERROR_KEYWORD")
            return _build_result(False, _reason, monitor, stress_exit_codes)
        uncovered = check_point_coverage(marker, "gpuburn")
        if uncovered is not None:
            return uncovered

    if target_point is not None and not stress_exit_codes:
        if point_lock_bus is not None:
            raise PointLockError("No workload ran for this point test")
        return _build_result(False, "INCONCLUSIVE_BIN_COVERAGE: no workload ran", monitor)

    aborted = monitor_abort_result()
    if aborted is not None:
        return aborted

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
    target_point: Optional[CurvePoint] = None,
) -> CandidateResult:
    all_results: List[CandidateResult] = []
    hard_fail_prefixes = ("APPLY_FAILED", "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT", "DOLOMING_", "GPUBURN_", "HARDWARE_ERROR_EVENTS", "INCONCLUSIVE_BIN_COVERAGE")

    def _is_hard_fail(reason: str) -> bool:
        return any(reason.startswith(p) for p in hard_fail_prefixes) or "CUDA_ERROR" in reason.upper()

    orig_stress = state.stress_seconds
    orig_multi = state.multi_stress_seconds
    point_options = {"target_point": target_point} if target_point is not None else {}
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
            w = evaluate_candidate(state, candidate_csv, f"{candidate_label}_warmup", interrupted_event, manual_recovery_event, max_freq_mhz, **point_options)
            all_results.append(w)
            if not w.ok and _is_hard_fail(w.reason): return _merge_results(all_results, False, f"WARMUP_HARD_FAIL:{w.reason}")

        state.stress_seconds, state.multi_stress_seconds = orig_stress, orig_multi
        passes = 0
        fails = 0
        fail_reasons = []
        for i in range(1, max_runs + 1):
            r = evaluate_candidate(state, candidate_csv, f"{candidate_label}_run{i}", interrupted_event, manual_recovery_event, max_freq_mhz, **point_options)
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
            if _HARD_FAULT_RE.search(result.reason):
                hard_fault.append(result.reason)

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
        f"P95 {float(best['p95_clock_mhz']):.0f} MHz | "
        f"SevereCap {float(best['throttle_severe_ratio_pct']):.1f}% | "
        f"PwrCap {float(best['throttle_pwr_ratio_pct']):.1f}%"
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
    if state.active_candidate_label:
        raise RuntimeError(f"Previous run ended during {state.active_candidate_label}; refusing automatic retry")
    if state.vlock_phase == "failed":
        raise RuntimeError("Vlock session failed; start a new session after reviewing the failure")
    if state.vlock_phase == "inconclusive":
        raise RuntimeError("Vlock lower-bin coverage was inconclusive; start a new session")
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

    if state.auto_plan and not state.auto_plan_probe_done:
        if state.vlock_phase != "oc" or state.current_step != 0:
            state.auto_plan_fallback_full = True
            state.auto_plan_min_bin_idx = 0
            state.auto_plan_modes = state.doloming_modes or state.doloming_mode
            print("Existing vlock progress has no stock probe; using the full sweep.")
        else:
            print("\n== Automatic stock-curve coverage probe ==")
            state.active_candidate_label = "vlock_plan_stock_probe"
            save_session(state)
            probe = _run_auto_plan_suite(
                state, Path(state.stock_curve_csv), "vlock_plan_stock_probe",
                _AUTO_PLAN_PROBE_SECONDS, interrupted_event, manual_recovery_event,
            )
            if not probe.ok:
                state.vlock_phase = "failed"
                save_session(state)
                raise RuntimeError(f"Stock coverage probe stopped: {probe.reason}")
            state.active_candidate_label = ""
            state.auto_plan_modes = _AUTO_PLAN_MODES
            _update_auto_plan_bound(state, probe.metrics, stock_points, anchor_idx)
        state.auto_plan_probe_done = True
        save_session(state)
    
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
            point_options = {"target_point": CurvePoint(anchor_v_uv, cand_freq)} if state.point_lock else {}
            state.active_candidate_label = label
            save_session(state)
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, max_freq_mhz=cand_freq//1000, **point_options)
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
                if any(token in result.reason for token in (
                    "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT", "CUDA_RUNTIME_ERROR", "STRESS_ERROR"
                )):
                    state.vlock_phase = "failed"
                    save_session(state)
                    raise RuntimeError(f"Vlock tuning stopped: {result.reason}")
            state.active_candidate_label = ""
            save_session(state)
            
            if result.ok:
                shutil.copyfile(cand_csv, Path(state.last_good_curve_csv))
                state.vlock_anchor_freq_khz = cand_freq
                if state.auto_plan and not state.auto_plan_fallback_full and not state.point_lock:
                    _update_auto_plan_bound(state, result.metrics, stock_points, anchor_idx)
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
        while True:
            bin_idx = state.vlock_uv_bin_idx
            plan_boundary_reached = (
                state.auto_plan and not state.auto_plan_fallback_full
                and bin_idx < state.auto_plan_min_bin_idx
            )
            if bin_idx < 0 or plan_boundary_reached:
                if state.auto_plan:
                    print("\n== Automatic final-curve validation ==")
                    state.active_candidate_label = "vlock_plan_final"
                    save_session(state)
                    final = _run_auto_plan_suite(
                        state, Path(state.last_good_curve_csv), "vlock_plan_final",
                        _AUTO_PLAN_FINAL_SECONDS, interrupted_event, manual_recovery_event,
                        include_gpuburn=True,
                        modes=state.auto_plan_modes or _AUTO_PLAN_MODES,
                    )
                    if not final.ok:
                        nvapi_apply_curve_safe(state.gpu, Path(state.stock_curve_csv), timeout_seconds=12.0)
                        shutil.copyfile(state.stock_curve_csv, state.last_good_curve_csv)
                        state.vlock_phase = "failed"
                        state.active_candidate_label = ""
                        save_session(state)
                        print(f"Final curve failed ({final.reason}); stock curve restored.")
                        uv_failed = True
                        break
                    state.active_candidate_label = ""
                    save_session(state)
                    if not state.auto_plan_fallback_full:
                        old_bound = state.auto_plan_min_bin_idx
                        _update_auto_plan_bound(state, final.metrics, stock_points, anchor_idx)
                        if state.auto_plan_fallback_full:
                            if bin_idx < 0:
                                state.vlock_uv_bin_idx = anchor_idx - 1
                            save_session(state)
                            continue
                        if state.auto_plan_min_bin_idx < old_bound:
                            save_session(state)
                            continue
                state.vlock_phase = "done"
                save_session(state)
                print("\n=== vlock tuning complete ===")
                break
            _check_for_manual_recovery(state, f"vlock_p2_bin{bin_idx}", manual_recovery_event)
            
            last_good_pts = load_curve_csv(Path(state.last_good_curve_csv))
            test_pts, save_pts = _build_vlock_phase2_curves(stock_points, last_good_pts, bin_idx, anchor_idx, anchor_v_uv, state.vlock_anchor_freq_khz, oc_gain)
            
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, test_pts)
            
            label = f"vlock_p2_bin{bin_idx:03d}_{stock_points[bin_idx].voltage_uv//1000}mv"
            print(f"\n== {label} ==")
            point_options = {"target_point": test_pts[bin_idx]}
            state.active_candidate_label = label
            save_session(state)
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, **point_options)
            if not result.ok and "INCONCLUSIVE_BIN_COVERAGE" in result.reason:
                print(f"Result: INCONCLUSIVE | {result.reason}")
                revert_to_last_good(state)
                state.vlock_phase = "inconclusive"
                state.active_candidate_label = ""
                save_session(state)
                print("Lower-bin sweep stopped; the unexercised candidate was not saved.")
                return
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
                if any(token in result.reason for token in (
                    "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT", "CUDA_RUNTIME_ERROR", "STRESS_ERROR"
                )):
                    state.vlock_phase = "failed"
                    save_session(state)
                    raise RuntimeError(f"Vlock tuning stopped: {result.reason}")
            state.active_candidate_label = ""
            save_session(state)
            
            if result.ok:
                write_curve_csv(Path(state.last_good_curve_csv), save_pts)
                if state.auto_plan and not state.auto_plan_fallback_full and not state.point_lock:
                    _update_auto_plan_bound(state, result.metrics, stock_points, anchor_idx)
            else:
                revert_to_last_good(state)
                uv_failed = True
                save_session(state)
                break
            
            state.vlock_uv_bin_idx -= 1
            save_session(state)

        if uv_failed and state.vlock_phase != "failed":
            print("\n=== vlock tuning stopped on failure (reverted to last good curve) ===")


_ANCHOR_VERIFY_RETRIES = 3
_ANCHOR_START_MARGIN_STEPS = 2  # back off the found gains by this many steps before verifying
_ANCHOR_VERIFY_PASSES = 2
_ANCHOR_UTIL_STEP = 10
_ANCHOR_MIN_UTIL_PCT = 35
_DEFAULT_ANCHOR_CAP_MHZ = 150  # conservative ceiling when neither the profile nor --max-gain-mhz sets one
_HARD_FAULT_RE = re.compile(r"HARD_FAIL|FATAL|cuda|illegal|device-side|launch|driver|reset|tdr", re.I)
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
