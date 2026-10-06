import json
import re
import threading
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models import SessionState, CandidateResult, CurvePoint
from ..utils import eprint
from ..curve import load_curve_csv
from .. import flightlog, faultmap
from ...hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from ...hardware.monitor import NvmlMonitor
from ...hardware.point_lock import (
    PointLockError, temporary_point_lock, inspect_point_lock, check_monitor_identity,
)
from ...hardware.events import hardware_errors_since, FaultWatch
from ...stress.runner import run_doloming, run_gpuburn, terminate_all_active_processes
from .recovery import revert_to_last_good
from .stability import _parse_doloming_stability

_POST_APPLY_QUIET_SECONDS = 3.0


def effective_point(points: List[CurvePoint], target: CurvePoint) -> CurvePoint:
    """Resolve a target bin to the lowest-voltage bin sharing its clock.

    The driver runs a flat stretch of the curve at its first (lowest-voltage)
    bin, so higher bins on the plateau can never be observed on their own.
    """
    same = [p.voltage_uv for p in points
            if p.freq_khz == target.freq_khz and p.voltage_uv <= target.voltage_uv]
    return CurvePoint(min(same), target.freq_khz) if same else target

_active_watch: Optional[FaultWatch] = None


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
