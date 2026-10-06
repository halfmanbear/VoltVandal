import threading
from pathlib import Path
from typing import List, Optional

from ..models import CandidateResult, CurvePoint, SessionState
from .evaluate import evaluate_candidate

_WARMUP_MIN_SECONDS = 30
_WARMUP_MAX_SECONDS = 60


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
