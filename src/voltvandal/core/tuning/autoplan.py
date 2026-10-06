import threading
from pathlib import Path
from typing import List, Optional

from ..models import SessionState, CandidateResult, CurvePoint
from ..planner import lower_bin_from_metrics
from .evaluate import evaluate_candidate


_WARMUP_MIN_SECONDS = 30
_WARMUP_MAX_SECONDS = 60
_AUTO_PLAN_MODES = "simple,matrix,ray,frequency-max"
_AUTO_PLAN_PROBE_SECONDS = 20
_AUTO_PLAN_FINAL_SECONDS = 120


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
