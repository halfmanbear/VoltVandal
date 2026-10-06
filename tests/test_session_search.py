import threading
from pathlib import Path
from unittest.mock import patch

from voltvandal.core.models import CandidateResult
from voltvandal.core.tuning import evaluate_candidate_confident, run_session


@patch("voltvandal.core.tuning.linear.evaluate_candidate")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
def test_run_session_uv_progression(mock_apply, mock_eval, mock_state):
    # Step 1 pass, Step 2 fail
    mock_eval.side_effect = [
        CandidateResult(True, "PASS", 70, 200, False, {}),
        CandidateResult(False, "FAIL", 75, 210, False, {}),
    ]
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    run_session(mock_state, interrupted, recovery)
    
    assert mock_state.current_step == 1
    assert mock_state.current_offset_mv == -5

@patch("voltvandal.core.tuning.linear.evaluate_candidate")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
def test_run_session_hybrid_transition(mock_apply, mock_eval, mock_state):
    mock_state.mode = "hybrid"
    mock_state.max_steps = 3
    # UV Step 1 fails immediately -> transitions to OC
    # OC Step 1 (baseline) passes
    # OC Step 2 passes
    # OC Step 3 fails
    mock_eval.side_effect = [
        CandidateResult(False, "FAIL", 70, 200, False, {}), # UV Step 1 fail -> switches to OC
        CandidateResult(True, "PASS", 70, 200, False, {}),  # OC Step 1 (offset 0/0) pass
        CandidateResult(True, "PASS", 70, 200, False, {}),  # OC Step 2 (offset 0/15) pass
        CandidateResult(False, "FAIL", 70, 200, False, {}), # OC Step 3 (offset 0/30) fail
    ]
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    with patch("voltvandal.core.tuning.linear.save_session") as mock_save:
        run_session(mock_state, interrupted, recovery)
    
    assert mock_state.hybrid_phase == "oc"
    assert mock_state.hybrid_locked_mv == 0
    assert mock_state.current_step == 2
    assert mock_state.current_offset_mhz == 15


@patch("voltvandal.core.tuning.confident.evaluate_candidate")
def test_evaluate_candidate_confident_enforces_warmup_minimum(mock_eval, mock_state):
    mock_state.stress_seconds = 12
    mock_state.multi_stress_seconds = 10

    captured_durations = []

    def _capture(state, *args, **kwargs):
        captured_durations.append((state.stress_seconds, state.multi_stress_seconds))
        return CandidateResult(True, "PASS", 70, 200, False, {})

    mock_eval.side_effect = _capture

    interrupted = threading.Event()
    recovery = threading.Event()
    evaluate_candidate_confident(
        mock_state, Path("mock.csv"), "label", interrupted, recovery, warmup=True
    )

    # First call is warmup.
    assert captured_durations
    assert captured_durations[0][0] >= 30
    assert captured_durations[0][1] >= 30


@patch("voltvandal.core.tuning.confident.evaluate_candidate")
def test_evaluate_candidate_confident_default_skips_warmup(mock_eval, mock_state):
    mock_state.stress_seconds = 12
    mock_state.multi_stress_seconds = 10

    captured_durations = []

    def _capture(state, *args, **kwargs):
        captured_durations.append((state.stress_seconds, state.multi_stress_seconds))
        return CandidateResult(True, "PASS", 70, 200, False, {})

    mock_eval.side_effect = _capture

    interrupted = threading.Event()
    recovery = threading.Event()
    evaluate_candidate_confident(
        mock_state, Path("mock.csv"), "label", interrupted, recovery
    )

    assert captured_durations
    # First call is run1 (no warmup), so durations stay unchanged.
    assert captured_durations[0][0] == 12
    assert captured_durations[0][1] == 10
