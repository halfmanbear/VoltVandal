import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from voltvandal.core.models import CandidateResult, CurvePoint
from voltvandal.core.tuning import run_vlock_session


def test_vlock_stops_after_cuda_runtime_error(mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_target_mv = 900
    failure = CandidateResult(False, "HARD_FAIL:DOLOMING_SIMPLE_CUDA_RUNTIME_ERROR")
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=failure):
        with pytest.raises(RuntimeError, match="CUDA_RUNTIME_ERROR"):
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert mock_state.current_step == 0
    assert mock_state.vlock_phase == "failed"
    assert mock_state.active_candidate_label.startswith("vlock_p1_step000")


def test_vlock_persists_active_candidate_before_evaluation(mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_target_mv = 900

    def interrupted(*args, **kwargs):
        assert mock_state.active_candidate_label.startswith("vlock_p1_step000")
        assert json.loads(Path(mock_state.checkpoint_json).read_text())[
            "active_candidate_label"
        ] == mock_state.active_candidate_label
        raise KeyboardInterrupt("simulated shutdown")

    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert mock_state.active_candidate_label


@patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.vlock.load_curve_csv")
def test_run_vlock_session_p2_break(mock_load, mock_apply, mock_eval, mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "uv"
    mock_state.vlock_uv_bin_idx = 0 # Correct: anchor_idx - 1
    mock_state.vlock_target_mv = 950
    mock_state.vlock_anchor_freq_khz = 1900000
    
    # Mock stock points: 900mV (idx 0), 950mV (idx 1, anchor)
    mock_load.return_value = [
        CurvePoint(900000, 1800000),
        CurvePoint(950000, 1900000),
    ]
    
    # Phase 2 Bin 0 fails
    mock_eval.return_value = CandidateResult(False, "FAIL", 70, 200, False, {})
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    from voltvandal.core.tuning import run_vlock_session
    
    with patch("voltvandal.core.tuning.vlock.save_session"):
        with patch("voltvandal.core.tuning.vlock.write_curve_csv"):
            run_vlock_session(mock_state, interrupted, recovery)
    
    # It failed at bin 0, didn't decrement further
    assert mock_state.vlock_uv_bin_idx == 0
    # A failed UV candidate must not mark the session as complete.
    assert mock_state.vlock_phase == "uv"


def test_vlock_phase2_stops_and_reverts_unexercised_bin(mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "uv"
    mock_state.vlock_uv_bin_idx = 0
    mock_state.vlock_target_mv = 950
    mock_state.vlock_anchor_freq_khz = 1900000
    before = Path(mock_state.last_good_curve_csv).read_bytes()
    inconclusive = CandidateResult(False, "HARD_FAIL:INCONCLUSIVE_BIN_COVERAGE: simple: 0/100 loaded samples")
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=inconclusive) as evaluate:
        with patch("voltvandal.core.tuning.vlock.revert_to_last_good") as revert:
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 1
    assert evaluate.call_args.kwargs["target_point"] == CurvePoint(900000, 1800000)
    revert.assert_called_once_with(mock_state)
    assert Path(mock_state.last_good_curve_csv).read_bytes() == before
    assert mock_state.vlock_phase == "inconclusive"
    assert mock_state.vlock_uv_bin_idx == 0
    assert mock_state.active_candidate_label == ""


@patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.vlock.load_curve_csv")
def test_run_vlock_session_uses_start_freq_override(mock_load, mock_apply, mock_eval, mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "oc"
    mock_state.max_steps = 0
    mock_state.vlock_target_mv = 950
    mock_state.vlock_oc_base_freq_khz = 1800000
    mock_state.vlock_start_freq_mhz = 2100
    mock_load.return_value = [CurvePoint(950000, 1900000)]
    mock_eval.return_value = CandidateResult(False, "FAIL", 70, 200, False, {})

    interrupted = threading.Event()
    recovery = threading.Event()

    from voltvandal.core.tuning import run_vlock_session

    with patch("voltvandal.core.tuning.vlock.save_session"):
        with patch("voltvandal.core.tuning.vlock.write_curve_csv"):
            with patch("voltvandal.core.tuning.vlock.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    assert mock_eval.call_count == 1
    # Requested 2100 MHz snaps down to nearest stock bin (1900 MHz in this fixture).
    assert mock_eval.call_args.kwargs.get("max_freq_mhz") == 1900


@patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.vlock.load_curve_csv")
def test_run_vlock_session_phase1_coarse_then_fine(mock_load, mock_apply, mock_eval, mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "oc"
    mock_state.max_steps = 10
    mock_state.step_mhz = 15
    mock_state.vlock_target_mv = 950
    mock_state.vlock_oc_base_freq_khz = 1900000
    mock_state.vlock_start_freq_mhz = 0
    mock_state.current_step = 0
    mock_state.vlock_last_fail_step = -1
    mock_load.return_value = [
        CurvePoint(950000, 1900000),
    ]
    # coarse: step 0 pass, step 3 pass, step 6 fail
    # fine: step 4 pass, step 5 fail -> transition to UV
    mock_eval.side_effect = [
        CandidateResult(True, "PASS", 70, 200, False, {}),
        CandidateResult(True, "PASS", 70, 200, False, {}),
        CandidateResult(False, "FAIL", 70, 200, False, {}),
        CandidateResult(True, "PASS", 70, 200, False, {}),
        CandidateResult(False, "FAIL", 70, 200, False, {}),
    ]

    interrupted = threading.Event()
    recovery = threading.Event()

    from voltvandal.core.tuning import run_vlock_session

    with patch("voltvandal.core.tuning.vlock.save_session"):
        with patch("voltvandal.core.tuning.vlock.write_curve_csv"):
            with patch("voltvandal.core.tuning.vlock.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs == [1900, 1930, 1960, 1945]
    assert mock_state.vlock_phase == "done"
    assert mock_state.vlock_anchor_freq_khz == 1945000


@patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.vlock.load_curve_csv")
def test_run_vlock_session_step0_fail_lowers_start_and_retries(mock_load, mock_apply, mock_eval, mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "oc"
    mock_state.max_steps = 0
    mock_state.step_mhz = 15
    mock_state.vlock_target_mv = 912
    mock_state.vlock_start_freq_mhz = 1980
    mock_state.current_step = 0
    mock_state.vlock_last_fail_step = -1
    # Anchor at 912mV = 1785 MHz, with higher stock bins available above it.
    mock_load.return_value = [
        CurvePoint(912000, 1785000),
        CurvePoint(930000, 1845000),
        CurvePoint(950000, 1905000),
        CurvePoint(970000, 1965000),
        CurvePoint(980000, 1980000),
    ]
    # First candidate (1980) fails immediately, second candidate (1965) passes.
    mock_eval.side_effect = [
        CandidateResult(False, "FAIL", 70, 200, False, {}),
        CandidateResult(True, "PASS", 70, 200, False, {}),
    ]

    interrupted = threading.Event()
    recovery = threading.Event()

    from voltvandal.core.tuning import run_vlock_session

    with patch("voltvandal.core.tuning.vlock.save_session"):
        with patch("voltvandal.core.tuning.vlock.write_curve_csv"):
            with patch("voltvandal.core.tuning.vlock.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs[:2] == [1980, 1965]
    assert mock_state.vlock_anchor_freq_khz == 1965000


@patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.recovery.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.vlock.load_curve_csv")
def test_run_vlock_session_step0_fail_does_not_jump_above_initial_fail(mock_load, mock_apply, mock_eval, mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "oc"
    mock_state.max_steps = 10
    mock_state.step_mhz = 15
    mock_state.vlock_target_mv = 912
    mock_state.vlock_start_freq_mhz = 1980
    mock_state.current_step = 0
    mock_state.vlock_last_fail_step = -1
    # Anchor at first bin (912mV), with higher bins available.
    mock_load.return_value = [
        CurvePoint(912000, 1785000),
        CurvePoint(970000, 1965000),
        CurvePoint(980000, 1980000),
        CurvePoint(990000, 1995000),
        CurvePoint(1000000, 2010000),
    ]
    # 1980 immediate fail, then 1965 pass, then 1980 pass.
    mock_eval.side_effect = [
        CandidateResult(False, "FAIL", 70, 200, False, {}),
        CandidateResult(True, "PASS", 70, 200, False, {}),
        CandidateResult(True, "PASS", 70, 200, False, {}),
    ]

    interrupted = threading.Event()
    recovery = threading.Event()

    from voltvandal.core.tuning import run_vlock_session

    with patch("voltvandal.core.tuning.vlock.save_session"):
        with patch("voltvandal.core.tuning.vlock.write_curve_csv"):
            with patch("voltvandal.core.tuning.vlock.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs == [1980, 1965, 1980]
    assert max(freqs) == 1980
