import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import threading

from voltvandal.core.models import SessionState, CandidateResult, CurvePoint
from voltvandal.core.tuning import (
    run_session,
    evaluate_candidate,
    evaluate_candidate_confident,
    _parse_doloming_stability,
)

@pytest.fixture
def mock_state(tmp_path):
    stock_csv = tmp_path / "stock.csv"
    stock_csv.write_text("voltageUV,frequencyKHz\n900000,1800000\n950000,1900000\n")
    
    last_good_csv = tmp_path / "last_good.csv"
    last_good_csv.write_text("voltageUV,frequencyKHz\n900000,1800000\n950000,1900000\n")
    
    checkpoint = tmp_path / "session.json"
    
    return SessionState(
        gpu=0,
        out_dir=str(tmp_path),
        stock_curve_csv=str(stock_csv),
        last_good_curve_csv=str(last_good_csv),
        checkpoint_json=str(checkpoint),
        mode="uv",
        bin_min_mv=800,
        bin_max_mv=1000,
        step_mv=5,
        step_mhz=15,
        max_steps=2,
        stress_seconds=1,
        doloming="mock",
        doloming_mode="simple",
        gpuburn=None,
        poll_seconds=0.1,
        temp_limit_c=90,
        hotspot_limit_c=100,
        hotspot_offset_c=15,
        power_limit_w=400,
        abort_on_throttle=True,
    )

@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.run_doloming")
@patch("voltvandal.core.tuning.NvmlMonitor")
def test_evaluate_candidate_success(mock_monitor_cls, mock_run_dolo, mock_apply, mock_state):
    mock_monitor = MagicMock()
    mock_monitor_cls.return_value = mock_monitor
    mock_monitor.abort_event = threading.Event()
    mock_monitor.max_temp = 70
    mock_monitor.max_power = 200
    mock_monitor.any_throttle = False
    
    mock_run_dolo.return_value = (0, "Success")
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    result = evaluate_candidate(mock_state, Path("mock.csv"), "label", interrupted, recovery)
    
    assert result.ok is True
    assert result.reason == "PASS"
    assert result.telemetry_max_temp_c == 70
    mock_monitor.start.assert_called_once()
    mock_monitor.stop.assert_called_once()

def test_parse_doloming_stability_unstable_status():
    sample = (
        "Starting integrated stress mode=simple gpu=0 duration=60s\n"
        "\nTest Summary:\n"
        "Status              : Unstable\n"
        "Average Utilization : 21.50%\n"
    )
    stable, reason = _parse_doloming_stability(sample, "simple")
    assert stable is False
    assert reason == "DOLOMING_SIMPLE_UNSTABLE_STATUS"

def test_parse_doloming_stability_failed_to_stabilize():
    sample = (
        "Warning: GPU 0 failed to fully stabilize within 30s.\n"
        "Continuing with current utilization (23.0%).\n"
    )
    stable, reason = _parse_doloming_stability(sample, "matrix")
    assert stable is False
    assert reason == "DOLOMING_MATRIX_FAILED_TO_STABILIZE"

@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.run_doloming")
@patch("voltvandal.core.tuning.NvmlMonitor")
def test_evaluate_candidate_rejects_unstable_summary(mock_monitor_cls, mock_run_dolo, mock_apply, mock_state):
    mock_monitor = MagicMock()
    mock_monitor_cls.return_value = mock_monitor
    mock_monitor.abort_event = threading.Event()
    mock_monitor.max_temp = 70
    mock_monitor.max_power = 200
    mock_monitor.any_throttle = False

    mock_run_dolo.return_value = (
        0,
        "Test Summary:\nStatus              : Unstable\nAverage Utilization : 20.00%\n",
    )

    interrupted = threading.Event()
    recovery = threading.Event()
    result = evaluate_candidate(mock_state, Path("mock.csv"), "label", interrupted, recovery)

    assert result.ok is False
    assert result.reason == "DOLOMING_SIMPLE_UNSTABLE_STATUS"

@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.NvmlMonitor")
def test_evaluate_candidate_monitor_fail(mock_monitor_cls, mock_apply, mock_state):
    mock_monitor = MagicMock()
    mock_monitor_cls.return_value = mock_monitor
    mock_monitor.start.side_effect = Exception("NVML Error")
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    result = evaluate_candidate(mock_state, Path("mock.csv"), "label", interrupted, recovery)
    
    assert result.ok is False
    assert "MONITOR_START_FAILED" in result.reason

@patch("voltvandal.core.tuning.evaluate_candidate")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
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

@patch("voltvandal.core.tuning.evaluate_candidate")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
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
    
    with patch("voltvandal.core.tuning.save_session") as mock_save:
        run_session(mock_state, interrupted, recovery)
    
    assert mock_state.hybrid_phase == "oc"
    assert mock_state.hybrid_locked_mv == 0
    assert mock_state.current_step == 2
    assert mock_state.current_offset_mhz == 15

@patch("voltvandal.core.tuning.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.load_curve_csv")
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
    
    with patch("voltvandal.core.tuning.save_session"):
        with patch("voltvandal.core.tuning.write_curve_csv"):
            run_vlock_session(mock_state, interrupted, recovery)
    
    # It failed at bin 0, didn't decrement further
    assert mock_state.vlock_uv_bin_idx == 0
    # A failed UV candidate must not mark the session as complete.
    assert mock_state.vlock_phase == "uv"


@patch("voltvandal.core.tuning.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.load_curve_csv")
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

    with patch("voltvandal.core.tuning.save_session"):
        with patch("voltvandal.core.tuning.write_curve_csv"):
            with patch("voltvandal.core.tuning.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    assert mock_eval.call_count == 1
    # Requested 2100 MHz snaps down to nearest stock bin (1900 MHz in this fixture).
    assert mock_eval.call_args.kwargs.get("max_freq_mhz") == 1900


@patch("voltvandal.core.tuning.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.load_curve_csv")
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

    with patch("voltvandal.core.tuning.save_session"):
        with patch("voltvandal.core.tuning.write_curve_csv"):
            with patch("voltvandal.core.tuning.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs == [1900, 1930, 1960, 1945]
    assert mock_state.vlock_phase == "done"
    assert mock_state.vlock_anchor_freq_khz == 1945000


@patch("voltvandal.core.tuning.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.load_curve_csv")
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

    with patch("voltvandal.core.tuning.save_session"):
        with patch("voltvandal.core.tuning.write_curve_csv"):
            with patch("voltvandal.core.tuning.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs[:2] == [1980, 1965]
    assert mock_state.vlock_anchor_freq_khz == 1965000


@patch("voltvandal.core.tuning.evaluate_candidate_confident")
@patch("voltvandal.core.tuning.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.load_curve_csv")
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

    with patch("voltvandal.core.tuning.save_session"):
        with patch("voltvandal.core.tuning.write_curve_csv"):
            with patch("voltvandal.core.tuning.shutil.copyfile"):
                run_vlock_session(mock_state, interrupted, recovery)

    freqs = [c.kwargs.get("max_freq_mhz") for c in mock_eval.call_args_list]
    assert freqs == [1980, 1965, 1980]
    assert max(freqs) == 1980


@patch("voltvandal.core.tuning.evaluate_candidate")
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


@patch("voltvandal.core.tuning.evaluate_candidate")
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

