import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import threading

from voltvandal.core.models import SessionState, CandidateResult, CurvePoint
from voltvandal.core.tuning import run_session, evaluate_candidate, _parse_doloming_stability

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

@patch("voltvandal.core.tuning.nvapi_apply_curve")
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

def test_parse_doloming_stability_low_score_gamma():
    sample = (
        "Score_gamma : 92.40\n"
        "Avg/Max ratio : 0.9240\n"
        "\nTest Summary:\n"
        "Status              : Successfully maintain\n"
        "Average Utilization : 74.10%\n"
    )
    stable, reason = _parse_doloming_stability(sample, "frequency-max")
    assert stable is False
    assert reason == "DOLOMING_FREQUENCY_MAX_LOW_SCORE_GAMMA"

@patch("voltvandal.core.tuning.nvapi_apply_curve")
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

@patch("voltvandal.core.tuning.nvapi_apply_curve")
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
@patch("voltvandal.core.tuning.nvapi_apply_curve")
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
@patch("voltvandal.core.tuning.nvapi_apply_curve")
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
@patch("voltvandal.core.tuning.nvapi_apply_curve")
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
