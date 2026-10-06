import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import threading

from voltvandal.core.models import SessionState, CandidateResult, CurvePoint
from contextlib import contextmanager
from voltvandal.core import tuning
from voltvandal.hardware.point_lock import PointLockError
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


@pytest.fixture
def point_test(monkeypatch, mock_state):
    mock_state.mode = "vlock"
    mock_state.point_lock = True
    calls = []

    @contextmanager
    def temporary(gpu, voltage, journal):
        calls.append(("lock", voltage))
        try:
            yield 3
        finally:
            calls.append(("restore", voltage))

    monitor = MagicMock()
    monitor.abort_event = threading.Event()
    monitor.abort_reason = ""
    monitor.metrics.return_value = {}
    monitor.point_coverage.return_value = dict(
        loaded_samples=5, measured_samples=5, matched_samples=5, coverage_pct=100.0)
    monitor.point_marker.return_value = (0, 0, 0)
    monkeypatch.setattr(tuning, "NvmlMonitor", MagicMock(return_value=monitor))
    monkeypatch.setattr(tuning, "inspect_point_lock", MagicMock(return_value={"bus": 3}))
    monkeypatch.setattr(tuning, "check_monitor_identity", MagicMock(return_value="GPU-test"))
    monkeypatch.setattr(tuning, "temporary_point_lock", temporary)
    monkeypatch.setattr(tuning, "nvapi_apply_curve_safe", MagicMock())
    monkeypatch.setattr(tuning, "run_doloming", MagicMock(return_value=(0, "Success")))
    event = MagicMock()
    event.is_set.return_value = False
    return monitor, calls, event


def test_point_candidate_passes_only_with_coverage(mock_state, point_test):
    monitor, calls, event = point_test
    result = evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                                threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert result.ok
    assert calls == [("lock", 900000), ("restore", 900000)]
    monitor.stop.assert_called_once()
    assert tuning.NvmlMonitor.call_args.kwargs["point_clock_mhz"] == 1800.0
    report = Path(mock_state.out_dir) / "logs" / "point_point.jsonl"
    assert json.loads(report.read_text())["coverage_pct"] == 100.0


def test_insufficient_coverage_is_inconclusive_and_restores(mock_state, point_test):
    monitor, calls, event = point_test
    monitor.point_coverage.return_value = dict(
        loaded_samples=10, measured_samples=10, matched_samples=1, coverage_pct=10.0)
    with pytest.raises(PointLockError, match="INCONCLUSIVE_POINT_COVERAGE"):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert calls[-1][0] == "restore"
    assert mock_state.current_step == 0
    monitor.stop.assert_called_once()


def test_unlocked_lower_bin_without_loaded_coverage_is_inconclusive(mock_state, point_test):
    monitor, calls, event = point_test
    mock_state.point_lock = False
    monitor.point_coverage.return_value = dict(
        loaded_samples=100, measured_samples=100, matched_samples=0,
        coverage_pct=0.0, target_voltage_mv=900.0, target_clock_mhz=1800.0,
    )
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "lower", event,
        threading.Event(), target_point=CurvePoint(900000, 1800000),
    )
    assert not result.ok
    assert result.reason.startswith("INCONCLUSIVE_BIN_COVERAGE")
    assert "0/100 loaded samples" in result.reason
    assert calls == []  # No voltage lock was requested.
    assert tuning.NvmlMonitor.call_args.kwargs["point_voltage_mv"] == 900.0
    assert tuning.NvmlMonitor.call_args.kwargs["point_lock_bus"] is None


def test_unlocked_lower_bin_passes_with_loaded_coverage(mock_state, point_test):
    monitor, calls, event = point_test
    mock_state.point_lock = False
    monitor.point_coverage.return_value = dict(
        loaded_samples=100, measured_samples=100, matched_samples=90,
        coverage_pct=90.0, target_voltage_mv=900.0, target_clock_mhz=1800.0,
    )
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "lower", event,
        threading.Event(), target_point=CurvePoint(900000, 1800000),
    )
    assert result.ok
    assert calls == []


@pytest.mark.parametrize("failure", [RuntimeError("spawn failed"), KeyboardInterrupt()])
def test_point_workload_exceptions_cleanup(mock_state, point_test, failure):
    monitor, calls, event = point_test
    tuning.run_doloming.side_effect = failure
    with pytest.raises(type(failure)):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert calls[-1][0] == "restore"
    monitor.stop.assert_called_once()


def test_point_timeout_does_not_count_as_instability(mock_state, point_test):
    monitor, calls, event = point_test
    tuning.run_doloming.return_value = (998, "STRESS_TIMEOUT")
    with pytest.raises(PointLockError, match="INCONCLUSIVE_POINT_TEST"):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert calls[-1][0] == "restore"


@pytest.mark.parametrize(
    ("driver_reset", "abort_reason", "expected"),
    [
        (True, "GPU_LOST", "GPU_DRIVER_RESET_DETECTED"),
        (False, "TELEMETRY_UNAVAILABLE", "MONITOR_ABORT_THRESHOLD:TELEMETRY_UNAVAILABLE"),
    ],
)
def test_monitor_abort_takes_priority_over_stress_exit_code(
    mock_state, point_test, driver_reset, abort_reason, expected
):
    monitor, _, event = point_test
    mock_state.point_lock = False
    mock_state.doloming_modes = "simple,ray"
    monitor.driver_reset_detected = driver_reset
    monitor.abort_reason = abort_reason

    def aborted(*args, **kwargs):
        monitor.abort_event.set()
        return 999, "ABORTED_BY_MONITOR"

    tuning.run_doloming.side_effect = aborted
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "probe", event, threading.Event()
    )
    assert not result.ok
    assert result.reason == expected
    assert result.stress_exit_codes == {"doloming_simple": 999}
    assert tuning.run_doloming.call_count == 1


def test_unknown_monitor_abort_cannot_become_probe_fallback(mock_state, point_test):
    _, _, event = point_test
    mock_state.point_lock = False
    mock_state.doloming_modes = "simple,ray"
    tuning.run_doloming.return_value = (999, "ABORTED_BY_MONITOR")
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "probe", event, threading.Event()
    )
    assert result.reason == "MONITOR_ABORT_UNKNOWN"
    assert tuning.run_doloming.call_count == 1


def test_cuda_error_is_reported_even_when_monitor_also_aborts(mock_state, point_test):
    monitor, _, event = point_test
    mock_state.point_lock = False
    monitor.driver_reset_detected = False
    monitor.abort_reason = "TELEMETRY_UNAVAILABLE"

    def failed(*args, **kwargs):
        monitor.abort_event.set()
        return 999, (
            "Error during test: access violation\n"
            "Error in GPU computation: CUDA_ERROR_ILLEGAL_ADDRESS\n"
            "ABORTED_BY_MONITOR"
        )

    tuning.run_doloming.side_effect = failed
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "candidate", event,
        threading.Event(),
    )
    assert result.reason == "DOLOMING_SIMPLE_CUDA_RUNTIME_ERROR"
    assert not result.ok


def test_vlock_stops_after_cuda_runtime_error(mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_target_mv = 900
    failure = CandidateResult(False, "HARD_FAIL:DOLOMING_SIMPLE_CUDA_RUNTIME_ERROR")
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=failure):
        with pytest.raises(RuntimeError, match="CUDA_RUNTIME_ERROR"):
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
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

    with patch("voltvandal.core.tuning.evaluate_candidate_confident", side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert mock_state.active_candidate_label


def test_point_capability_failure_does_not_apply_curve(mock_state, point_test):
    _, calls, event = point_test
    tuning.inspect_point_lock.side_effect = PointLockError("unsupported")
    with pytest.raises(PointLockError, match="unsupported"):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    tuning.nvapi_apply_curve_safe.assert_not_called()
    assert calls == []


def test_second_workload_must_also_cover_the_point(mock_state, point_test):
    monitor, calls, event = point_test
    mock_state.doloming_modes = "simple,matrix"
    monitor.point_coverage.side_effect = [
        dict(matched_samples=5, coverage_pct=100.0),
        dict(matched_samples=0, coverage_pct=0.0),
    ]
    with pytest.raises(PointLockError, match="matrix"):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert tuning.run_doloming.call_count == 2
    assert calls[-1][0] == "restore"

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


def test_vlock_phase2_stops_and_reverts_unexercised_bin(mock_state):
    mock_state.mode = "vlock"
    mock_state.vlock_phase = "uv"
    mock_state.vlock_uv_bin_idx = 0
    mock_state.vlock_target_mv = 950
    mock_state.vlock_anchor_freq_khz = 1900000
    before = Path(mock_state.last_good_curve_csv).read_bytes()
    inconclusive = CandidateResult(False, "HARD_FAIL:INCONCLUSIVE_BIN_COVERAGE: simple: 0/100 loaded samples")
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=inconclusive) as evaluate:
        with patch("voltvandal.core.tuning.revert_to_last_good") as revert:
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 1
    assert evaluate.call_args.kwargs["target_point"] == CurvePoint(900000, 1800000)
    revert.assert_called_once_with(mock_state)
    assert Path(mock_state.last_good_curve_csv).read_bytes() == before
    assert mock_state.vlock_phase == "inconclusive"
    assert mock_state.vlock_uv_bin_idx == 0
    assert mock_state.active_candidate_label == ""


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


def _plan_metrics(minimum_mv):
    return {
        "loaded_sample_count": 40,
        "measured_loaded_sample_count": 40,
        "min_measured_loaded_voltage_mv": minimum_mv,
        "sustained_min_measured_loaded_voltage_mv": minimum_mv,
    }


def _configure_auto_plan_phase2(state):
    rows = "voltageUV,frequencyKHz\n" + "".join(
        f"{mv * 1000},{freq * 1000}\n"
        for mv, freq in ((800, 1200), (825, 1300), (850, 1400),
                         (875, 1500), (900, 1600), (925, 1700))
    )
    Path(state.stock_curve_csv).write_text(rows, encoding="utf-8")
    Path(state.last_good_curve_csv).write_text(rows, encoding="utf-8")
    state.mode = "vlock"
    state.vlock_phase = "uv"
    state.vlock_target_mv = 925
    state.vlock_anchor_freq_khz = 1760000
    state.vlock_uv_bin_idx = 4
    state.auto_plan = True
    state.auto_plan_probe_done = True
    state.auto_plan_min_bin_idx = 3


def test_auto_plan_expands_after_final_curve_reaches_lower_voltage(mock_state):
    _configure_auto_plan_phase2(mock_state)
    candidate = CandidateResult(True, "PASS", metrics=_plan_metrics(925))
    final_results = [
        CandidateResult(True, "PASS", metrics=_plan_metrics(875)),
        CandidateResult(True, "PASS", metrics=_plan_metrics(925)),
    ]
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=candidate) as evaluate:
        with patch("voltvandal.core.tuning._run_auto_plan_suite", side_effect=final_results) as final:
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 4  # bins 4, 3, then newly discovered 2, 1
    assert final.call_count == 2
    assert mock_state.vlock_uv_bin_idx == 0
    assert mock_state.vlock_phase == "done"


def test_auto_plan_missing_measurements_completes_full_sweep(mock_state):
    _configure_auto_plan_phase2(mock_state)
    candidate = CandidateResult(True, "PASS", metrics=None)
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=candidate) as evaluate:
        with patch("voltvandal.core.tuning._run_auto_plan_suite", return_value=candidate):
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 5
    assert mock_state.auto_plan_fallback_full
    assert mock_state.vlock_phase == "done"


def test_auto_plan_final_failure_restores_stock_curve(mock_state):
    _configure_auto_plan_phase2(mock_state)
    passing = CandidateResult(True, "PASS", metrics=_plan_metrics(925))
    failing = CandidateResult(False, "DOLOMING_RAY_RC_1")
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=passing):
        with patch("voltvandal.core.tuning._run_auto_plan_suite", return_value=failing):
            with patch("voltvandal.core.tuning.nvapi_apply_curve_safe") as apply:
                tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    apply.assert_called_once_with(0, Path(mock_state.stock_curve_csv), timeout_seconds=12.0)
    assert Path(mock_state.last_good_curve_csv).read_bytes() == Path(mock_state.stock_curve_csv).read_bytes()
    assert mock_state.vlock_phase == "failed"


def test_auto_plan_suite_restores_stress_settings_on_error(mock_state):
    mock_state.doloming_modes = "ray"
    mock_state.multi_stress_seconds = 40
    mock_state.gpuburn = "gpu-burn"
    with patch("voltvandal.core.tuning.evaluate_candidate", side_effect=RuntimeError("probe failed")):
        with pytest.raises(RuntimeError, match="probe failed"):
            tuning._run_auto_plan_suite(
                mock_state, Path(mock_state.stock_curve_csv), "probe", 20,
                threading.Event(), threading.Event(),
            )
    assert mock_state.doloming_modes == "ray"
    assert mock_state.multi_stress_seconds == 40
    assert mock_state.gpuburn == "gpu-burn"


def test_auto_plan_probe_failure_stops_before_tuning(mock_state):
    mock_state.mode = "vlock"
    mock_state.auto_plan = True
    mock_state.vlock_target_mv = 950
    mock_state.max_steps = 0
    probe_failure = CandidateResult(False, "DOLOMING_MATRIX_RC_996")
    with patch("voltvandal.core.tuning.evaluate_candidate_confident") as evaluate:
        with patch("voltvandal.core.tuning._run_auto_plan_suite",
                   return_value=probe_failure):
            with pytest.raises(RuntimeError, match="DOLOMING_MATRIX_RC_996"):
                tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    evaluate.assert_not_called()
    assert not mock_state.auto_plan_probe_done


def test_auto_plan_stock_probe_selects_shorter_contiguous_sweep(mock_state):
    _configure_auto_plan_phase2(mock_state)
    mock_state.vlock_phase = "oc"
    mock_state.vlock_uv_bin_idx = -1
    mock_state.auto_plan_probe_done = False
    mock_state.auto_plan_min_bin_idx = -1
    mock_state.max_steps = 0
    mock_state.current_step = 0
    passing = CandidateResult(True, "PASS", metrics=_plan_metrics(925))
    with patch("voltvandal.core.tuning.evaluate_candidate_confident", return_value=passing) as evaluate:
        with patch("voltvandal.core.tuning._run_auto_plan_suite",
                   side_effect=[passing, passing]) as suite:
            tuning.run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert suite.call_count == 2
    assert suite.call_args_list[0].args[2] == "vlock_plan_stock_probe"
    assert evaluate.call_count == 3  # anchor and only two selected lower bins
    assert mock_state.auto_plan_probe_done
    assert mock_state.auto_plan_min_bin_idx == 3
    assert mock_state.vlock_phase == "done"


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
