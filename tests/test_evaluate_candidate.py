import json
import threading
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voltvandal.core.models import CurvePoint
from voltvandal.core.tuning import evaluate as evaluate_module, evaluate_candidate
from voltvandal.core.tuning.stability import _parse_doloming_stability
from voltvandal.hardware.point_lock import PointLockError


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
    monkeypatch.setattr(evaluate_module, "NvmlMonitor", MagicMock(return_value=monitor))
    monkeypatch.setattr(evaluate_module, "inspect_point_lock", MagicMock(return_value={"bus": 3}))
    monkeypatch.setattr(evaluate_module, "check_monitor_identity", MagicMock(return_value="GPU-test"))
    monkeypatch.setattr(evaluate_module, "temporary_point_lock", temporary)
    monkeypatch.setattr(evaluate_module, "nvapi_apply_curve_safe", MagicMock())
    monkeypatch.setattr(evaluate_module, "run_doloming", MagicMock(return_value=(0, "Success")))
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
    assert evaluate_module.NvmlMonitor.call_args.kwargs["point_clock_mhz"] == 1800.0
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
    assert evaluate_module.NvmlMonitor.call_args.kwargs["point_voltage_mv"] == 900.0
    assert evaluate_module.NvmlMonitor.call_args.kwargs["point_lock_bus"] is None


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
    evaluate_module.run_doloming.side_effect = failure
    with pytest.raises(type(failure)):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    assert calls[-1][0] == "restore"
    monitor.stop.assert_called_once()


def test_point_timeout_does_not_count_as_instability(mock_state, point_test):
    monitor, calls, event = point_test
    evaluate_module.run_doloming.return_value = (998, "STRESS_TIMEOUT")
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

    evaluate_module.run_doloming.side_effect = aborted
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "probe", event, threading.Event()
    )
    assert not result.ok
    assert result.reason == expected
    assert result.stress_exit_codes == {"doloming_simple": 999}
    assert evaluate_module.run_doloming.call_count == 1


def test_unknown_monitor_abort_cannot_become_probe_fallback(mock_state, point_test):
    _, _, event = point_test
    mock_state.point_lock = False
    mock_state.doloming_modes = "simple,ray"
    evaluate_module.run_doloming.return_value = (999, "ABORTED_BY_MONITOR")
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "probe", event, threading.Event()
    )
    assert result.reason == "MONITOR_ABORT_UNKNOWN"
    assert evaluate_module.run_doloming.call_count == 1


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

    evaluate_module.run_doloming.side_effect = failed
    result = evaluate_candidate(
        mock_state, Path(mock_state.stock_curve_csv), "candidate", event,
        threading.Event(),
    )
    assert result.reason == "DOLOMING_SIMPLE_CUDA_RUNTIME_ERROR"
    assert not result.ok



def test_point_capability_failure_does_not_apply_curve(mock_state, point_test):
    _, calls, event = point_test
    evaluate_module.inspect_point_lock.side_effect = PointLockError("unsupported")
    with pytest.raises(PointLockError, match="unsupported"):
        evaluate_candidate(mock_state, Path(mock_state.stock_curve_csv), "point", event,
                           threading.Event(), target_point=CurvePoint(900000, 1800000))
    evaluate_module.nvapi_apply_curve_safe.assert_not_called()
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
    assert evaluate_module.run_doloming.call_count == 2


@patch("voltvandal.core.tuning.evaluate.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.evaluate.run_doloming")
@patch("voltvandal.core.tuning.evaluate.NvmlMonitor")
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


@patch("voltvandal.core.tuning.evaluate.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.evaluate.run_doloming")
@patch("voltvandal.core.tuning.evaluate.NvmlMonitor")
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

@patch("voltvandal.core.tuning.evaluate.nvapi_apply_curve_safe")
@patch("voltvandal.core.tuning.evaluate.NvmlMonitor")
def test_evaluate_candidate_monitor_fail(mock_monitor_cls, mock_apply, mock_state):
    mock_monitor = MagicMock()
    mock_monitor_cls.return_value = mock_monitor
    mock_monitor.start.side_effect = Exception("NVML Error")
    
    interrupted = threading.Event()
    recovery = threading.Event()
    
    result = evaluate_candidate(mock_state, Path("mock.csv"), "label", interrupted, recovery)
    
    assert result.ok is False
    assert "MONITOR_START_FAILED" in result.reason
