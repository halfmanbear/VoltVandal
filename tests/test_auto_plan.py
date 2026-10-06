import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from voltvandal.core.models import CandidateResult
from voltvandal.core.tuning import run_vlock_session
from voltvandal.core.tuning.autoplan import _run_auto_plan_suite


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
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=candidate) as evaluate:
        with patch("voltvandal.core.tuning.vlock._run_auto_plan_suite", side_effect=final_results) as final:
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 4  # bins 4, 3, then newly discovered 2, 1
    assert final.call_count == 2
    assert mock_state.vlock_uv_bin_idx == 0
    assert mock_state.vlock_phase == "done"


def test_auto_plan_missing_measurements_completes_full_sweep(mock_state):
    _configure_auto_plan_phase2(mock_state)
    candidate = CandidateResult(True, "PASS", metrics=None)
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=candidate) as evaluate:
        with patch("voltvandal.core.tuning.vlock._run_auto_plan_suite", return_value=candidate):
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert evaluate.call_count == 5
    assert mock_state.auto_plan_fallback_full
    assert mock_state.vlock_phase == "done"


def test_auto_plan_final_failure_restores_stock_curve(mock_state):
    _configure_auto_plan_phase2(mock_state)
    passing = CandidateResult(True, "PASS", metrics=_plan_metrics(925))
    failing = CandidateResult(False, "DOLOMING_RAY_RC_1")
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=passing):
        with patch("voltvandal.core.tuning.vlock._run_auto_plan_suite", return_value=failing):
            with patch("voltvandal.core.tuning.vlock.nvapi_apply_curve_safe") as apply:
                run_vlock_session(mock_state, threading.Event(), threading.Event())
    apply.assert_called_once_with(0, Path(mock_state.stock_curve_csv), timeout_seconds=12.0)
    assert Path(mock_state.last_good_curve_csv).read_bytes() == Path(mock_state.stock_curve_csv).read_bytes()
    assert mock_state.vlock_phase == "failed"


def test_auto_plan_suite_restores_stress_settings_on_error(mock_state):
    mock_state.doloming_modes = "ray"
    mock_state.multi_stress_seconds = 40
    mock_state.gpuburn = "gpu-burn"
    with patch("voltvandal.core.tuning.autoplan.evaluate_candidate", side_effect=RuntimeError("probe failed")):
        with pytest.raises(RuntimeError, match="probe failed"):
            _run_auto_plan_suite(
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
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident") as evaluate:
        with patch("voltvandal.core.tuning.vlock._run_auto_plan_suite",
                   return_value=probe_failure):
            with pytest.raises(RuntimeError, match="DOLOMING_MATRIX_RC_996"):
                run_vlock_session(mock_state, threading.Event(), threading.Event())
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
    with patch("voltvandal.core.tuning.vlock.evaluate_candidate_confident", return_value=passing) as evaluate:
        with patch("voltvandal.core.tuning.vlock._run_auto_plan_suite",
                   side_effect=[passing, passing]) as suite:
            run_vlock_session(mock_state, threading.Event(), threading.Event())
    assert suite.call_count == 2
    assert suite.call_args_list[0].args[2] == "vlock_plan_stock_probe"
    assert evaluate.call_count == 3  # anchor and only two selected lower bins
    assert mock_state.auto_plan_probe_done
    assert mock_state.auto_plan_min_bin_idx == 3
    assert mock_state.vlock_phase == "done"
