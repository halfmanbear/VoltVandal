from types import SimpleNamespace
from unittest.mock import patch

import pytest

from voltvandal.main import (
    _apply_pre_tune_controls, _normalize_state_controls, _run_tuning_with_hotkey,
    _resume_safety_reason, interrupted,
)


def test_normalize_state_controls_forces_manual_when_speed_is_set():
    state = SimpleNamespace(fan_mode="auto", fan_speed_pct=65)
    _normalize_state_controls(state)
    assert state.fan_mode == "manual"


def test_normalize_state_controls_keeps_manual_when_already_manual():
    state = SimpleNamespace(fan_mode="manual", fan_speed_pct=65)
    _normalize_state_controls(state)
    assert state.fan_mode == "manual"


def test_resume_refuses_candidate_left_active_by_crash():
    state = SimpleNamespace(
        mode="vlock", active_candidate_label="vlock_p1_step026_2115mhz_coarse",
        vlock_phase="oc",
    )
    assert "automatic retry is disabled" in _resume_safety_reason(state)


def test_resume_refuses_inconclusive_lower_bin_sweep():
    state = SimpleNamespace(
        mode="vlock", active_candidate_label="", vlock_phase="inconclusive",
    )
    assert "automatic retry is disabled" in _resume_safety_reason(state)


def test_new_run_refuses_existing_baseline_without_overwriting_it(tmp_path):
    from voltvandal.main import main

    out = tmp_path / "old-session"
    out.mkdir()
    baseline = out / "stock_curve.csv"
    baseline.write_text("old baseline", encoding="utf-8")
    args = SimpleNamespace(command="run", out=str(out))
    with patch("voltvandal.main.parse_args", return_value=args):
        with patch("voltvandal.main.warn_if_not_admin"):
            with pytest.raises(ValueError, match="choose a new --out"):
                main()
    assert baseline.read_text(encoding="utf-8") == "old baseline"


def test_pre_tune_controls_apply_max_power_limit_when_selected():
    state = SimpleNamespace(
        gpu=1, power_limit_max=True, power_limit_pct=100,
        gpu_throttle_temp_c=0, gpu_throttle_temp_restore_c=0,
        fan_mode="auto", fan_speed_pct=0,
    )
    with patch("voltvandal.main.apply_power_limit_max", return_value="370 W") as apply_max:
        with patch("voltvandal.main.apply_power_limit_percent") as apply_pct:
            with patch("voltvandal.main.apply_gpu_throttle_temp", return_value=None):
                with patch("voltvandal.main.apply_fan_control", return_value=None):
                    _apply_pre_tune_controls(state)
    apply_max.assert_called_once_with(1)
    apply_pct.assert_not_called()


def test_run_tuning_with_hotkey_ctrl_c_resets_factory_defaults(tmp_path):
    state = SimpleNamespace(
        out_dir=str(tmp_path),
        mode="uv",
        recovery_hotkey_enabled=False,
        recovery_hotkey="ctrl+shift+f12",
        last_good_curve_csv="mock.csv",
        gpu=0,
    )
    interrupted.set()
    with patch("voltvandal.main.run_session", side_effect=KeyboardInterrupt("User pressed Ctrl+C")):
        with patch("voltvandal.main._reset_board_to_factory_defaults") as mock_reset:
            _run_tuning_with_hotkey(state)
    mock_reset.assert_called_once_with(state)


def test_run_tuning_with_hotkey_non_ctrl_c_uses_revert(tmp_path):
    state = SimpleNamespace(
        out_dir=str(tmp_path),
        mode="uv",
        recovery_hotkey_enabled=False,
        recovery_hotkey="ctrl+shift+f12",
        last_good_curve_csv="mock.csv",
        gpu=0,
    )
    interrupted.clear()
    with patch("voltvandal.main.run_session", side_effect=KeyboardInterrupt("Manual recovery hotkey")):
        with patch("voltvandal.main.revert_to_last_good") as mock_revert:
            _run_tuning_with_hotkey(state)
    mock_revert.assert_called_once_with(state)


def test_run_tuning_with_hotkey_dispatches_mvscan_mode(tmp_path):
    state = SimpleNamespace(
        out_dir=str(tmp_path),
        mode="mvscan",
        recovery_hotkey_enabled=False,
        recovery_hotkey="ctrl+shift+f12",
        last_good_curve_csv="mock.csv",
        gpu=0,
    )
    interrupted.clear()
    with patch("voltvandal.main.run_mvscan_session") as mock_mvscan:
        _run_tuning_with_hotkey(state)
    mock_mvscan.assert_called_once()
