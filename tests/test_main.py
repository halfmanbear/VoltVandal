from types import SimpleNamespace
from unittest.mock import patch

from voltvandal.main import _normalize_state_controls, _run_tuning_with_hotkey, interrupted


def test_normalize_state_controls_forces_manual_when_speed_is_set():
    state = SimpleNamespace(fan_mode="auto", fan_speed_pct=65)
    _normalize_state_controls(state)
    assert state.fan_mode == "manual"


def test_normalize_state_controls_keeps_manual_when_already_manual():
    state = SimpleNamespace(fan_mode="manual", fan_speed_pct=65)
    _normalize_state_controls(state)
    assert state.fan_mode == "manual"


def test_run_tuning_with_hotkey_ctrl_c_resets_factory_defaults():
    state = SimpleNamespace(
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


def test_run_tuning_with_hotkey_non_ctrl_c_uses_revert():
    state = SimpleNamespace(
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


def test_run_tuning_with_hotkey_dispatches_mvscan_mode():
    state = SimpleNamespace(
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
