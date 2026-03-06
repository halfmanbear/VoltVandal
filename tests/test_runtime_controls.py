import types
import sys

from voltvandal.hardware.runtime_controls import (
    _parse_windows_hotkey,
    apply_fan_control,
    apply_gpu_throttle_temp,
    apply_power_limit_percent,
    read_gpu_target_temp,
    reset_gpu_throttle_temp,
    reset_power_limit_default,
)


def _fake_pynvml(default_mw=300000, min_mw=150000, max_mw=330000, current_mw=300000):
    state = {"set_limit_mw": None}

    def nvmlInit():
        return None

    def nvmlShutdown():
        return None

    def nvmlDeviceGetHandleByIndex(idx):
        return idx

    def nvmlDeviceGetPowerManagementDefaultLimit(_handle):
        return default_mw

    def nvmlDeviceGetPowerManagementLimitConstraints(_handle):
        return (min_mw, max_mw)

    def nvmlDeviceGetPowerManagementLimit(_handle):
        return current_mw

    def nvmlDeviceSetPowerManagementLimit(_handle, limit_mw):
        state["set_limit_mw"] = int(limit_mw)

    module = types.SimpleNamespace(
        nvmlInit=nvmlInit,
        nvmlShutdown=nvmlShutdown,
        nvmlDeviceGetHandleByIndex=nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerManagementDefaultLimit=nvmlDeviceGetPowerManagementDefaultLimit,
        nvmlDeviceGetPowerManagementLimitConstraints=nvmlDeviceGetPowerManagementLimitConstraints,
        nvmlDeviceGetPowerManagementLimit=nvmlDeviceGetPowerManagementLimit,
        nvmlDeviceSetPowerManagementLimit=nvmlDeviceSetPowerManagementLimit,
    )
    return module, state


def test_apply_power_limit_percent_sets_expected_target(monkeypatch):
    fake_module, state = _fake_pynvml(current_mw=300000)
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    msg = apply_power_limit_percent(gpu_index=0, power_limit_pct=90)

    assert state["set_limit_mw"] == 270000
    assert msg is not None
    assert "90%" in msg


def test_apply_power_limit_percent_clamps_to_supported_range(monkeypatch):
    fake_module, state = _fake_pynvml(current_mw=300000)
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    msg = apply_power_limit_percent(gpu_index=0, power_limit_pct=200)

    assert state["set_limit_mw"] == 330000
    assert msg is not None
    assert "Clamped" in msg


def test_reset_power_limit_default_sets_driver_default(monkeypatch):
    fake_module, state = _fake_pynvml(default_mw=280000, current_mw=300000)
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    msg = reset_power_limit_default(gpu_index=0)

    assert state["set_limit_mw"] == 280000
    assert "280.0 W" in msg


def test_apply_gpu_throttle_temp_uses_nvidia_smi(monkeypatch):
    called = {"cmd": None}

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd):
        called["cmd"] = cmd
        return Result()

    monkeypatch.setattr(
        "voltvandal.hardware.runtime_controls._run_command",
        fake_run,
    )
    msg = apply_gpu_throttle_temp(gpu_index=1, gpu_throttle_temp_c=85)
    assert called["cmd"] == ["nvidia-smi", "-i", "1", "-gtt", "85"]
    assert "85 C" in msg


def test_reset_gpu_throttle_temp_uses_rgtt(monkeypatch):
    called = {"cmd": None}

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd):
        called["cmd"] = cmd
        return Result()

    monkeypatch.setattr(
        "voltvandal.hardware.runtime_controls._run_command",
        fake_run,
    )
    msg = reset_gpu_throttle_temp(gpu_index=1)
    assert called["cmd"] == ["nvidia-smi", "-i", "1", "-rgtt"]
    assert "default" in msg.lower()


def test_reset_gpu_throttle_temp_prefers_restore_value(monkeypatch):
    called = {"cmd": None}

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd):
        called["cmd"] = cmd
        return Result()

    monkeypatch.setattr(
        "voltvandal.hardware.runtime_controls._run_command",
        fake_run,
    )
    msg = reset_gpu_throttle_temp(gpu_index=0, restore_temp_c=83)
    assert called["cmd"] == ["nvidia-smi", "-i", "0", "-gtt", "83"]
    assert "83 C" in msg


def test_read_gpu_target_temp_parses_query_output(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = (
            "Temperature\n"
            "    GPU Current Temp            : 35 C\n"
            "    GPU Target Temperature      : 83 C\n"
            "    Default GPU Target Temperature : 83 C\n"
        )

    monkeypatch.setattr(
        "voltvandal.hardware.runtime_controls._run_command",
        lambda _cmd: Result(),
    )
    assert read_gpu_target_temp(gpu_index=0) == 83


def test_apply_fan_control_manual(monkeypatch):
    state = {"set_manual": []}

    def nvmlInit():
        return None

    def nvmlShutdown():
        return None

    def nvmlDeviceGetHandleByIndex(idx):
        return idx

    def nvmlDeviceGetNumFans(_handle):
        return 2

    def nvmlDeviceSetFanSpeed_v2(_handle, fan_idx, speed):
        state["set_manual"].append((fan_idx, speed))

    fake_module = types.SimpleNamespace(
        nvmlInit=nvmlInit,
        nvmlShutdown=nvmlShutdown,
        nvmlDeviceGetHandleByIndex=nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetNumFans=nvmlDeviceGetNumFans,
        nvmlDeviceSetFanSpeed_v2=nvmlDeviceSetFanSpeed_v2,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_module)

    msg = apply_fan_control(gpu_index=0, fan_mode="manual", fan_speed_pct=70)
    assert state["set_manual"] == [(0, 70), (1, 70)]
    assert "70%" in msg


def test_parse_windows_hotkey_ctrl_shift_f12():
    mods, vk = _parse_windows_hotkey("ctrl+shift+f12")
    assert mods == (0x0002 | 0x0004)
    assert vk == 0x7B
