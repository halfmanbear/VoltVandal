from __future__ import annotations

import subprocess
import re
from typing import Optional


def _require_pynvml():
    try:
        import pynvml  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "pynvml is required to apply GPU runtime controls. "
            "Install dependencies and ensure NVIDIA drivers are available."
        ) from exc
    return pynvml


def _clamp_power_limit_mw(target_mw: int, min_mw: int, max_mw: int) -> int:
    return max(min_mw, min(max_mw, target_mw))


def _target_power_from_percent(default_mw: int, power_limit_pct: int) -> int:
    raw = (default_mw * power_limit_pct) / 100.0
    # NVML power limits are commonly 1000 mW granularity; rounding here avoids
    # set-limit errors on strict drivers.
    return int(round(raw / 1000.0) * 1000)


def _run_command(cmd: list[str], timeout_seconds: float = 8.0) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or "timed out"
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=stdout,
            stderr=f"Command timeout after {timeout_seconds:.1f}s: {detail}",
        )


def apply_power_limit_percent(gpu_index: int, power_limit_pct: int) -> Optional[str]:
    """
    Apply GPU power limit as % of default board power via NVML.

    Returns a human-readable summary string when a change is applied or clamped.
    Returns None when no action is required (100% means unchanged).
    """
    if power_limit_pct <= 0:
        raise ValueError("--power-limit-pct must be greater than 0")
    if power_limit_pct == 100:
        return None

    pynvml = _require_pynvml()
    handle = None
    clamped = False
    target_mw = 0
    default_mw = 0
    min_mw = 0
    max_mw = 0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        default_mw = int(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
        min_mw, max_mw = [
            int(v) for v in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        ]

        target_mw = _target_power_from_percent(default_mw, power_limit_pct)
        clamped_target_mw = _clamp_power_limit_mw(target_mw, min_mw, max_mw)
        clamped = clamped_target_mw != target_mw
        target_mw = clamped_target_mw

        current_mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        if current_mw != target_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, target_mw)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply --power-limit-pct={power_limit_pct} on GPU {gpu_index}: {exc}"
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    applied_w = target_mw / 1000.0
    default_w = default_mw / 1000.0
    msg = (
        f"Applied GPU power limit: {power_limit_pct}% of default "
        f"({applied_w:.1f} W target; default {default_w:.1f} W)."
    )
    if clamped:
        msg += (
            f" Clamped to GPU-supported range "
            f"[{min_mw / 1000.0:.1f}, {max_mw / 1000.0:.1f}] W."
        )
    return msg


def apply_power_limit_max(gpu_index: int) -> str:
    """Apply the selected GPU's reported maximum power limit via NVML."""
    pynvml = _require_pynvml()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        _, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        max_mw = int(max_mw)
        current_mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        if current_mw != max_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, max_mw)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply --power-limit-max on GPU {gpu_index}: {exc}"
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return f"Applied GPU {gpu_index} maximum power limit: {max_mw / 1000.0:.1f} W."


def reset_power_limit_default(gpu_index: int) -> str:
    """Reset GPU power limit to driver default via NVML."""
    pynvml = _require_pynvml()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        default_mw = int(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle))
        current_mw = int(pynvml.nvmlDeviceGetPowerManagementLimit(handle))
        if current_mw != default_mw:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
        return f"Reset power limit to default ({default_mw / 1000.0:.1f} W)."
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reset power limit to default on GPU {gpu_index}: {exc}"
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def apply_gpu_throttle_temp(gpu_index: int, gpu_throttle_temp_c: int) -> Optional[str]:
    """
    Apply GPU throttle temperature target via nvidia-smi -gtt.

    Returns a summary string if a value is applied, or None when unchanged (0).
    """
    if gpu_throttle_temp_c < 0:
        raise ValueError("--gpu-throttle-temp-c must be >= 0")
    if gpu_throttle_temp_c == 0:
        return None
    if gpu_throttle_temp_c < 30 or gpu_throttle_temp_c > 120:
        raise ValueError("--gpu-throttle-temp-c must be in range 30-120")

    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "-gtt",
        str(gpu_throttle_temp_c),
    ]
    result = _run_command(cmd)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Failed to apply --gpu-throttle-temp-c={gpu_throttle_temp_c} on GPU {gpu_index}: {detail}"
        )
    return f"Applied GPU throttle temp target: {gpu_throttle_temp_c} C."


def read_gpu_target_temp(gpu_index: int) -> Optional[int]:
    """Best-effort read of current GPU target temperature from nvidia-smi -q."""
    cmd = ["nvidia-smi", "-q", "-i", str(gpu_index), "-d", "TEMPERATURE"]
    result = _run_command(cmd)
    if result.returncode != 0:
        return None
    text = result.stdout or ""
    # Match the non-default line: "GPU Target Temperature : 83 C"
    m = re.search(
        r"^\s*(?!Default )GPU Target Temperature\s*:\s*(\d+)\s*C\s*$",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    return int(m.group(1))


def reset_gpu_throttle_temp(gpu_index: int, restore_temp_c: Optional[int] = None) -> str:
    """
    Reset GPU throttle temperature target to driver default.

    Uses `nvidia-smi -rgtt` where available, with a fallback attempt to `-gtt 0`.
    """
    if restore_temp_c is not None and restore_temp_c > 0:
        cmd = ["nvidia-smi", "-i", str(gpu_index), "-gtt", str(int(restore_temp_c))]
        result = _run_command(cmd)
        if result.returncode == 0:
            return f"Restored GPU throttle temp target to {int(restore_temp_c)} C."

    attempts = [
        ["nvidia-smi", "-i", str(gpu_index), "-rgtt"],
        ["nvidia-smi", "-i", str(gpu_index), "-gtt", "0"],
    ]
    errors: list[str] = []
    for cmd in attempts:
        result = _run_command(cmd)
        if result.returncode == 0:
            return "Reset GPU throttle temp target to default."
        detail = (result.stderr or result.stdout or "").strip()
        errors.append(f"{' '.join(cmd)} -> {detail}")
    raise RuntimeError(
        "Failed to reset GPU throttle temp target. Attempts: " + " | ".join(errors)
    )


def _fan_indices(pynvml, handle) -> list[int]:
    get_num_fans = getattr(pynvml, "nvmlDeviceGetNumFans", None)
    if callable(get_num_fans):
        try:
            count = int(get_num_fans(handle))
            if count > 0:
                return list(range(count))
        except Exception:
            pass
    return [0]


def _set_fan_manual(pynvml, handle, speed_pct: int) -> int:
    set_v2 = getattr(pynvml, "nvmlDeviceSetFanSpeed_v2", None)
    set_legacy = getattr(pynvml, "nvmlDeviceSetFanSpeed", None)
    if not callable(set_v2) and not callable(set_legacy):
        raise RuntimeError("NVML manual fan control API is unavailable on this system.")

    indices = _fan_indices(pynvml, handle)
    if callable(set_v2):
        for idx in indices:
            set_v2(handle, idx, speed_pct)
        return len(indices)

    try:
        for idx in indices:
            set_legacy(handle, idx, speed_pct)
        return len(indices)
    except TypeError:
        set_legacy(handle, speed_pct)
        return 1


def _set_fan_auto(pynvml, handle) -> int:
    set_default_v2 = getattr(pynvml, "nvmlDeviceSetDefaultFanSpeed_v2", None)
    set_default_legacy = getattr(pynvml, "nvmlDeviceSetDefaultFanSpeed", None)
    if not callable(set_default_v2) and not callable(set_default_legacy):
        raise RuntimeError("NVML auto fan reset API is unavailable on this system.")

    indices = _fan_indices(pynvml, handle)
    if callable(set_default_v2):
        for idx in indices:
            set_default_v2(handle, idx)
        return len(indices)

    try:
        for idx in indices:
            set_default_legacy(handle, idx)
        return len(indices)
    except TypeError:
        set_default_legacy(handle)
        return 1


def apply_fan_control(gpu_index: int, fan_mode: str, fan_speed_pct: int) -> Optional[str]:
    """
    Apply fan control mode via NVML where supported.

    - manual: requires fan_speed_pct in [1, 100] and is strict (errors if unsupported).
    - auto: best-effort reset to driver control (returns None if unsupported).
    """
    mode = (fan_mode or "auto").strip().lower()
    if mode not in ("auto", "manual"):
        raise ValueError("--fan-mode must be either 'auto' or 'manual'")
    if fan_speed_pct < 0 or fan_speed_pct > 100:
        raise ValueError("--fan-speed-pct must be in range 0-100")
    if mode == "manual" and fan_speed_pct == 0:
        raise ValueError("--fan-speed-pct must be > 0 when --fan-mode=manual")

    pynvml = _require_pynvml()
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        if mode == "manual":
            fans = _set_fan_manual(pynvml, handle, fan_speed_pct)
            return f"Applied manual fan speed: {fan_speed_pct}% on {fans} fan(s)."
        try:
            fans = _set_fan_auto(pynvml, handle)
            return f"Set fan control to auto on {fans} fan(s)."
        except Exception:
            return None
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply fan control (mode={mode}, speed={fan_speed_pct}) on GPU {gpu_index}: {exc}"
        ) from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
