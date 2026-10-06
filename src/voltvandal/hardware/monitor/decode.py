"""Throttle/perf-decrease bit decoding and streak helpers."""

from typing import Optional

_THROTTLE_ABORT_CONSECUTIVE_POLLS: int = 3
# A hung GPU keeps its clock but its power falls far below what the run has
# already shown it draws (seen before the 0x116 TDR crash). Abort early.
_COLLAPSE_MIN_PEAK_W: float = 150.0
_COLLAPSE_POWER_RATIO: float = 0.45
_COLLAPSE_ABORT_CONSECUTIVE_POLLS: int = 5

_THROTTLE_LABELS = {
    0x0000000000000001: "Idle",
    0x0000000000000002: "AppClk",
    0x0000000000000004: "PwrCap",
    0x0000000000000008: "HwSlowdn",
    0x0000000000000010: "SyncBst",
    0x0000000000000020: "SwTherm",
    0x0000000000000040: "HwTherm",
    0x0000000000000080: "PwrBrake",
    0x0000000000000100: "DispClk",
}

_THROTTLE_IDLE_BIT = 0x0000000000000001
_THROTTLE_PWRCAP_BIT = 0x0000000000000004
_THROTTLE_SEVERE_BITS = (
    0x0000000000000008  # HwSlowdn
    | 0x0000000000000020  # SwTherm
    | 0x0000000000000040  # HwTherm
    | 0x0000000000000080  # PwrBrake
)
_PERF_DECREASE_LABELS = {
    0x00000001: "InsufficientPower",
    0x00000004: "AcPower",
    0x00000010: "PowerBrake",
    0x00000040: "Thermal",
}

def _decode_throttle(reasons: int) -> str:
    if reasons == 0:
        return ""
    active = [lbl for bit, lbl in _THROTTLE_LABELS.items() if reasons & bit]
    return "+".join(active) if active else f"0x{reasons:X}"

def _has_actionable_throttle(reasons: int) -> bool:
    actionable = reasons & ~_THROTTLE_IDLE_BIT
    if actionable == 0:
        return False
    # Ignore pure power-cap throttling for abort logic; this is common and
    # not by itself a stability failure.
    if actionable == _THROTTLE_PWRCAP_BIT:
        return False
    return True

def _decode_perf_decrease(info: Optional[int]) -> str:
    if info is None:
        return ""
    if info == 0:
        return "None"
    active = [lbl for bit, lbl in _PERF_DECREASE_LABELS.items() if info & bit]
    return "+".join(active) if active else f"0x{info:X}"

def _next_throttle_streak(prev_streak: int, reasons: int) -> int:
    return prev_streak + 1 if _has_actionable_throttle(reasons) else 0

def _next_collapse_streak(prev_streak: int, power_w: float, peak_power_w: float) -> int:
    collapsed = (
        peak_power_w >= _COLLAPSE_MIN_PEAK_W
        and power_w < peak_power_w * _COLLAPSE_POWER_RATIO
    )
    return prev_streak + 1 if collapsed else 0

def _fmt_signed_int(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)
