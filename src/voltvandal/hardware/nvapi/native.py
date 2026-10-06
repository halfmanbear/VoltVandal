"""nvapi DLL loading, QueryInterface resolution and low-level curve helpers."""

from __future__ import annotations

import ctypes
import sys
import time
from typing import List, Tuple

from .structs import (
    _ID_Initialize, _ID_EnumPhysicalGPUs, _ID_GetClockBoostMask, _ID_GetVFPCurve,
    _ID_GetClockBoostTable, _ID_SetClockBoostTable, _ID_GetThermalSensors,
    _NV_GPU_CLOCK_MASKS, _NV_GPU_VFP_CURVE, _NV_GPU_CLOCK_TABLE, _NV_GPU_THERMAL_SENSORS,
)


# ── DLL + function-pointer resolution ────────────────────────────────────────

def _load_dll() -> ctypes.CDLL:
    dll_name = "nvapi64.dll" if sys.maxsize > 2**32 else "nvapi.dll"
    try:
        return ctypes.CDLL(dll_name)
    except OSError:
        raise RuntimeError(
            f"{dll_name} not found — NVIDIA drivers must be installed."
        )


_dll: ctypes.CDLL | None = None
_func_cache: dict = {}


def _get_dll() -> ctypes.CDLL:
    global _dll
    if _dll is None:
        _dll = _load_dll()
    return _dll


def _qif(func_id: int, restype, argtypes):
    """
    Resolve a function pointer via NvAPI_QueryInterface and return a
    callable ctypes function object.  Results are cached.
    """
    if func_id in _func_cache:
        return _func_cache[func_id]

    dll = _get_dll()
    qi = dll.nvapi_QueryInterface
    qi.restype = ctypes.c_void_p
    qi.argtypes = [ctypes.c_uint32]

    ptr = qi(func_id)
    if not ptr:
        raise RuntimeError(
            f"NvAPI_QueryInterface(0x{func_id:08X}) returned NULL — "
            "function not supported by this driver version."
        )

    # NVAPI QueryInterface exports use __cdecl (CFUNCTYPE, not WINFUNCTYPE).
    ftype = ctypes.CFUNCTYPE(restype, *argtypes)
    func = ftype(ptr)
    _func_cache[func_id] = func
    return func


# ── high-level wrappers around each NVAPI call ────────────────────────────────

def _nvapi_init() -> None:
    f = _qif(_ID_Initialize, ctypes.c_int, [])
    rc = f()
    if rc != 0:
        raise RuntimeError(f"NvAPI_Initialize failed: {rc:#010x}")


def _nvapi_enum_gpus() -> List[int]:
    """Return a list of opaque GPU handle values (as Python ints)."""
    handles = (ctypes.c_void_p * 64)()
    count = ctypes.c_uint32(0)
    f = _qif(
        _ID_EnumPhysicalGPUs,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
    )
    rc = f(
        ctypes.cast(handles, ctypes.c_void_p),
        ctypes.byref(count),
    )
    if rc != 0:
        raise RuntimeError(f"NvAPI_EnumPhysicalGPUs failed: {rc:#010x}")
    return [handles[i] for i in range(count.value)]


def _get_handle(gpu_index: int) -> int:
    handles = _nvapi_enum_gpus()
    if not handles:
        raise RuntimeError("No NVIDIA GPUs found.")
    if gpu_index >= len(handles):
        raise IndexError(
            f"GPU index {gpu_index} out of range "
            f"(system has {len(handles)} GPU(s))."
        )
    h = handles[gpu_index]
    if h is None:
        raise RuntimeError(f"GPU {gpu_index} handle is NULL.")
    return h


def _get_clock_masks(handle: int) -> _NV_GPU_CLOCK_MASKS:
    # Increased sleep to 0.05s as 0.01s wasn't enough for very long tuning runs
    time.sleep(0.05)
    masks = _NV_GPU_CLOCK_MASKS()
    masks.version = ctypes.sizeof(masks) | (1 << 16)
    f = _qif(
        _ID_GetClockBoostMask,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_CLOCK_MASKS)],
    )
    rc = f(ctypes.c_void_p(handle), ctypes.byref(masks))
    if rc != 0:
        raise RuntimeError(f"NvAPI_GPU_GetClockBoostMask failed: {rc:#010x}")
    return masks


def _get_vfp_curve(handle: int, masks: _NV_GPU_CLOCK_MASKS) -> _NV_GPU_VFP_CURVE:
    curve = _NV_GPU_VFP_CURVE()
    curve.version = ctypes.sizeof(curve) | (1 << 16)
    ctypes.memmove(curve.mask, masks.mask, 32)
    f = _qif(
        _ID_GetVFPCurve,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_VFP_CURVE)],
    )
    rc = f(ctypes.c_void_p(handle), ctypes.byref(curve))
    if rc != 0:
        raise RuntimeError(f"NvAPI_GPU_GetVFPCurve failed: {rc:#010x}")
    return curve


def _get_clock_table(handle: int, masks: _NV_GPU_CLOCK_MASKS) -> _NV_GPU_CLOCK_TABLE:
    table = _NV_GPU_CLOCK_TABLE()
    table.version = ctypes.sizeof(table) | (1 << 16)
    ctypes.memmove(table.mask, masks.mask, 32)
    f = _qif(
        _ID_GetClockBoostTable,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_CLOCK_TABLE)],
    )
    rc = f(ctypes.c_void_p(handle), ctypes.byref(table))
    if rc != 0:
        raise RuntimeError(f"NvAPI_GPU_GetClockBoostTable failed: {rc:#010x}")
    return table


def _set_clock_table(handle: int, table: _NV_GPU_CLOCK_TABLE) -> None:
    f = _qif(
        _ID_SetClockBoostTable,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_CLOCK_TABLE)],
    )
    rc = f(ctypes.c_void_p(handle), ctypes.byref(table))
    if rc != 0:
        raise RuntimeError(f"NvAPI_GPU_SetClockBoostTable failed: {rc:#010x}")


# ── internal helpers ──────────────────────────────────────────────────────────

def _active_core_indices(masks: _NV_GPU_CLOCK_MASKS, vfp: _NV_GPU_VFP_CURVE) -> List[int]:
    """Return the slot indices of enabled core-clock (clockType==0) bins."""
    return [
        i for i in range(255)
        if masks.clocks[i].enabled == 1 and vfp.clocks[i].clockType == 0
    ]


def _read_active_bins(handle: int) -> Tuple[_NV_GPU_CLOCK_MASKS, _NV_GPU_VFP_CURVE, List[int]]:
    """Return (masks, vfp_curve, active_indices)."""
    masks = _get_clock_masks(handle)
    vfp   = _get_vfp_curve(handle, masks)
    idx   = _active_core_indices(masks, vfp)
    if not idx:
        raise RuntimeError(
            "No active core-clock voltage bins found — "
            "driver or GPU may not support VFP curve editing."
        )
    return masks, vfp, idx


def _reset_curve(handle: int) -> None:
    """Zero all frequency deltas (restore driver default frequencies)."""
    masks = _get_clock_masks(handle)
    vfp   = _get_vfp_curve(handle, masks)
    table = _get_clock_table(handle, masks)
    for i in _active_core_indices(masks, vfp):
        table.clocks[i].frequencyDeltaKHz = 0
    _set_clock_table(handle, table)


def _probe_thermal_mask(handle: int) -> int:
    """Probe which thermal sensor bits are supported by trying each one."""
    f = _qif(
        _ID_GetThermalSensors,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_THERMAL_SENSORS)],
    )
    mask = 0
    for bit in range(32):
        sensors = _NV_GPU_THERMAL_SENSORS()
        sensors.version = ctypes.sizeof(sensors) | (2 << 16)
        sensors.mask = 1 << bit
        rc = f(ctypes.c_void_p(handle), ctypes.byref(sensors))
        if rc != 0:
            break
        mask |= 1 << bit
    return mask
