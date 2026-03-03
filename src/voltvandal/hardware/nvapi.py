"""
nvapi_curve.py — pure-Python NVAPI VF-curve interface for VoltVandal
---------------------------------------------------------------------
Uses direct in-process calls to
nvapi64.dll via ctypes.  Only the two operations VoltVandal needs are
implemented:

    dump_curve(gpu_index, out_csv)   — read GPU VF curve → write CSV
    apply_curve(gpu_index, in_csv)   — read CSV → apply to GPU

Design is a faithful Python translation of the relevant C++ reference
implementation from buswedg (MIT / public domain).

Struct layouts and QueryInterface IDs cross-referenced against:
  - https://github.com/Demion/nvapioc
  - https://github.com/vertcoin-project/vertminer-nvidia

Requirements:
  - Windows only (nvapi64.dll / nvapi.dll)
  - NVIDIA driver installed
  - Administrator privilege at runtime
  - Python 3.7+, no third-party packages

GPU index convention:
  gpu_index=0 → the first GPU returned by NvAPI_EnumPhysicalGPUs
  (matches VoltVandal --gpu 0).  The C++ tool used PCI bus-ID as the
  index; this Python version uses the friendlier enumeration position.
"""

from __future__ import annotations

import csv
import ctypes
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ── platform guard ────────────────────────────────────────────────────────────
if os.name != "nt":
    raise ImportError("nvapi_curve is Windows-only (requires nvapi64.dll)")

# ── NVAPI QueryInterface IDs ──────────────────────────────────────────────────
_ID_Initialize              = 0x0150E828
_ID_Unload                  = 0xD22BDD7E
_ID_EnumPhysicalGPUs        = 0xE5AC921F
_ID_GetClockBoostMask       = 0x507B4B59
_ID_GetVFPCurve             = 0x21537AD4
_ID_GetClockBoostTable      = 0x23F1B133
_ID_SetClockBoostTable      = 0x0733E009
_ID_GetThermalSensors        = 0x65FE3AAD
# Voltage reading — tried in order by get_current_voltage_mv():
#   1. GetCoreVoltage           0x58337FA3  — simple NvU32* getter (mV or µV)
#   2. GetVoltageDomainsStatus  0xC16C7E2C  — struct-based domain table
#                                             (0x7296F6D4 was wrong, returned NULL)
#   3. ClientVoltRailsGetStatus 0x465F9BCF  — per-rail struct (Pack=1)
_ID_GetCoreVoltage           = 0x58337FA3
_ID_GetVoltDomainsStatus     = 0xC16C7E2C
_ID_ClientVoltRailsGetStatus = 0x465F9BCF
# Performance / telemetry
_ID_GetPerfDecreaseInfo      = 0x7F7F4600  # NvAPI_GPU_GetPerfDecreaseInfo — throttle bitmask
_ID_GetCurrentPstate         = 0x927DA4F6  # NvAPI_GPU_GetCurrentPstate — P0/P8/etc.
_ID_ClientPowerTopoGetStatus = 0xEDCF624E  # NvAPI_GPU_ClientPowerTopologyGetStatus

# ── struct definitions (mirrors the reference C++ layout) ─────────────────────

class _MaskEntry(ctypes.Structure):
    """One entry in NV_GPU_CLOCK_MASKS.clocks[255]."""
    _fields_ = [
        ("clockType", ctypes.c_uint32),
        ("enabled",   ctypes.c_uint8),
        ("unknown2",  ctypes.c_uint8 * 19),  # pad to 24 bytes
    ]

class _NV_GPU_CLOCK_MASKS(ctypes.Structure):
    """
    C layout (6188 bytes):
      uint  version
      u8    mask[32]
      u8    unknown1[32]
      _MaskEntry clocks[255]   // 255 × 24 = 6120
    """
    _pack_ = 8
    _fields_ = [
        ("version",  ctypes.c_uint32),
        ("mask",     ctypes.c_uint8 * 32),
        ("unknown1", ctypes.c_uint8 * 32),
        ("clocks",   _MaskEntry * 255),
    ]


class _VFPEntry(ctypes.Structure):
    """One entry in NV_GPU_VFP_CURVE.clocks[255]."""
    _fields_ = [
        ("clockType",    ctypes.c_uint32),
        ("frequencyKHz", ctypes.c_uint32),
        ("voltageUV",    ctypes.c_uint32),
        ("unknown2",     ctypes.c_uint8 * 16),  # pad to 28 bytes
    ]

class _NV_GPU_VFP_CURVE(ctypes.Structure):
    """
    C layout (7208 bytes):
      uint  version
      u8    mask[32]
      u8    unknown1[32]
      _VFPEntry clocks[255]    // 255 × 28 = 7140
    """
    _pack_ = 8
    _fields_ = [
        ("version",  ctypes.c_uint32),
        ("mask",     ctypes.c_uint8 * 32),
        ("unknown1", ctypes.c_uint8 * 32),
        ("clocks",   _VFPEntry * 255),
    ]


class _TableEntry(ctypes.Structure):
    """One entry in NV_GPU_CLOCK_TABLE.clocks[255]."""
    _fields_ = [
        ("clockType",         ctypes.c_uint32),
        ("unknown2",          ctypes.c_uint8 * 16),  # pad to reach delta at +20
        ("frequencyDeltaKHz", ctypes.c_int32),
        ("unknown3",          ctypes.c_uint8 * 12),  # pad to 36 bytes total
    ]

class _NV_GPU_CLOCK_TABLE(ctypes.Structure):
    """
    C layout (9248 bytes):
      uint  version
      u8    mask[32]
      u8    unknown1[32]
      _TableEntry clocks[255]  // 255 × 36 = 9180
    """
    _pack_ = 8
    _fields_ = [
        ("version",  ctypes.c_uint32),
        ("mask",     ctypes.c_uint8 * 32),
        ("unknown1", ctypes.c_uint8 * 32),
        ("clocks",   _TableEntry * 255),
    ]


class _NV_GPU_THERMAL_SENSORS(ctypes.Structure):
    """
    Undocumented NvAPI struct for GPU thermal sensors.
    Discovered via LibreHardwareMonitor (NvApi.cs).
    Pack=8, version=2.  Temperatures are fixed-point: value / 256.0 = °C.

    Sensor indices (Ampere / RTX 30xx):
      [0] = GPU edge (same as NVML TEMPERATURE_GPU)
      [1] = GPU hotspot / junction
      [9] = VRAM junction temperature
    """
    _pack_ = 8
    _fields_ = [
        ("version",      ctypes.c_uint32),
        ("mask",         ctypes.c_uint32),
        ("reserved",     ctypes.c_int32 * 8),
        ("temperatures", ctypes.c_int32 * 32),
    ]


_NVAPI_MAX_GPU_VOLT_DOMAINS = 16


class _VoltDomainEntry(ctypes.Structure):
    """
    One entry in NV_GPU_VOLTAGE_DOMAINS_STATUS.entries[].

    domain == 0  → GPU core voltage domain.
    current_mv   → present core voltage in millivolts (e.g. 862).

    Layout (Pack=8, all fields uint32 → natural align = 4 < pack, no padding):
      domain(4) + flags(4) + current_mv(4) + _reserved[8](32) = 44 bytes.
    """
    _pack_ = 8
    _fields_ = [
        ("domain",     ctypes.c_uint32),
        ("flags",      ctypes.c_uint32),
        ("current_mv", ctypes.c_uint32),   # millivolts
        ("_reserved",  ctypes.c_uint32 * 8),
    ]


class _NV_GPU_VOLTAGE_DOMAINS_STATUS(ctypes.Structure):
    """
    Response struct for NvAPI_GPU_GetVoltageDomainsStatus (0xC16C7E2C).

    version  = sizeof(struct) | (1 << 16)
    count    = number of populated entries
    entries  = up to _NVAPI_MAX_GPU_VOLT_DOMAINS domain records
    """
    _pack_ = 8
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags",   ctypes.c_uint32),
        ("count",   ctypes.c_uint32),
        ("entries", _VoltDomainEntry * _NVAPI_MAX_GPU_VOLT_DOMAINS),
    ]


# ── Voltage-rail structs (ClientVoltRailsGetStatus) ──────────────────────────

_NVAPI_MAX_VOLT_RAILS = 16


class _NV_GPU_VOLT_RAIL_ENTRY(ctypes.Structure):
    """
    One voltage-rail entry in NV_GPU_CLIENT_VOLT_RAILS_STATUS.

    rail_id == 0  → GPU core voltage rail.
    volt_uv       → current rail voltage in microvolts.

    Reverse-engineered from LibreHardwareMonitor (NvApi.cs) and nvapioc.
    Pack=1, 44 bytes per entry:
      rail_id(4) + flags(4) + volt_uv(4) + unknown[8](32) = 44 bytes.
    """
    _pack_ = 1
    _fields_ = [
        ("rail_id", ctypes.c_uint32),
        ("flags",   ctypes.c_uint32),
        ("volt_uv", ctypes.c_uint32),
        ("unknown", ctypes.c_uint32 * 8),
    ]


class _NV_GPU_CLIENT_VOLT_RAILS_STATUS(ctypes.Structure):
    """
    Response struct for NvAPI_GPU_ClientVoltRailsGetStatus (0x465F9BCF).

    version  = sizeof(struct) | (1 << 16)
    num_rails = number of populated rail entries
    """
    _pack_ = 1
    _fields_ = [
        ("version",   ctypes.c_uint32),
        ("flags",     ctypes.c_uint32),
        ("num_rails", ctypes.c_uint32),
        ("rails",     _NV_GPU_VOLT_RAIL_ENTRY * _NVAPI_MAX_VOLT_RAILS),
    ]


# ── Power topology structs ────────────────────────────────────────────────────

_NVAPI_MAX_POWER_TOPO_CHANNELS = 4


class _NV_GPU_POWER_TOPO_CHANNEL(ctypes.Structure):
    """
    One power channel entry in NV_GPU_CLIENT_POWER_TOPOLOGY_STATUS.

    channelId:
      0 = Total board (GPU + fans + everything)
      1 = GPU die
      2 = Memory

    powerMw is milliwatts.  Discovered via LibreHardwareMonitor (NvApi.cs).
    """
    _pack_ = 1
    _fields_ = [
        ("channelId", ctypes.c_uint32),
        ("flags",     ctypes.c_uint32),
        ("powerMw",   ctypes.c_uint32),
        ("unknown",   ctypes.c_uint32),
    ]


class _NV_GPU_POWER_TOPO_STATUS(ctypes.Structure):
    """
    Response struct for NvAPI_GPU_ClientPowerTopologyGetStatus (0xEDCF624E).

    version = sizeof(struct) | (1 << 16)
    """
    _pack_ = 1
    _fields_ = [
        ("version",  ctypes.c_uint32),
        ("flags",    ctypes.c_uint32),
        ("channels", _NV_GPU_POWER_TOPO_CHANNEL * _NVAPI_MAX_POWER_TOPO_CHANNELS),
    ]


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


# ── public API ────────────────────────────────────────────────────────────────

_thermal_mask_cache: dict[int, int] = {}


def get_thermal_sensors(gpu_index: int) -> dict[str, float | None]:
    """
    Read GPU thermal sensors via undocumented NvAPI_GPU_ThermalGetSensors.

    Returns a dict with keys:
        "gpu_edge_c"       — GPU edge temp (°C) or None
        "hotspot_c"        — GPU hotspot / junction temp (°C) or None
        "vram_junction_c"  — VRAM junction temp (°C) or None

    Raises RuntimeError if NvAPI call fails entirely.
    Returns None values for sensors that aren't populated.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)

    # Probe and cache the supported sensor mask
    if handle not in _thermal_mask_cache:
        _thermal_mask_cache[handle] = _probe_thermal_mask(handle)
    mask = _thermal_mask_cache[handle]

    if mask == 0:
        return {"gpu_edge_c": None, "hotspot_c": None, "vram_junction_c": None}

    sensors = _NV_GPU_THERMAL_SENSORS()
    sensors.version = ctypes.sizeof(sensors) | (2 << 16)
    sensors.mask = mask

    f = _qif(
        _ID_GetThermalSensors,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_THERMAL_SENSORS)],
    )
    rc = f(ctypes.c_void_p(handle), ctypes.byref(sensors))
    if rc != 0:
        raise RuntimeError(f"NvAPI_GPU_GetThermalSensors failed: {rc:#010x}")

    def _read(idx: int) -> float | None:
        raw = sensors.temperatures[idx]
        if raw == 0:
            return None
        return raw / 256.0

    return {
        "gpu_edge_c": _read(0),
        "hotspot_c": _read(1),
        "vram_junction_c": _read(9),  # index 9 for Ampere (RTX 30xx)
    }


def _mv_from_raw(v: int) -> Optional[int]:
    """Normalise a raw voltage value to millivolts.

    Some drivers return millivolts directly (600–1200 range), others return
    microvolts (600_000–1_200_000 range).  Values > 5_000 are treated as µV.
    Returns None for zero / implausible readings.
    """
    if v <= 0:
        return None
    if v > 5_000:
        v = v // 1000
    # Sanity-check: GPU core voltage should be 400–1500 mV.
    if not (400 <= v <= 1_500):
        return None
    return v


def get_current_voltage_mv(gpu_index: int) -> Optional[int]:
    """
    Read the current GPU core voltage in millivolts.

    Tries up to three undocumented NvAPI methods in order of reliability:

      1. NvAPI_GPU_GetCoreVoltage (0x58337FA3) — simple NvU32 pointer; returns
         mV or µV directly.  Most likely to work on modern drivers (RTX series).
      2. NvAPI_GPU_GetVoltageDomainsStatus (0xC16C7E2C) — versioned struct that
         returns per-domain voltages; domain 0 = GPU core.
      3. NvAPI_GPU_ClientVoltRailsGetStatus (0x465F9BCF) — per-rail struct
         (Pack=1); rail_id 0 = GPU core voltage rail; volt_uv in microvolts.

    Returns the core voltage in mV (e.g. 875), or None if no method succeeds
    or no valid reading is available.

    Callers wanting a best-effort reading should catch all exceptions:

        try:
            mv = get_current_voltage_mv(0)
        except Exception:
            mv = None
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)

    # ── Attempt 1: NvAPI_GPU_GetCoreVoltage (0x58337FA3) ─────────────────────
    try:
        f1 = _qif(
            _ID_GetCoreVoltage,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        raw = ctypes.c_uint32(0)
        rc = f1(ctypes.c_void_p(handle), ctypes.byref(raw))
        if rc == 0:
            mv = _mv_from_raw(raw.value)
            if mv is not None:
                return mv
    except Exception:
        pass  # function not available on this driver; try next

    # ── Attempt 2: NvAPI_GPU_GetVoltageDomainsStatus (0xC16C7E2C) ────────────
    try:
        f2 = _qif(
            _ID_GetVoltDomainsStatus,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_VOLTAGE_DOMAINS_STATUS)],
        )
        status = _NV_GPU_VOLTAGE_DOMAINS_STATUS()
        status.version = ctypes.sizeof(status) | (1 << 16)
        rc = f2(ctypes.c_void_p(handle), ctypes.byref(status))
        if rc == 0:
            n = min(status.count, _NVAPI_MAX_GPU_VOLT_DOMAINS)
            for i in range(n):
                entry = status.entries[i]
                if entry.domain == 0 and entry.current_mv > 0:
                    mv = _mv_from_raw(entry.current_mv)
                    if mv is not None:
                        return mv
    except Exception:
        pass

    # ── Attempt 3: NvAPI_GPU_ClientVoltRailsGetStatus (0x465F9BCF) ───────────
    try:
        f3 = _qif(
            _ID_ClientVoltRailsGetStatus,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_CLIENT_VOLT_RAILS_STATUS)],
        )
        rails_status = _NV_GPU_CLIENT_VOLT_RAILS_STATUS()
        rails_status.version = ctypes.sizeof(rails_status) | (1 << 16)
        rc = f3(ctypes.c_void_p(handle), ctypes.byref(rails_status))
        if rc == 0:
            n = min(rails_status.num_rails, _NVAPI_MAX_VOLT_RAILS)
            for i in range(n):
                rail = rails_status.rails[i]
                if rail.rail_id == 0 and rail.volt_uv > 0:
                    mv = _mv_from_raw(rail.volt_uv)
                    if mv is not None:
                        return mv
    except Exception:
        pass

    return None


def get_perf_decrease_info(gpu_index: int) -> Optional[int]:
    """
    Return the NvAPI_GPU_GetPerfDecreaseInfo bitmask (0x7F7F4600).

    The bitmask encodes why GPU performance was reduced.  Known bits:
      0x01 = Insufficient power (power connector / supply)
      0x04 = AC power level
      0x10 = Power brake (external power-brake signal)
      0x40 = Temperature (thermal slowdown)

    Returns an int (may be 0 = no decrease active), or None if the NvAPI
    function is unavailable on this driver.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)
    try:
        f = _qif(
            _ID_GetPerfDecreaseInfo,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        info = ctypes.c_uint32(0)
        rc = f(ctypes.c_void_p(handle), ctypes.byref(info))
        if rc == 0:
            return info.value
    except Exception:
        pass
    return None


def get_current_pstate(gpu_index: int) -> Optional[int]:
    """
    Return the current GPU P-state index via NvAPI_GPU_GetCurrentPstate
    (0x927DA4F6).

    P0 = maximum performance, P8 = idle / low power.
    Returns an int (0, 1, 2, 8, …) or None if unavailable.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)
    try:
        f = _qif(
            _ID_GetCurrentPstate,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        pstate = ctypes.c_uint32(0)
        rc = f(ctypes.c_void_p(handle), ctypes.byref(pstate))
        if rc == 0:
            return pstate.value
    except Exception:
        pass
    return None


def get_power_topology_mw(gpu_index: int) -> Optional[dict]:
    """
    Return per-rail power in milliwatts via NvAPI_GPU_ClientPowerTopologyGetStatus
    (0xEDCF624E).

    Returns a dict with keys:
      "total_mw"  — total board power (GPU + fans + everything), mW
      "gpu_mw"    — GPU die power, mW
      "mem_mw"    — memory power, mW

    Missing channels are absent from the dict.  Returns None if the call
    fails or the function is not available.

    Channel IDs: 0 = total board, 1 = GPU die, 2 = memory.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)
    try:
        f = _qif(
            _ID_ClientPowerTopoGetStatus,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_POWER_TOPO_STATUS)],
        )
        topo = _NV_GPU_POWER_TOPO_STATUS()
        topo.version = ctypes.sizeof(topo) | (1 << 16)
        rc = f(ctypes.c_void_p(handle), ctypes.byref(topo))
        if rc != 0:
            return None
        _CHAN_NAMES = {0: "total_mw", 1: "gpu_mw", 2: "mem_mw"}
        result: dict = {}
        for i in range(_NVAPI_MAX_POWER_TOPO_CHANNELS):
            ch = topo.channels[i]
            name = _CHAN_NAMES.get(ch.channelId)
            if name and ch.powerMw > 0:
                result[name] = ch.powerMw
        return result if result else None
    except Exception:
        return None


def dump_curve(gpu_index: int, out_csv: "str | Path") -> None:
    """
    Read the current VF curve from *gpu_index* and write it to *out_csv*.

    CSV format:
        voltageUV,frequencyKHz
        850000,1830000
        ...

    Raises RuntimeError on any NVAPI failure.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)

    masks, vfp, idx = _read_active_bins(handle)

    out_csv = Path(out_csv)
    with out_csv.open("w", newline="") as fh:
        fh.write("voltageUV,frequencyKHz\n")
        for i in idx:
            fh.write(f"{vfp.clocks[i].voltageUV},{vfp.clocks[i].frequencyKHz}\n")


def apply_curve(gpu_index: int, in_csv: "str | Path") -> None:
    """
    Apply a VF curve from *in_csv* to *gpu_index*.

    The CSV must have *voltageUV* and *frequencyKHz* columns (same format
    produced by dump_curve).  Frequency values are treated as **absolute**
    targets.  Internally this:

      1. Resets all frequency deltas to zero (driver defaults).
      2. Reads the resulting default absolute frequencies.
      3. Computes per-bin deltas: Δ = target_freq − default_freq.
      4. Writes the delta table back via SetClockBoostTable.

    Only bins whose voltage appears in the CSV are modified; all others
    retain zero delta (driver default).

    Raises RuntimeError on any NVAPI failure or CSV format error.
    """
    in_csv = Path(in_csv)
    csv_by_voltage: dict[int, int] = {}  # voltageUV → target frequencyKHz
    with in_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        norm = {k.lower(): k for k in (reader.fieldnames or [])}
        col_v = norm.get("voltageuv")
        col_f = norm.get("frequencykhz")
        if not col_v or not col_f:
            raise ValueError(
                f"CSV must have voltageUV and frequencyKHz columns: {in_csv}"
            )
        for row in reader:
            csv_by_voltage[int(row[col_v])] = int(row[col_f])

    if not csv_by_voltage:
        raise ValueError(f"No curve points found in {in_csv}")

    _nvapi_init()
    handle = _get_handle(gpu_index)

    # Step 1: zero all deltas → driver restores base frequencies
    _reset_curve(handle)

    # Step 2: read the default absolute frequencies (after reset)
    masks_def, vfp_def, idx_def = _read_active_bins(handle)
    default_by_voltage: dict[int, int] = {
        vfp_def.clocks[i].voltageUV: vfp_def.clocks[i].frequencyKHz
        for i in idx_def
    }

    # Step 3: fetch the delta table (mask already zeroed from step 1)
    masks2 = _get_clock_masks(handle)
    vfp2   = _get_vfp_curve(handle, masks2)
    table  = _get_clock_table(handle, masks2)

    # Step 4: compute and write deltas for CSV-supplied voltages
    for i in _active_core_indices(masks2, vfp2):
        volt = vfp2.clocks[i].voltageUV
        if volt in csv_by_voltage and volt in default_by_voltage:
            table.clocks[i].frequencyDeltaKHz = (
                csv_by_voltage[volt] - default_by_voltage[volt]
            )

    _set_clock_table(handle, table)


# ── CLI shim — compact -curve interface ───────────────────────────────────────
#
#   python nvapi_curve.py -curve <gpu> -1 <out.csv>   (dump)
#   python nvapi_curve.py -curve <gpu>  1 <in.csv>    (apply)
#   python nvapi_curve.py -curve <gpu>  0              (reset to defaults)
#
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 3 and args[0].lower() == "-curve":
        try:
            gpu = int(args[1])
            op  = int(args[2])
        except ValueError:
            print("Usage: nvapi_curve.py -curve <gpu> <op> [file]",
                  file=sys.stderr)
            sys.exit(1)

        if op == -1:
            if len(args) < 4:
                print("-curve <gpu> -1 requires a filename", file=sys.stderr)
                sys.exit(1)
            dump_curve(gpu, args[3])
        elif op == 1:
            if len(args) < 4:
                print("-curve <gpu> 1 requires a filename", file=sys.stderr)
                sys.exit(1)
            apply_curve(gpu, args[3])
        elif op == 0:
            _nvapi_init()
            _reset_curve(_get_handle(gpu))
        else:
            print(f"Unknown operation: {op}", file=sys.stderr)
            sys.exit(1)
    else:
        print(
            "Usage:\n"
            "  nvapi_curve.py -curve <gpu> -1 <out.csv>   (dump VF curve)\n"
            "  nvapi_curve.py -curve <gpu>  1 <in.csv>    (apply VF curve)\n"
            "  nvapi_curve.py -curve <gpu>  0             (reset to defaults)",
            file=sys.stderr,
        )
        sys.exit(1)
