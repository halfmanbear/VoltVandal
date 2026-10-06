"""Undocumented NVAPI telemetry readers (thermals, voltage, perf state, power)."""

from __future__ import annotations

import ctypes
from typing import Optional

from .native import _nvapi_init, _get_handle, _qif, _probe_thermal_mask
from .structs import (
    _ID_GetThermalSensors, _ID_GetCoreVoltage, _ID_GetVoltDomainsStatus,
    _ID_ClientVoltRailsGetStatus, _ID_GetPerfDecreaseInfo, _ID_GetCurrentPstate,
    _ID_ClientPowerTopoGetStatus,
    _NVAPI_MAX_GPU_VOLT_DOMAINS, _NVAPI_MAX_VOLT_RAILS, _NVAPI_MAX_POWER_TOPO_CHANNELS,
    _NV_GPU_THERMAL_SENSORS, _NV_GPU_VOLTAGE_DOMAINS_STATUS,
    _NV_GPU_CLIENT_VOLT_RAILS_STATUS, _NV_GPU_POWER_TOPO_STATUS,
)


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

    Tries up to three undocumented NvAPI methods in order:

      1. NvAPI_GPU_ClientVoltRailsGetStatus (0x465F9BCF) — per-rail struct
         (Pack=1); rail_id 0 = GPU core voltage rail; volt_uv in microvolts.
      2. NvAPI_GPU_GetVoltageDomainsStatus (0xC16C7E2C) — versioned struct that
         returns per-domain voltages; domain 0 = GPU core.
      3. NvAPI_GPU_GetCoreVoltage (0x58337FA3) — simple NvU32 pointer; returns
         mV or µV directly.

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

    # ── Attempt 1: NvAPI_GPU_ClientVoltRailsGetStatus (0x465F9BCF) ───────────
    try:
        f1 = _qif(
            _ID_ClientVoltRailsGetStatus,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(_NV_GPU_CLIENT_VOLT_RAILS_STATUS)],
        )
        rails_status = _NV_GPU_CLIENT_VOLT_RAILS_STATUS()
        rails_status.version = ctypes.sizeof(rails_status) | (1 << 16)
        rc = f1(ctypes.c_void_p(handle), ctypes.byref(rails_status))
        if rc == 0:
            n = min(rails_status.num_rails, _NVAPI_MAX_VOLT_RAILS)
            for i in range(n):
                rail = rails_status.rails[i]
                if rail.rail_id == 0 and rail.volt_uv > 0:
                    mv = _mv_from_raw(rail.volt_uv)
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

    # ── Attempt 3: NvAPI_GPU_GetCoreVoltage (0x58337FA3) ─────────────────────
    try:
        f3 = _qif(
            _ID_GetCoreVoltage,
            ctypes.c_int,
            [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        raw = ctypes.c_uint32(0)
        rc = f3(ctypes.c_void_p(handle), ctypes.byref(raw))
        if rc == 0:
            mv = _mv_from_raw(raw.value)
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
