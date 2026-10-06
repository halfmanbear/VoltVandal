"""NVAPI QueryInterface IDs and ctypes struct layouts."""

import ctypes


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
#   1. ClientVoltRailsGetStatus 0x465F9BCF  — per-rail struct (Pack=1)
#   2. GetVoltageDomainsStatus  0xC16C7E2C  — struct-based domain table
#                                             (0x7296F6D4 was wrong, returned NULL)
#   3. GetCoreVoltage           0x58337FA3  — simple NvU32* getter (mV or µV)
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
