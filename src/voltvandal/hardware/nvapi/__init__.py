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

import os

# ── platform guard ────────────────────────────────────────────────────────────
if os.name != "nt":
    raise ImportError("nvapi_curve is Windows-only (requires nvapi64.dll)")

from .curves import (
    apply_curve, apply_curve_safe, dump_curve, reset_curve, reset_curve_safe,
)
from .native import (
    _get_handle, _nvapi_enum_gpus, _nvapi_init, _qif, _read_active_bins,
)
from .telemetry import (
    get_current_pstate, get_current_voltage_mv, get_perf_decrease_info,
    get_power_topology_mw, get_thermal_sensors,
)

__all__ = [
    "apply_curve", "apply_curve_safe", "dump_curve", "reset_curve", "reset_curve_safe",
    "get_current_pstate", "get_current_voltage_mv", "get_perf_decrease_info",
    "get_power_topology_mw", "get_thermal_sensors",
    # Internal hooks that point_lock reaches through this package.
    "_get_handle", "_nvapi_enum_gpus", "_nvapi_init", "_qif", "_read_active_bins",
]
