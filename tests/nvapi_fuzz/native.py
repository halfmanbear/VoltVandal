"""NVAPI constants, DLL helpers, voltage decoding structs and buffer scanning."""

from __future__ import annotations

import ctypes
import os
import struct
from typing import Dict, List, Optional


if os.name != "nt":
    raise SystemExit("Windows only (requires nvapi64.dll / nvapi.dll).")


_ID_INITIALIZE = 0x0150E828
_ID_ENUM_PHYSICAL_GPUS = 0xE5AC921F
_NVAPI_OK = 0
_ID_GET_CORE_VOLTAGE = 0x58337FA3
_ID_GET_VOLT_DOMAINS_STATUS = 0xC16C7E2C
_ID_CLIENT_VOLT_RAILS_GET_STATUS = 0x465F9BCF
_ID_VOLT_VOLT_RAILS_GET_STATUS = 0x5D0634EE


def _load_nvapi():
    dll_name = "nvapi64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 else "nvapi.dll"
    return ctypes.WinDLL(dll_name)


def _resolve_ptr(dll, func_id: int) -> int:
    qi = dll.nvapi_QueryInterface
    qi.restype = ctypes.c_void_p
    qi.argtypes = [ctypes.c_uint32]
    return int(qi(func_id) or 0)


def _make_fn(dll, func_id: int, restype, argtypes):
    ptr = _resolve_ptr(dll, func_id)
    if not ptr:
        return None
    return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)


def _init_and_get_gpu_handle(dll, gpu_index: int) -> int:
    fn_init = _make_fn(dll, _ID_INITIALIZE, ctypes.c_int, [])
    if fn_init is None:
        raise RuntimeError("Failed to resolve NvAPI_Initialize.")
    rc = int(fn_init())
    if rc != _NVAPI_OK:
        raise RuntimeError(f"NvAPI_Initialize failed rc=0x{rc & 0xFFFFFFFF:08X}")

    fn_enum = _make_fn(
        dll,
        _ID_ENUM_PHYSICAL_GPUS,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
    )
    if fn_enum is None:
        raise RuntimeError("Failed to resolve NvAPI_EnumPhysicalGPUs.")

    handles = (ctypes.c_void_p * 64)()
    count = ctypes.c_uint32(0)
    rc = int(fn_enum(ctypes.cast(handles, ctypes.c_void_p), ctypes.byref(count)))
    if rc != _NVAPI_OK:
        raise RuntimeError(f"NvAPI_EnumPhysicalGPUs failed rc=0x{rc & 0xFFFFFFFF:08X}")
    if gpu_index < 0 or gpu_index >= int(count.value):
        raise RuntimeError(f"GPU index {gpu_index} out of range (count={count.value})")
    return int(ctypes.cast(handles[gpu_index], ctypes.c_void_p).value or 0)


def _normalize_mv(v: int) -> Optional[int]:
    if v <= 0:
        return None
    mv = v
    if mv > 20000:
        mv = int(round(mv / 1000.0))
    if 400 <= mv <= 2000:
        return mv
    return None


def _scan_buf_voltage_hits(buf: bytes) -> List[Dict[str, int]]:
    hits: List[Dict[str, int]] = []
    for off in range(0, max(0, len(buf) - 4), 4):
        val = struct.unpack_from("<I", buf, off)[0]
        mv = _normalize_mv(val)
        if mv is not None:
            hits.append({"offset": off, "mv": mv})
    return hits[:16]


_NVAPI_MAX_GPU_VOLT_DOMAINS = 16
_NVAPI_MAX_VOLT_RAILS = 16


class _VoltDomainEntry(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("current_mv", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NV_GPU_VOLTAGE_DOMAINS_STATUS(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("entries", _VoltDomainEntry * _NVAPI_MAX_GPU_VOLT_DOMAINS),
    ]


class _NV_GPU_VOLT_RAIL_ENTRY(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("rail_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("volt_uv", ctypes.c_uint32),
        ("unknown", ctypes.c_uint32 * 8),
    ]


class _NV_GPU_CLIENT_VOLT_RAILS_STATUS(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("num_rails", ctypes.c_uint32),
        ("rails", _NV_GPU_VOLT_RAIL_ENTRY * _NVAPI_MAX_VOLT_RAILS),
    ]
