"""Child-process probes: per-ID voltage decoders and generic signatures."""

from __future__ import annotations

import ctypes
import json
from typing import Dict, List

from .native import (
    _ID_CLIENT_VOLT_RAILS_GET_STATUS, _ID_GET_CORE_VOLTAGE, _ID_GET_VOLT_DOMAINS_STATUS,
    _ID_VOLT_VOLT_RAILS_GET_STATUS, _NVAPI_MAX_GPU_VOLT_DOMAINS, _NVAPI_MAX_VOLT_RAILS,
    _NVAPI_OK, _NV_GPU_CLIENT_VOLT_RAILS_STATUS, _NV_GPU_VOLTAGE_DOMAINS_STATUS,
    _init_and_get_gpu_handle, _load_nvapi, _normalize_mv, _resolve_ptr, _scan_buf_voltage_hits,
)


def _custom_probe_voltage_api(func_id: int, ptr: int, gpu_handle: int) -> Dict[str, object]:
    """Per-ID voltage decoders (read-only) before generic signatures."""
    # 1) direct U32 voltage getter
    if func_id == _ID_GET_CORE_VOLTAGE:
        fn = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
        )(ptr)
        out = ctypes.c_uint32(0)
        rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(out)))
        return {
            "decoder": "NvAPI_GPU_GetCoreVoltage",
            "rc": f"0x{rc & 0xFFFFFFFF:08X}",
            "raw": int(out.value),
            "mv": _normalize_mv(int(out.value)),
        }

    # 2) domains status (known from nvapi_curve.py)
    if func_id == _ID_GET_VOLT_DOMAINS_STATUS:
        fn = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(_NV_GPU_VOLTAGE_DOMAINS_STATUS),
        )(ptr)
        status = _NV_GPU_VOLTAGE_DOMAINS_STATUS()
        out = []
        for vword in (
            ctypes.sizeof(status) | (1 << 16),
            ctypes.sizeof(status) | (2 << 16),
            (1 << 24) | ctypes.sizeof(status),
        ):
            status.version = vword
            rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(status)))
            row: Dict[str, object] = {
                "version_word": f"0x{vword:08X}",
                "rc": f"0x{rc & 0xFFFFFFFF:08X}",
            }
            if rc == _NVAPI_OK:
                n = min(int(status.count), _NVAPI_MAX_GPU_VOLT_DOMAINS)
                domains = []
                for i in range(n):
                    e = status.entries[i]
                    mv = _normalize_mv(int(e.current_mv))
                    domains.append(
                        {
                            "idx": i,
                            "domain": int(e.domain),
                            "current_mv_raw": int(e.current_mv),
                            "current_mv": mv,
                        }
                    )
                row["count"] = n
                row["domains"] = domains
                # likely core = domain 0
                core_mv = None
                for d in domains:
                    if d["domain"] == 0 and isinstance(d["current_mv"], int):
                        core_mv = d["current_mv"]
                        break
                row["core_mv"] = core_mv
            out.append(row)
        return {"decoder": "NvAPI_GPU_GetVoltageDomainsStatus", "attempts": out}

    # 3) client / legacy rail status layouts (same shape tried against both IDs)
    if func_id in (_ID_CLIENT_VOLT_RAILS_GET_STATUS, _ID_VOLT_VOLT_RAILS_GET_STATUS):
        fn = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(_NV_GPU_CLIENT_VOLT_RAILS_STATUS),
        )(ptr)
        st = _NV_GPU_CLIENT_VOLT_RAILS_STATUS()
        out = []
        for vword in (
            ctypes.sizeof(st) | (1 << 16),
            ctypes.sizeof(st) | (2 << 16),
            (1 << 24) | ctypes.sizeof(st),
        ):
            st.version = vword
            rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(st)))
            row: Dict[str, object] = {
                "version_word": f"0x{vword:08X}",
                "rc": f"0x{rc & 0xFFFFFFFF:08X}",
            }
            if rc == _NVAPI_OK:
                n = min(int(st.num_rails), _NVAPI_MAX_VOLT_RAILS)
                rails = []
                for i in range(n):
                    r = st.rails[i]
                    mv = _normalize_mv(int(r.volt_uv))
                    rails.append(
                        {
                            "idx": i,
                            "rail_id": int(r.rail_id),
                            "volt_uv_raw": int(r.volt_uv),
                            "mv": mv,
                        }
                    )
                row["num_rails"] = n
                row["rails"] = rails
                core_mv = None
                for rr in rails:
                    if rr["rail_id"] == 0 and isinstance(rr["mv"], int):
                        core_mv = rr["mv"]
                        break
                row["core_mv"] = core_mv
            out.append(row)
        name = (
            "NvAPI_GPU_ClientVoltRailsGetStatus"
            if func_id == _ID_CLIENT_VOLT_RAILS_GET_STATUS
            else "NvAPI_GPU_VoltVoltRailsGetStatus"
        )
        return {"decoder": name, "attempts": out}

    return {"decoder": "none", "note": "no custom decoder for this id"}


def _probe_single(func_id: int, gpu: int, sig: str) -> Dict[str, object]:
    dll = _load_nvapi()
    gpu_handle = _init_and_get_gpu_handle(dll, gpu)
    ptr = _resolve_ptr(dll, func_id)
    if not ptr:
        return {"resolved": False}

    if sig == "custom":
        return {"resolved": True, "custom": _custom_probe_voltage_api(func_id, ptr, gpu_handle)}

    if sig == "u32_ptr":
        fn = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
        )(ptr)
        out = ctypes.c_uint32(0)
        rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(out)))
        return {
            "resolved": True,
            "rc": f"0x{rc & 0xFFFFFFFF:08X}",
            "raw": int(out.value),
            "mv": _normalize_mv(int(out.value)),
        }

    if sig == "struct_ptr":
        fn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(ptr)
        sizes = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
        versions = range(0, 12)
        hits: List[Dict[str, object]] = []
        for sz in sizes:
            for ver in versions:
                for vword in (sz | (ver << 16), (ver << 24) | sz):
                    buf = (ctypes.c_ubyte * sz)()
                    vw = ctypes.c_uint32(vword)
                    ctypes.memmove(ctypes.addressof(buf), ctypes.byref(vw), 4)
                    rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(buf)))
                    if rc != _NVAPI_OK:
                        continue
                    vh = _scan_buf_voltage_hits(bytes(buf))
                    if vh:
                        hits.append(
                            {
                                "size": sz,
                                "ver": ver,
                                "vword": f"0x{vword:08X}",
                                "rc": f"0x{rc & 0xFFFFFFFF:08X}",
                                "voltage_hits": vh,
                            }
                        )
                    else:
                        hits.append(
                            {
                                "size": sz,
                                "ver": ver,
                                "vword": f"0x{vword:08X}",
                                "rc": f"0x{rc & 0xFFFFFFFF:08X}",
                                "voltage_hits": [],
                            }
                        )
                    if len(hits) >= 12:
                        return {"resolved": True, "hits": hits}
        return {"resolved": True, "hits": hits}

    return {"error": f"unknown sig {sig}"}


def _run_child(func_id: int, gpu: int, sig: str) -> int:
    payload: Dict[str, object] = {
        "func_id": f"0x{func_id:08X}",
        "gpu": gpu,
        "sig": sig,
    }
    try:
        payload["result"] = _probe_single(func_id, gpu, sig)
        payload["status"] = "ok"
    except Exception as exc:
        payload["status"] = "exception"
        payload["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(payload), flush=True)
    return 0
