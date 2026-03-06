#!/usr/bin/env python3
"""
Quick NVAPI voltage-function probe using IDs from doc/NVAPI-key-table.html.

What it does:
1) Parses NvAPI names + IDs from the local key-table HTML.
2) Resolves voltage-related IDs via nvapi_QueryInterface.
3) Calls a few read-only voltage APIs with guarded struct-version brute-force.

Run as Administrator on Windows.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if os.name != "nt":
    raise SystemExit("Windows only (requires nvapi64.dll / nvapi.dll).")

_ID_INITIALIZE = 0x0150E828
_ID_ENUM_PHYSICAL_GPUS = 0xE5AC921F
_NVAPI_OK = 0


def _load_nvapi():
    dll_name = "nvapi64.dll" if ctypes.sizeof(ctypes.c_void_p) == 8 else "nvapi.dll"
    try:
        return ctypes.WinDLL(dll_name)
    except OSError as exc:
        raise SystemExit(f"Failed to load {dll_name}: {exc}")


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


def _parse_key_table(path: Path) -> Dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pairs = re.findall(
        r"<td>\s*(NvAPI_[^<]+?)\s*</td>\s*<td>\s*(0x[0-9A-Fa-f]+)\s*</td>",
        text,
        flags=re.S,
    )
    out: Dict[str, int] = {}
    for name, hex_id in pairs:
        out[name.strip()] = int(hex_id, 16)
    return out


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


def _scan_plausible_voltages(buf: bytes) -> List[Tuple[int, int, str]]:
    hits: List[Tuple[int, int, str]] = []
    for off in range(0, max(0, len(buf) - 4), 4):
        val = struct.unpack_from("<I", buf, off)[0]
        mv = _normalize_mv(val)
        if mv is not None:
            hits.append((off, mv, "u32"))
    return hits


def _probe_u32_voltage(dll, func_id: int, gpu_handle: int) -> Dict[str, object]:
    fn = _make_fn(
        dll,
        func_id,
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
    )
    if fn is None:
        return {"resolved": False}
    raw = ctypes.c_uint32(0)
    rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(raw)))
    return {
        "resolved": True,
        "rc": f"0x{rc & 0xFFFFFFFF:08X}",
        "raw": int(raw.value),
        "mv": _normalize_mv(int(raw.value)),
    }


def _probe_struct_bruteforce(dll, func_id: int, gpu_handle: int) -> Dict[str, object]:
    fn = _make_fn(dll, func_id, ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p])
    if fn is None:
        return {"resolved": False, "hits": []}

    sizes = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    versions = range(0, 16)
    hits_out: List[Dict[str, object]] = []

    for sz in sizes:
        for ver in versions:
            buf = (ctypes.c_ubyte * sz)()
            # Try common NVAPI version encodings.
            for vword in (sz | (ver << 16), (ver << 24) | sz):
                version = ctypes.c_uint32(vword)
                ctypes.memmove(ctypes.addressof(buf), ctypes.byref(version), 4)
                rc = int(fn(ctypes.c_void_p(gpu_handle), ctypes.byref(buf)))
                if rc != _NVAPI_OK:
                    continue
                data = bytes(buf)
                volts = _scan_plausible_voltages(data)
                if volts:
                    top = [{"offset": o, "mv": mv, "kind": k} for (o, mv, k) in volts[:12]]
                    hits_out.append(
                        {
                            "size": sz,
                            "version": ver,
                            "version_word": f"0x{vword:08X}",
                            "rc": f"0x{rc & 0xFFFFFFFF:08X}",
                            "voltage_hits": top,
                        }
                    )
    return {"resolved": True, "hits": hits_out}


def _sweep_resolved_ids(dll, ids: Dict[str, int], keywords: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    kws = [k.lower().strip() for k in keywords if k.strip()]
    for name, fid in ids.items():
        n = name.lower()
        if kws and not any(k in n for k in kws):
            continue
        ptr = _resolve_ptr(dll, fid)
        if ptr:
            out.append({"name": name, "id": f"0x{fid:08X}", "ptr": f"0x{ptr:016X}"})
    out.sort(key=lambda x: x["name"])
    return out


def _sample_live_voltage(seconds: int, gpu_index: int, do_load: bool) -> Dict[str, object]:
    """
    Sample nvapi_curve.get_current_voltage_mv over time.
    Optional cupy load can expose telemetry that is hidden at idle.
    """
    samples: List[Dict[str, object]] = []
    stop_ev = threading.Event()

    def _load_worker():
        try:
            import cupy as cp  # type: ignore
            with cp.cuda.Device(gpu_index):
                a = cp.random.random((2048, 2048), dtype=cp.float32)
                b = cp.random.random((2048, 2048), dtype=cp.float32)
                out = cp.zeros((2048, 2048), dtype=cp.float32)
                while not stop_ev.is_set():
                    cp.matmul(a, b, out=out)
                    cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass

    t: Optional[threading.Thread] = None
    if do_load:
        t = threading.Thread(target=_load_worker, daemon=True)
        t.start()
        time.sleep(0.5)

    try:
        import nvapi_curve as nvc  # type: ignore
    except Exception as exc:
        return {"error": f"nvapi_curve import failed: {exc}", "samples": samples}

    t_end = time.time() + max(1, seconds)
    while time.time() < t_end:
        mv = None
        try:
            mv = nvc.get_current_voltage_mv(gpu_index)
        except Exception:
            pass
        samples.append({"t": round(time.time(), 3), "mv": mv})
        time.sleep(0.5)

    stop_ev.set()
    if t is not None:
        t.join(timeout=1.0)

    valid = [s["mv"] for s in samples if isinstance(s.get("mv"), int)]
    return {
        "seconds": seconds,
        "load": do_load,
        "count": len(samples),
        "valid_count": len(valid),
        "min_mv": min(valid) if valid else None,
        "max_mv": max(valid) if valid else None,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe NVAPI voltage-related QueryInterface IDs")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    ap.add_argument(
        "--table",
        type=Path,
        default=Path("doc") / "NVAPI-key-table.html",
        help="Path to NVAPI key table HTML",
    )
    ap.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument(
        "--sweep-keywords",
        default="volt,voltage,pmumon,sample,clockclkvolt",
        help="Comma-separated keywords for QueryInterface resolve sweep",
    )
    ap.add_argument(
        "--sample-seconds",
        type=int,
        default=0,
        help="If >0, sample nvapi_curve.get_current_voltage_mv for this many seconds",
    )
    ap.add_argument(
        "--sample-load",
        action="store_true",
        help="Apply a small cupy load during --sample-seconds",
    )
    args = ap.parse_args()

    if not args.table.exists():
        raise SystemExit(f"Key table not found: {args.table}")

    ids = _parse_key_table(args.table)
    dll = _load_nvapi()
    gpu_handle = _init_and_get_gpu_handle(dll, args.gpu)

    targets = [
        "NvAPI_GPU_GetCoreVoltage",
        "NvAPI_GPU_GetVoltageDomainsStatus",
        "NvAPI_GPU_GetVoltages",
        "NvAPI_GPU_GetVoltagesInternal",
        "NvAPI_GPU_VoltVoltRailsGetStatus",
        "NvAPI_GPU_ClientVoltRailsGetStatus",
        "NvAPI_GPU_VoltPmumonVoltRailsGetSamples",
        "NvAPI_GPU_ClockClkVoltControllerGetStatus",
    ]

    result: Dict[str, object] = {
        "gpu_index": args.gpu,
        "table_entries": len(ids),
        "resolved_sweep": [],
        "targets": {},
    }

    sweep_keywords = [k.strip() for k in args.sweep_keywords.split(",")]
    result["resolved_sweep"] = _sweep_resolved_ids(dll, ids, sweep_keywords)

    for name in targets:
        fid = ids.get(name)
        if fid is None:
            result["targets"][name] = {"in_table": False}
            continue

        if name == "NvAPI_GPU_GetCoreVoltage":
            probe = _probe_u32_voltage(dll, fid, gpu_handle)
        else:
            probe = _probe_struct_bruteforce(dll, fid, gpu_handle)
        probe["in_table"] = True
        probe["id"] = f"0x{fid:08X}"
        result["targets"][name] = probe

    # compact console summary
    print(f"GPU index: {args.gpu}")
    print(f"Resolved sweep ({len(result['resolved_sweep'])} hits):")
    for item in result["resolved_sweep"][:25]:
        print(f"  - {item['name']:<55} {item['id']}")
    if len(result["resolved_sweep"]) > 25:
        print(f"  ... ({len(result['resolved_sweep']) - 25} more)")

    print("Voltage target summary:")
    for name in targets:
        entry = result["targets"].get(name, {})
        if not entry.get("in_table"):
            print(f"  - {name:<45}  not in key table")
            continue
        rid = entry.get("id", "0x????????")
        if not entry.get("resolved"):
            print(f"  - {name:<45}  {rid}  unresolved")
            continue
        if name == "NvAPI_GPU_GetCoreVoltage":
            mv = entry.get("mv")
            raw = entry.get("raw")
            print(f"  - {name:<45}  {rid}  rc={entry.get('rc')}  raw={raw}  mv={mv}")
        else:
            hits = entry.get("hits", [])
            print(f"  - {name:<45}  {rid}  success_variants={len(hits)}")

    print("\nTop recommendation:")
    core = result["targets"].get("NvAPI_GPU_GetCoreVoltage", {})
    if core and core.get("mv") is not None:
        print(f"  Use NvAPI_GPU_GetCoreVoltage (ID {core.get('id')}): {core.get('mv')} mV")
    else:
        print("  NvAPI_GPU_GetCoreVoltage did not yield a usable mV value on this driver.")
        print("  Inspect struct-hit entries in JSON for domain/rail status candidates.")

    if args.sample_seconds > 0:
        sample = _sample_live_voltage(
            seconds=args.sample_seconds,
            gpu_index=args.gpu,
            do_load=bool(args.sample_load),
        )
        result["live_sample"] = sample
        print("\nLive voltage sample:")
        if "error" in sample:
            print(f"  {sample['error']}")
        else:
            print(
                f"  load={sample['load']} count={sample['count']} "
                f"valid={sample['valid_count']} min={sample['min_mv']} max={sample['max_mv']}"
            )

    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
