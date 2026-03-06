#!/usr/bin/env python3
"""
Hardened NVAPI probe runner.

Design:
- Parent process parses local key table and selects read-oriented targets.
- Each target/signature probe runs in a separate child process.
- Parent enforces timeout and classifies: ok / nvapi error / timeout / crash.

This is intentionally conservative and avoids setter/control APIs by default.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


def _is_safe_name(name: str) -> bool:
    low = name.lower()
    allow_tokens = ("get", "status", "info", "sample", "samples")
    deny_tokens = (
        "set",
        "control",
        "register",
        "start",
        "stop",
        "enable",
        "disable",
        "reset",
        "restore",
        "override",
        "ocscanner",
    )
    if not any(tok in low for tok in allow_tokens):
        return False
    if any(tok in low for tok in deny_tokens):
        return False
    return True


def _target_names(ids: Dict[str, int], keywords: List[str], max_targets: int) -> List[str]:
    kws = [k.lower().strip() for k in keywords if k.strip()]
    names = []
    for name in sorted(ids):
        low = name.lower()
        if kws and not any(k in low for k in kws):
            continue
        if not _is_safe_name(name):
            continue
        names.append(name)
    return names[:max_targets] if max_targets > 0 else names


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


def _spawn_child(script: Path, func_id: int, gpu: int, sig: str, timeout_s: float) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(script),
        "--child",
        "--func-id",
        f"0x{func_id:08X}",
        "--gpu",
        str(gpu),
        "--sig",
        sig,
    ]
    t0 = time.time()
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    elapsed = time.time() - t0
    out = (cp.stdout or "").strip()
    err = (cp.stderr or "").strip()

    rec: Dict[str, object] = {
        "elapsed_s": round(elapsed, 3),
        "returncode": cp.returncode,
        "stderr": err[:8000],
    }

    # Crash classification
    if cp.returncode < 0:
        rec["class"] = "crash_signal"
    elif cp.returncode not in (0,):
        rec["class"] = "child_nonzero"
    else:
        rec["class"] = "ok"

    try:
        rec["child"] = json.loads(out.splitlines()[-1]) if out else {}
    except Exception:
        rec["child"] = {"status": "bad_json", "stdout_tail": out[-8000:]}
    return rec


def _spawn_child_timeout_safe(
    script: Path, func_id: int, gpu: int, sig: str, timeout_s: float
) -> Dict[str, object]:
    try:
        return _spawn_child(script, func_id, gpu, sig, timeout_s)
    except subprocess.TimeoutExpired as exc:
        return {
            "class": "timeout",
            "elapsed_s": timeout_s,
            "returncode": None,
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "child": {"status": "timeout"},
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Safe NVAPI fuzz/probe harness")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    ap.add_argument(
        "--table",
        type=Path,
        default=Path("doc") / "NVAPI-key-table.html",
        help="Path to NVAPI key table HTML",
    )
    ap.add_argument(
        "--keywords",
        default="volt,voltage,pmumon,sample,clockclkvolt",
        help="Comma-separated name filters",
    )
    ap.add_argument("--max-targets", type=int, default=40, help="Cap number of APIs to test")
    ap.add_argument(
        "--sigs",
        default="custom,u32_ptr,struct_ptr",
        help="Comma-separated child signatures to try",
    )
    ap.add_argument("--timeout", type=float, default=1.5, help="Per-child timeout seconds")
    ap.add_argument("--cooldown", type=float, default=0.05, help="Sleep between probes")
    ap.add_argument("--json", type=Path, default=None, help="Optional JSON report path")

    # child mode
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--func-id", default="", help=argparse.SUPPRESS)
    ap.add_argument("--sig", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        if not args.func_id or not args.sig:
            print(json.dumps({"status": "bad_args"}))
            return 2
        func_id = int(args.func_id, 16)
        return _run_child(func_id=func_id, gpu=args.gpu, sig=args.sig)

    if not args.table.exists():
        raise SystemExit(f"Key table not found: {args.table}")

    ids = _parse_key_table(args.table)
    names = _target_names(
        ids=ids,
        keywords=[k.strip() for k in args.keywords.split(",")],
        max_targets=args.max_targets,
    )
    sigs = [s.strip() for s in args.sigs.split(",") if s.strip()]

    report: Dict[str, object] = {
        "gpu": args.gpu,
        "table_entries": len(ids),
        "target_count": len(names),
        "sigs": sigs,
        "timeout_s": args.timeout,
        "results": [],
    }

    script = Path(__file__).resolve()
    print(f"Targets: {len(names)} | sigs: {sigs} | timeout={args.timeout}s")

    for i, name in enumerate(names, start=1):
        fid = ids[name]
        print(f"[{i:03d}/{len(names):03d}] {name} 0x{fid:08X}")
        for sig in sigs:
            rec = _spawn_child_timeout_safe(script, fid, args.gpu, sig, args.timeout)
            rec["name"] = name
            rec["id"] = f"0x{fid:08X}"
            rec["sig"] = sig
            report["results"].append(rec)
            cls = rec.get("class")
            child = rec.get("child", {})
            status = child.get("status", "")
            print(f"   - {sig:<10} class={cls:<12} status={status}")
            time.sleep(max(0.0, args.cooldown))

    # Summary
    class_counts: Dict[str, int] = {}
    for r in report["results"]:
        cls = str(r.get("class"))
        class_counts[cls] = class_counts.get(cls, 0) + 1
    report["class_counts"] = class_counts

    plausible_voltage_hits = 0
    for r in report["results"]:
        child = r.get("child", {})
        cres = child.get("result", {})
        if isinstance(cres, dict):
            custom = cres.get("custom")
            if isinstance(custom, dict):
                # direct custom value
                cmv = custom.get("mv")
                if isinstance(cmv, int):
                    plausible_voltage_hits += 1
                    continue
                # per-version attempts
                attempts = custom.get("attempts")
                if isinstance(attempts, list):
                    hit = False
                    for a in attempts:
                        if not isinstance(a, dict):
                            continue
                        if isinstance(a.get("core_mv"), int):
                            hit = True
                            break
                        rails = a.get("rails")
                        if isinstance(rails, list):
                            for rr in rails:
                                if isinstance(rr, dict) and isinstance(rr.get("mv"), int):
                                    hit = True
                                    break
                        if hit:
                            break
                    if hit:
                        plausible_voltage_hits += 1
                        continue
            mv = cres.get("mv")
            if isinstance(mv, int):
                plausible_voltage_hits += 1
                continue
            hits = cres.get("hits")
            if isinstance(hits, list):
                for h in hits:
                    vh = h.get("voltage_hits", [])
                    if isinstance(vh, list) and vh:
                        plausible_voltage_hits += 1
                        break
    report["plausible_voltage_hit_records"] = plausible_voltage_hits

    print("\nSummary:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls:<12} {count}")
    print(f"  plausible_voltage_hit_records {plausible_voltage_hits}")

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
