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
import json
import time
from pathlib import Path
from typing import Dict

from nvapi_fuzz.harness import _spawn_child_timeout_safe, summarize
from nvapi_fuzz.keytable import _parse_key_table, _target_names
from nvapi_fuzz.probes import _run_child


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

    class_counts, plausible_voltage_hits = summarize(report["results"])
    report["class_counts"] = class_counts
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
