"""Parent-side child spawning with crash/timeout classification and result summary."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


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


def summarize(results: List[Dict[str, object]]) -> Tuple[Dict[str, int], int]:
    """Return (record count per class, records holding a plausible voltage reading)."""
    class_counts: Dict[str, int] = {}
    for r in results:
        cls = str(r.get("class"))
        class_counts[cls] = class_counts.get(cls, 0) + 1
    plausible_voltage_hits = 0
    for r in results:
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
    return class_counts, plausible_voltage_hits
