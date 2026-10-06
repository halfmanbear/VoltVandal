"""Persistent record of hard-faulting operating points, written ahead of each test.

A bugcheck kills the process before any result is recorded, so the point under
test is journalled as in-flight (fsynced) before the curve is applied. If it is
still in-flight at the next start, the machine crashed on it and it is blocked.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

_FILE = "fault_map.json"


def _load(out_dir) -> dict:
    try:
        data = json.loads((Path(out_dir) / _FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"failed": {str(k): int(v) for k, v in data.get("failed", {}).items()},
            "in_flight": data.get("in_flight")}


def _save(out_dir, data: dict) -> None:
    path = Path(out_dir) / _FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mark_in_flight(out_dir, label: str, voltage_uv: int, freq_khz: int) -> None:
    data = _load(out_dir)
    data["in_flight"] = {"label": label, "voltage_uv": voltage_uv, "freq_khz": freq_khz}
    _save(out_dir, data)


def clear_in_flight(out_dir) -> None:
    data = _load(out_dir)
    if data["in_flight"] is not None:
        data["in_flight"] = None
        _save(out_dir, data)


def record_fault(out_dir, voltage_uv: int, freq_khz: int) -> None:
    data = _load(out_dir)
    key = str(voltage_uv)
    data["failed"][key] = min(freq_khz, data["failed"].get(key, freq_khz))
    _save(out_dir, data)


def recover(out_dir) -> Optional[dict]:
    """Block a point left in-flight by a crashed run; return it (or None)."""
    stale = _load(out_dir)["in_flight"]
    if stale:
        record_fault(out_dir, stale["voltage_uv"], stale["freq_khz"])
        clear_in_flight(out_dir)
    return stale


def failed_floors(out_dir) -> Dict[int, int]:
    return {int(k): v for k, v in _load(out_dir)["failed"].items()}


def is_blocked(out_dir, voltage_uv: int, freq_khz: int) -> bool:
    """True if a point at this voltage or higher already faulted at or below this clock."""
    return any(voltage_uv <= v and freq_khz >= f for v, f in failed_floors(out_dir).items())
