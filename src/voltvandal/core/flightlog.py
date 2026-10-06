"""Crash-proof flight recorder: every entry is fsynced so it survives a bugcheck."""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_path: Optional[Path] = None


def start(out_dir) -> Optional[str]:
    """Open <out_dir>/flight.jsonl; return the last event of a previous run that never ended."""
    global _path
    _path = Path(out_dir) / "flight.jsonl"
    _path.parent.mkdir(parents=True, exist_ok=True)
    unfinished = _unfinished_tail(_path)
    log("session_start", pid=os.getpid(), previous_unfinished=unfinished)
    return unfinished


def _unfinished_tail(path: Path) -> Optional[str]:
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        last = json.loads(lines[-1])
    except (OSError, ValueError, IndexError):
        return None
    return None if last.get("ev") == "session_end" else lines[-1][:500]


def log(ev: str, **fields) -> None:
    if _path is None:
        return
    now = datetime.now(timezone.utc)
    entry = {"t_utc": now.isoformat(timespec="milliseconds"), "t_local": now.astimezone().isoformat(timespec="milliseconds"), "ev": ev}
    entry.update(fields)
    try:
        with _lock, _path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except OSError:
        pass
