import os
import sys
from datetime import datetime
from pathlib import Path

def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def is_windows() -> bool:
    return os.name == "nt"

def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)

def warn_if_not_admin() -> None:
    """On Windows, emit a warning when the process is not elevated."""
    if not is_windows():
        return
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            eprint(
                "WARNING: Not running as Administrator. "
                "NVAPI VF-curve operations require elevated privileges. "
                "Re-run from an elevated terminal if commands fail."
            )
    except Exception:
        pass
