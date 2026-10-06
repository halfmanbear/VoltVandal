"""Windows event-log checks for silent GPU instability (WHEA, driver faults, TDR)."""

import json
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

from .runtime_controls import _run_command

_WHEA = "Microsoft-Windows-WHEA-Logger"
_NVIDIA = "nvlddmkm"
_DISPLAY = "Display"
_TDR_ID = 4101

_SCRIPT = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$s=[DateTime]::Parse('{start}');"
    "Get-WinEvent -FilterHashtable @{{LogName='System';StartTime=$s}} | "
    "Where-Object {{ $_.ProviderName -in '{whea}','{nv}','{disp}' }} | "
    "Select-Object Id,ProviderName | ConvertTo-Json -Compress"
)


def is_hardware_error(provider: str, event_id: int) -> bool:
    if provider in (_WHEA, _NVIDIA):
        return True
    return provider == _DISPLAY and event_id == _TDR_ID


def parse_events(raw: str) -> List[str]:
    """Hardware-error descriptions from Get-WinEvent JSON (one object or a list)."""
    raw = (raw or "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    return [f"{e['ProviderName']}#{e['Id']}" for e in data
            if is_hardware_error(str(e.get("ProviderName")), int(e.get("Id", 0)))]


def hardware_errors_since(start_local: datetime) -> Optional[List[str]]:
    """Events logged since start_local, or None when the log cannot be read."""
    if not sys.platform.startswith("win"):
        return None
    script = _SCRIPT.format(start=start_local.strftime("%Y-%m-%dT%H:%M:%S"),
                            whea=_WHEA, nv=_NVIDIA, disp=_DISPLAY)
    result = _run_command(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                          timeout_seconds=20.0)
    if result.returncode != 0:
        return None
    try:
        return parse_events(result.stdout)
    except (ValueError, KeyError, TypeError):
        return None


_PROVIDER_FILTER = (
    f"(Provider[@Name='{_NVIDIA}'] or Provider[@Name='{_WHEA}'] or "
    f"(Provider[@Name='{_DISPLAY}'] and EventID={_TDR_ID}))"
)
_EVENT_RE = re.compile(r"<Provider Name='([^']+)'.*?<EventID[^>]*>(\d+)</EventID>", re.S)


def parse_wevtutil_xml(xml: str) -> List[str]:
    return [f"{p}#{i}" for p, i in _EVENT_RE.findall(xml or "") if is_hardware_error(p, int(i))]


def fast_hardware_errors_since(start_utc: datetime) -> Optional[List[str]]:
    """Quick (~0.1 s) event-log poll for use while a test is running; None if unreadable."""
    if not sys.platform.startswith("win"):
        return None
    stamp = start_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    query = f"*[System[{_PROVIDER_FILTER} and TimeCreated[@SystemTime>='{stamp}']]]"
    result = _run_command(["wevtutil", "qe", "System", f"/q:{query}", "/f:xml", "/c:20"],
                          timeout_seconds=5.0)
    return parse_wevtutil_xml(result.stdout) if result.returncode == 0 else None


class FaultWatch:
    """Background poller that fires on_fault once when a GPU fault event appears."""

    def __init__(self, on_fault: Callable[[List[str]], None],
                 query: Callable[[datetime], Optional[List[str]]] = fast_hardware_errors_since,
                 interval_s: float = 0.5):
        self._on_fault, self._query, self._interval = on_fault, query, interval_s
        self._stop = threading.Event()
        self.tripped = threading.Event()
        self.faults: List[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_utc = datetime.now(timezone.utc) - timedelta(seconds=1)

    def start(self) -> "FaultWatch":
        self._start_utc = datetime.now(timezone.utc) - timedelta(seconds=1)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            found = self._query(self._start_utc)
            if found:
                self.faults = sorted(set(found))
                self.tripped.set()
                try:
                    self._on_fault(self.faults)
                finally:
                    return
            self._stop.wait(self._interval)
