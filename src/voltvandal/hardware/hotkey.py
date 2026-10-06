"""Global recovery hotkey (Windows RegisterHotKey) listener."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional


def _parse_windows_hotkey(hotkey: str) -> tuple[int, int]:
    token_map = {
        "alt": 0x0001,
        "ctrl": 0x0002,
        "control": 0x0002,
        "shift": 0x0004,
        "win": 0x0008,
        "windows": 0x0008,
    }
    key_special = {
        "esc": 0x1B,
        "escape": 0x1B,
        "tab": 0x09,
        "enter": 0x0D,
        "space": 0x20,
    }

    if not hotkey or not hotkey.strip():
        raise ValueError("Recovery hotkey cannot be empty.")
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        raise ValueError("Recovery hotkey cannot be empty.")

    modifiers = 0
    key_vk: Optional[int] = None
    for part in parts:
        if part in token_map:
            modifiers |= token_map[part]
            continue
        if part in key_special:
            key_vk = key_special[part]
            continue
        if part.startswith("f") and part[1:].isdigit():
            fn = int(part[1:])
            if 1 <= fn <= 24:
                key_vk = 0x70 + (fn - 1)
                continue
        if len(part) == 1 and ("a" <= part <= "z" or "0" <= part <= "9"):
            key_vk = ord(part.upper())
            continue
        raise ValueError(f"Unsupported hotkey token: '{part}'")

    if key_vk is None:
        raise ValueError("Recovery hotkey must include a non-modifier key (e.g. F12).")
    return modifiers, key_vk


@dataclass
class RecoveryHotkeyHandle:
    hotkey: str
    _stop_event: threading.Event
    _thread: threading.Thread

    def stop(self, timeout: float = 1.5) -> None:
        self._stop_event.set()
        self._thread.join(timeout=timeout)


def start_recovery_hotkey_listener(
    hotkey: str, trigger_event: threading.Event
) -> Optional[RecoveryHotkeyHandle]:
    """
    Start a global recovery hotkey listener on Windows.

    Returns a handle to stop the listener. Returns None on non-Windows systems.
    """
    if not sys.platform.startswith("win"):
        return None

    modifiers, key_vk = _parse_windows_hotkey(hotkey)
    stop_event = threading.Event()
    started = threading.Event()
    startup_error = {"message": ""}

    def _worker() -> None:
        import ctypes
        import ctypes.wintypes as wintypes

        user32 = ctypes.windll.user32
        WM_HOTKEY = 0x0312
        PM_REMOVE = 0x0001
        MOD_NOREPEAT = 0x4000
        HOTKEY_ID = 0xB00

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt_x", wintypes.LONG),
                ("pt_y", wintypes.LONG),
            ]

        if not user32.RegisterHotKey(None, HOTKEY_ID, modifiers | MOD_NOREPEAT, key_vk):
            startup_error["message"] = "RegisterHotKey failed."
            started.set()
            return

        started.set()
        msg = MSG()
        try:
            while not stop_event.is_set():
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                        trigger_event.set()
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                time.sleep(0.05)
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID)

    worker = threading.Thread(target=_worker, name="vv-recovery-hotkey", daemon=True)
    worker.start()
    started.wait(timeout=2.0)

    if startup_error["message"]:
        stop_event.set()
        worker.join(timeout=0.5)
        raise RuntimeError(
            f"Failed to enable recovery hotkey '{hotkey}': {startup_error['message']}"
        )

    return RecoveryHotkeyHandle(hotkey=hotkey, _stop_event=stop_event, _thread=worker)
