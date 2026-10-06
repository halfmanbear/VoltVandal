import threading
from pathlib import Path

from ..models import SessionState
from ..utils import eprint
from ...hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe


def revert_to_last_good(state: SessionState) -> None:
    nvapi_apply_curve_safe(
        state.gpu,
        Path(state.last_good_curve_csv),
        timeout_seconds=12.0,
    )

def _check_for_manual_recovery(state: SessionState, label: str, manual_recovery_event: threading.Event) -> None:
    if not manual_recovery_event.is_set(): return
    manual_recovery_event.clear()
    eprint(f"\nManual recovery requested ({label}) — reverting...")
    try:
        revert_to_last_good(state)
        eprint("Revert complete.")
    except Exception as ex:
        eprint(f"WARNING: revert failed: {ex}")
    raise KeyboardInterrupt("Manual recovery hotkey")
