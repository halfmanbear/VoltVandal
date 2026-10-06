"""Parse the NVAPI key table and select read-oriented targets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


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
