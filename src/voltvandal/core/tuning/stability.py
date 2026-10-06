import re
from typing import Optional, Tuple


def _doloming_mode_tag(mode: str) -> str:
    return mode.upper().replace("-", "_")

def _extract_summary_value(text: str, label: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else None

def _extract_first_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None

def _parse_doloming_stability(out_text: str, mode: str) -> Tuple[bool, Optional[str]]:
    mode_tag = _doloming_mode_tag(mode)
    summary_idx = out_text.rfind("Test Summary:")
    scan_text = out_text[summary_idx:] if summary_idx >= 0 else out_text
    if re.search(r"cuda_?error\w*|illegal memory access|device-side assert|unspecified launch failure|launch timeout|driver shutting down", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_CUDA_RUNTIME_ERROR"
    if re.search(r"Error during (?:stress )?test:", scan_text, re.I):
        return False, f"DOLOMING_{mode_tag}_STRESS_ERROR"

    status = _extract_summary_value(scan_text, "Status")
    if status and re.search(r"\b(unstable|fail(?:ed|ure)?|error)\b", status, re.I):
        return False, f"DOLOMING_{mode_tag}_UNSTABLE_STATUS"
    if re.search(r"failed to fully stabilize", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_FAILED_TO_STABILIZE"
    return True, None
