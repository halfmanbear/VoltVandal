"""Single-line live status rendering on stderr."""

import shutil
import sys
import time
from typing import List, Optional

from .decode import _fmt_signed_int


class LiveDisplayMixin:
    def _render_live_line(
        self,
        temp: int,
        hotspot: float,
        vram_junc: Optional[float],
        pstate: Optional[int],
        clock: int,
        mem_clock: int,
        util: int,
        voltage_mv: Optional[float],
        voltage_estimated: bool,
        power: float,
        throttle_lbl: str,
        pdec_lbl: str,
    ) -> None:
        core_parts = [f"Edge {temp}C", f"Hot {hotspot:.0f}C"]
        _elapsed = int(max(0.0, time.monotonic() - self._started_monotonic))
        if self.expected_test_seconds:
            core_parts.append(f"T {_elapsed}/{self.expected_test_seconds}s")
        else:
            core_parts.append(f"T {_elapsed}s")
        optional_parts: List[str] = []
        if vram_junc is not None: optional_parts.append(f"VRAM {vram_junc:.0f}C")
        if pstate is not None: optional_parts.append(f"P{pstate}")
        core_parts += [f"Gfx {clock}MHz", f"U {util}%"]
        optional_parts.append(f"Mem {mem_clock}MHz")
        if self.mode == "vlock" and self.vlock_target_mv > 0:
            target_mv = int(self.vlock_target_mv)
            optional_parts.append(f"Target {target_mv}mV")
            stock_mv = self._estimate_stock_voltage_mv(clock)
            if stock_mv is not None:
                vdelta_mv = target_mv - stock_mv
                optional_parts.append(f"Vdelta {_fmt_signed_int(vdelta_mv)}mV")
            stock_freq_mhz = self._estimate_stock_freq_mhz(target_mv)
            if stock_freq_mhz is not None:
                fdelta_mhz = clock - stock_freq_mhz
                optional_parts.append(f"Fdelta {_fmt_signed_int(fdelta_mhz)}MHz")
        if voltage_mv is not None:
            core_parts.append(f"V~ {voltage_mv}mV" if voltage_estimated else f"V {voltage_mv}mV")
        else:
            core_parts.append("V n/a")
        core_parts.append(f"PwrNVML {power:.0f}W")
        if throttle_lbl and throttle_lbl != "Idle":
            optional_parts.append(f"Thr:{throttle_lbl}")
            self._sticky_warn_text = f"WARN:{throttle_lbl}"
            self._sticky_warn_until = time.monotonic() + 6.0
        if pdec_lbl and pdec_lbl != "None":
            optional_parts.append(f"Perf:{pdec_lbl}")
        if self.driver_reset_detected: optional_parts.append("DRIVER_RESET")
        if time.monotonic() < self._sticky_warn_until and self._sticky_warn_text:
            optional_parts.append(self._sticky_warn_text)

        parts = core_parts + optional_parts
        line = "  " + " | ".join(parts)
        _term_width = shutil.get_terminal_size(fallback=(120, 20)).columns
        _max_width = max(40, _term_width - 1)
        while len(line) > _max_width and len(parts) > len(core_parts):
            parts.pop()
            line = "  " + " | ".join(parts)
        if len(line) > _max_width: line = line[:_max_width]
        self._live_line_len = min(max(self._live_line_len, len(line)), _max_width)
        sys.stderr.write(f"\r{line:<{self._live_line_len}}")
        sys.stderr.flush()
