import csv
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

try:
    import pynvml
except ImportError:
    pynvml = None

from ..core.models import MonitorSnapshot, CurvePoint
from ..core.utils import eprint, now_utc_iso
from ..core.curve import load_curve_csv

# Import native nvapi if available
try:
    from . import nvapi as _nvapi_native
except ImportError:
    _nvapi_native = None

_TDR_MIN_BOOST_MHZ:  int = 1200
_TDR_BASE_CLOCK_MHZ: int = 600
_TDR_ARM_MIN_UTIL_PCT: int = 35
_TDR_MIN_SECONDS_BEFORE_DETECT: float = 8.0
_TDR_COLLAPSE_POLLS: int = 3

_THROTTLE_LABELS = {
    0x0000000000000001: "Idle",
    0x0000000000000002: "AppClk",
    0x0000000000000004: "PwrCap",
    0x0000000000000008: "HwSlowdn",
    0x0000000000000010: "SyncBst",
    0x0000000000000020: "SwTherm",
    0x0000000000000040: "HwTherm",
    0x0000000000000080: "PwrBrake",
    0x0000000000000100: "DispClk",
}

_THROTTLE_IDLE_BIT = 0x0000000000000001

def _decode_throttle(reasons: int) -> str:
    if reasons == 0:
        return ""
    active = [lbl for bit, lbl in _THROTTLE_LABELS.items() if reasons & bit]
    return "+".join(active) if active else f"0x{reasons:X}"

def _has_actionable_throttle(reasons: int) -> bool:
    return (reasons & ~_THROTTLE_IDLE_BIT) != 0

class NvmlMonitor:
    def __init__(
        self,
        gpu_index: int,
        poll_seconds: float,
        temp_limit_c: int,
        hotspot_limit_c: Optional[int],
        hotspot_offset_c: int,
        power_limit_w: float,
        abort_on_throttle: bool,
        log_csv: Path,
        curve_csv: Optional[Path] = None,
        expected_test_seconds: Optional[int] = None,
        live_display: bool = True,
    ):
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.temp_limit_c = temp_limit_c
        self.hotspot_limit_c = hotspot_limit_c
        self.hotspot_offset_c = hotspot_offset_c
        self.power_limit_w = power_limit_w
        self.abort_on_throttle = abort_on_throttle
        self.log_csv = log_csv
        self.curve_csv = curve_csv
        self.expected_test_seconds = expected_test_seconds
        self.live_display = live_display
        self._live_line_len: int = 0

        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.max_temp: Optional[int] = None
        self.max_hotspot: Optional[float] = None
        self.max_power: Optional[float] = None
        self.any_throttle: bool = False
        self.last_snapshot: Optional[MonitorSnapshot] = None
        self._consecutive_errors: int = 0
        self._curve_points: Optional[List[CurvePoint]] = None
        self._sticky_warn_until: float = 0.0
        self._sticky_warn_text: str = ""

        self.driver_reset_detected: bool = False
        self._had_boost_clock: bool = False
        self._tdr_collapse_polls: int = 0
        self._started_monotonic: float = time.monotonic()

    def _estimate_voltage_mv_from_curve(self, clock_mhz: int) -> Optional[int]:
        if clock_mhz <= 0:
            return None
        if self._curve_points is None:
            if self.curve_csv is None or not self.curve_csv.exists():
                return None
            try:
                self._curve_points = load_curve_csv(self.curve_csv)
            except Exception:
                self._curve_points = []
        if not self._curve_points:
            return None

        target_khz = clock_mhz * 1000
        best = min(self._curve_points, key=lambda p: abs(p.freq_khz - target_khz))
        if abs(best.freq_khz - target_khz) > 250_000:
            return None
        return int(round(best.voltage_uv / 1000.0))

    def start(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml not installed. `pip install nvidia-ml-py`")
        pynvml.nvmlInit()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            pynvml.nvmlShutdown()
            raise

        first = not self.log_csv.exists()
        with self.log_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(
                    [
                        "utc", "temp_c", "hotspot_c", "vram_junction_c",
                        "power_w", "clock_mhz", "mem_clock_mhz", "util_gpu",
                        "voltage_mv", "throttle_reasons", "pstate",
                        "perf_decrease", "topo_gpu_mw", "topo_total_mw",
                    ]
                )

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                temp = int(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
                power = float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
                clock = int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS))
                mem_clock = int(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM))
                util = int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                throttle = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle))

                hotspot: float = temp + self.hotspot_offset_c
                vram_junc: Optional[float] = None
                voltage_mv: Optional[int] = None
                voltage_estimated = False
                pstate: Optional[int] = None
                perf_decrease: Optional[int] = None
                topo_gpu_mw: Optional[int] = None
                topo_total_mw: Optional[int] = None

                if _nvapi_native is not None:
                    try:
                        thr = _nvapi_native.get_thermal_sensors(self.gpu_index)
                        if thr["hotspot_c"] is not None:
                            hotspot = thr["hotspot_c"]
                        vram_junc = thr.get("vram_junction_c")
                    except Exception:
                        pass
                    try:
                        voltage_mv = _nvapi_native.get_current_voltage_mv(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        pstate = _nvapi_native.get_current_pstate(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        perf_decrease = _nvapi_native.get_perf_decrease_info(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        topo = _nvapi_native.get_power_topology_mw(self.gpu_index)
                        if topo:
                            topo_gpu_mw   = topo.get("gpu_mw")
                            topo_total_mw = topo.get("total_mw")
                    except Exception:
                        pass

                if voltage_mv is not None:
                    if voltage_mv > 20000:
                        voltage_mv = int(round(voltage_mv / 1000.0))
                    if voltage_mv < 400 or voltage_mv > 2000:
                        voltage_mv = None
                if voltage_mv is None:
                    _est = self._estimate_voltage_mv_from_curve(clock)
                    if _est is not None and 400 <= _est <= 2000:
                        voltage_mv = _est
                        voltage_estimated = True

                snap = MonitorSnapshot(
                    temp, hotspot, vram_junc, power, clock, mem_clock, util, throttle,
                    voltage_mv, voltage_estimated, pstate, perf_decrease, topo_gpu_mw, topo_total_mw,
                )
                self.last_snapshot = snap

                self.max_temp = temp if self.max_temp is None else max(self.max_temp, temp)
                self.max_hotspot = hotspot if self.max_hotspot is None else max(self.max_hotspot, hotspot)
                self.max_power = power if self.max_power is None else max(self.max_power, power)
                if _has_actionable_throttle(throttle):
                    self.any_throttle = True

                vram_str = f"{vram_junc:.1f}" if vram_junc is not None else ""
                volt_str = str(voltage_mv) if voltage_mv is not None else ""
                pstate_str = str(pstate) if pstate is not None else ""
                pdec_str = f"0x{perf_decrease:X}" if perf_decrease is not None else ""
                gpu_mw_str = str(topo_gpu_mw) if topo_gpu_mw is not None else ""
                tot_mw_str = str(topo_total_mw) if topo_total_mw is not None else ""
                with self.log_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [now_utc_iso(), temp, f"{hotspot:.1f}", vram_str,
                         f"{power:.1f}", clock, mem_clock, util, volt_str, throttle,
                         pstate_str, pdec_str, gpu_mw_str, tot_mw_str]
                    )

                if self.live_display:
                    display_power_w = power
                    if topo_total_mw is not None:
                        topo_total_w = topo_total_mw / 1000.0
                        if topo_total_w > 20.0 and (power <= 0.0 or (0.5 * power <= topo_total_w <= 2.0 * power)):
                            display_power_w = topo_total_w

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
                    if voltage_mv is not None:
                        core_parts.append(f"V~ {voltage_mv}mV" if voltage_estimated else f"V {voltage_mv}mV")
                    else:
                        core_parts.append("V n/a")
                    core_parts.append(f"Pwr {display_power_w:.0f}W")
                    throttle_lbl = _decode_throttle(throttle)
                    if throttle_lbl and throttle_lbl != "Idle":
                        optional_parts.append(f"Thr:{throttle_lbl}")
                        self._sticky_warn_text = f"WARN:{throttle_lbl}"
                        self._sticky_warn_until = time.monotonic() + 6.0
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

                if temp >= self.temp_limit_c or (self.hotspot_limit_c and hotspot >= self.hotspot_limit_c) or power >= self.power_limit_w or (self.abort_on_throttle and _has_actionable_throttle(throttle)):
                    self.abort_event.set()

                if clock >= _TDR_MIN_BOOST_MHZ and util >= _TDR_ARM_MIN_UTIL_PCT:
                    self._had_boost_clock = True
                _elapsed_s = time.monotonic() - self._started_monotonic
                if self._had_boost_clock and _elapsed_s >= _TDR_MIN_SECONDS_BEFORE_DETECT and clock <= _TDR_BASE_CLOCK_MHZ and util <= 20 and (pstate is None or pstate >= 5):
                    self._tdr_collapse_polls += 1
                else:
                    self._tdr_collapse_polls = 0
                if self._tdr_collapse_polls >= _TDR_COLLAPSE_POLLS and not self.driver_reset_detected:
                    self.driver_reset_detected = True
                    eprint("\n  !! GPU DRIVER RESET (TDR) detected. Aborting.")
                    self.abort_event.set()

            except Exception as e:
                self._consecutive_errors += 1
                if self._consecutive_errors >= 3: self.abort_event.set()
                self.stop_event.wait(timeout=self.poll_seconds)
                continue
            self._consecutive_errors = 0
            self.stop_event.wait(timeout=self.poll_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5.0)
        if self.live_display and self._live_line_len > 0:
            sys.stderr.write("\r" + " " * (self._live_line_len + 2) + "\r")
            sys.stderr.flush()
        try:
            if pynvml: pynvml.nvmlShutdown()
        except Exception: pass
