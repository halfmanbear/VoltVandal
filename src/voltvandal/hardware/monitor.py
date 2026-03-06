import csv
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

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
_THROTTLE_ABORT_CONSECUTIVE_POLLS: int = 3

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
_THROTTLE_PWRCAP_BIT = 0x0000000000000004
_THROTTLE_SEVERE_BITS = (
    0x0000000000000008  # HwSlowdn
    | 0x0000000000000020  # SwTherm
    | 0x0000000000000040  # HwTherm
    | 0x0000000000000080  # PwrBrake
)
_PERF_DECREASE_LABELS = {
    0x00000001: "InsufficientPower",
    0x00000004: "AcPower",
    0x00000010: "PowerBrake",
    0x00000040: "Thermal",
}

def _decode_throttle(reasons: int) -> str:
    if reasons == 0:
        return ""
    active = [lbl for bit, lbl in _THROTTLE_LABELS.items() if reasons & bit]
    return "+".join(active) if active else f"0x{reasons:X}"

def _has_actionable_throttle(reasons: int) -> bool:
    actionable = reasons & ~_THROTTLE_IDLE_BIT
    if actionable == 0:
        return False
    # Ignore pure power-cap throttling for abort logic; this is common and
    # not by itself a stability failure.
    if actionable == _THROTTLE_PWRCAP_BIT:
        return False
    return True

def _decode_perf_decrease(info: Optional[int]) -> str:
    if info is None:
        return ""
    if info == 0:
        return "None"
    active = [lbl for bit, lbl in _PERF_DECREASE_LABELS.items() if info & bit]
    return "+".join(active) if active else f"0x{info:X}"

def _next_throttle_streak(prev_streak: int, reasons: int) -> int:
    return prev_streak + 1 if _has_actionable_throttle(reasons) else 0

def _fmt_signed_int(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)

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
        stock_curve_csv: Optional[Path] = None,
        mode: str = "",
        vlock_target_mv: int = 0,
        expected_test_seconds: Optional[int] = None,
        live_display: bool = True,
        use_nvapi_live: bool = False,
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
        self.stock_curve_csv = stock_curve_csv
        self.mode = mode
        self.vlock_target_mv = vlock_target_mv
        self.expected_test_seconds = expected_test_seconds
        self.live_display = live_display
        self.use_nvapi_live = use_nvapi_live
        self._live_line_len: int = 0

        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.max_temp: Optional[int] = None
        self.max_hotspot: Optional[float] = None
        self.max_power: Optional[float] = None
        self.any_throttle: bool = False
        self.last_snapshot: Optional[MonitorSnapshot] = None
        self.abort_reason: str = ""
        self._consecutive_errors: int = 0
        self._curve_points: Optional[List[CurvePoint]] = None
        self._stock_curve_points: Optional[List[CurvePoint]] = None
        self._sticky_warn_until: float = 0.0
        self._sticky_warn_text: str = ""

        self.driver_reset_detected: bool = False
        self._had_boost_clock: bool = False
        self._tdr_collapse_polls: int = 0
        self._actionable_throttle_streak: int = 0
        self._started_monotonic: float = time.monotonic()
        self._sample_count: int = 0
        self._clock_samples: List[int] = []
        self._clock_sum_mhz: float = 0.0
        self._throttle_any_count: int = 0
        self._throttle_pwr_count: int = 0
        self._throttle_severe_count: int = 0
        self._throttle_label_counts: Dict[str, int] = {}

    def _estimate_voltage_mv_from_curve(self, clock_mhz: int) -> Optional[int]:
        return self._estimate_voltage_mv_from_points(clock_mhz, stock=False)

    def _estimate_stock_voltage_mv(self, clock_mhz: int) -> Optional[int]:
        return self._estimate_voltage_mv_from_points(clock_mhz, stock=True)

    def _estimate_stock_freq_mhz(self, voltage_mv: int) -> Optional[int]:
        if voltage_mv <= 0:
            return None
        points = self._stock_curve_points
        if points is None:
            if self.stock_curve_csv is None or not self.stock_curve_csv.exists():
                return None
            try:
                points = load_curve_csv(self.stock_curve_csv)
            except Exception:
                points = []
            self._stock_curve_points = points

        if not points:
            return None

        target_uv = voltage_mv * 1000
        best = min(points, key=lambda p: abs(p.voltage_uv - target_uv))
        return int(round(best.freq_khz / 1000.0))

    def _estimate_voltage_mv_from_points(self, clock_mhz: int, stock: bool) -> Optional[int]:
        if clock_mhz <= 0:
            return None

        points_cache_name = "_stock_curve_points" if stock else "_curve_points"
        points = getattr(self, points_cache_name)
        csv_path = self.stock_curve_csv if stock else self.curve_csv
        if points is None:
            if csv_path is None or not csv_path.exists():
                return None
            try:
                points = load_curve_csv(csv_path)
            except Exception:
                points = []
            setattr(self, points_cache_name, points)

        if not points:
            return None

        target_khz = clock_mhz * 1000
        best = min(points, key=lambda p: abs(p.freq_khz - target_khz))
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

                if self.use_nvapi_live and _nvapi_native is not None:
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
                self._sample_count += 1
                self._clock_samples.append(clock)
                self._clock_sum_mhz += float(clock)
                throttle_no_idle = throttle & ~_THROTTLE_IDLE_BIT
                if throttle_no_idle:
                    self._throttle_any_count += 1
                if throttle & _THROTTLE_PWRCAP_BIT:
                    self._throttle_pwr_count += 1
                if throttle & _THROTTLE_SEVERE_BITS:
                    self._throttle_severe_count += 1
                for bit, lbl in _THROTTLE_LABELS.items():
                    if bit == _THROTTLE_IDLE_BIT:
                        continue
                    if throttle & bit:
                        self._throttle_label_counts[lbl] = self._throttle_label_counts.get(lbl, 0) + 1

                self.max_temp = temp if self.max_temp is None else max(self.max_temp, temp)
                self.max_hotspot = hotspot if self.max_hotspot is None else max(self.max_hotspot, hotspot)
                self.max_power = power if self.max_power is None else max(self.max_power, power)
                self._actionable_throttle_streak = _next_throttle_streak(
                    self._actionable_throttle_streak, throttle
                )
                if self._actionable_throttle_streak > 0:
                    self.any_throttle = True

                vram_str = f"{vram_junc:.1f}" if vram_junc is not None else ""
                volt_str = str(voltage_mv) if voltage_mv is not None else ""
                pstate_str = str(pstate) if pstate is not None else ""
                throttle_lbl = _decode_throttle(throttle)
                throttle_str = str(throttle) if not throttle_lbl else f"{throttle} ({throttle_lbl})"
                pdec_lbl = _decode_perf_decrease(perf_decrease)
                pdec_raw = f"0x{perf_decrease:X}" if perf_decrease is not None else ""
                pdec_str = pdec_raw if not pdec_lbl else f"{pdec_raw} ({pdec_lbl})"
                gpu_mw_str = str(topo_gpu_mw) if topo_gpu_mw is not None else ""
                tot_mw_str = str(topo_total_mw) if topo_total_mw is not None else ""
                with self.log_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [now_utc_iso(), temp, f"{hotspot:.1f}", vram_str,
                         f"{power:.1f}", clock, mem_clock, util, volt_str, throttle_str,
                         pstate_str, pdec_str, gpu_mw_str, tot_mw_str]
                    )

                if self.live_display:
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
                        core_parts.append(f"V {target_mv}mV")
                        stock_mv = self._estimate_stock_voltage_mv(clock)
                        if stock_mv is not None:
                            vdelta_mv = target_mv - stock_mv
                            optional_parts.append(f"Vdelta {_fmt_signed_int(vdelta_mv)}mV")
                        stock_freq_mhz = self._estimate_stock_freq_mhz(target_mv)
                        if stock_freq_mhz is not None:
                            fdelta_mhz = clock - stock_freq_mhz
                            optional_parts.append(f"Fdelta {_fmt_signed_int(fdelta_mhz)}MHz")
                    elif voltage_mv is not None:
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

                if temp >= self.temp_limit_c:
                    if not self.abort_event.is_set():
                        self.abort_reason = f"EDGE_TEMP_{temp}C_GE_{self.temp_limit_c}C"
                    self.abort_event.set()
                elif self.hotspot_limit_c and hotspot >= self.hotspot_limit_c:
                    if not self.abort_event.is_set():
                        self.abort_reason = f"HOTSPOT_{hotspot:.1f}C_GE_{self.hotspot_limit_c}C"
                    self.abort_event.set()
                elif power >= self.power_limit_w:
                    if not self.abort_event.is_set():
                        self.abort_reason = f"POWER_{power:.1f}W_GE_{self.power_limit_w:.1f}W"
                    self.abort_event.set()
                elif self.abort_on_throttle and self._actionable_throttle_streak >= _THROTTLE_ABORT_CONSECUTIVE_POLLS:
                    if not self.abort_event.is_set():
                        self.abort_reason = (
                            f"THROTTLE_{throttle_lbl or throttle}"
                            f"_STREAK_{self._actionable_throttle_streak}"
                        )
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

    def _clock_p95_mhz(self) -> Optional[float]:
        if not self._clock_samples:
            return None
        values = sorted(self._clock_samples)
        if len(values) == 1:
            return float(values[0])
        idx = int(round(0.95 * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return float(values[idx])

    def metrics(self) -> Dict[str, float]:
        samples = float(self._sample_count)
        if samples <= 0.0:
            return {
                "sample_count": 0.0,
                "avg_clock_mhz": 0.0,
                "p95_clock_mhz": 0.0,
                "max_clock_mhz": 0.0,
                "throttle_any_ratio_pct": 0.0,
                "throttle_pwr_ratio_pct": 0.0,
                "throttle_severe_ratio_pct": 0.0,
                "throttle_any_count": 0.0,
                "throttle_pwr_count": 0.0,
                "throttle_severe_count": 0.0,
            }
        p95 = self._clock_p95_mhz() or 0.0
        max_clock = float(max(self._clock_samples)) if self._clock_samples else 0.0
        return {
            "sample_count": samples,
            "avg_clock_mhz": self._clock_sum_mhz / samples,
            "p95_clock_mhz": p95,
            "max_clock_mhz": max_clock,
            "throttle_any_ratio_pct": (self._throttle_any_count / samples) * 100.0,
            "throttle_pwr_ratio_pct": (self._throttle_pwr_count / samples) * 100.0,
            "throttle_severe_ratio_pct": (self._throttle_severe_count / samples) * 100.0,
            "throttle_any_count": float(self._throttle_any_count),
            "throttle_pwr_count": float(self._throttle_pwr_count),
            "throttle_severe_count": float(self._throttle_severe_count),
        }

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5.0)
        if self.live_display and self._live_line_len > 0:
            sys.stderr.write("\r" + " " * (self._live_line_len + 2) + "\r")
            sys.stderr.flush()
        try:
            if pynvml: pynvml.nvmlShutdown()
        except Exception: pass
