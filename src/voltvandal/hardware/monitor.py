import csv
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pynvml
except ImportError:
    pynvml = None

from ..core.models import MonitorSnapshot, CurvePoint
from ..core.utils import eprint, now_utc_iso
from ..core.curve import load_curve_csv
from .point_lock import read_voltage_mv

# Import native nvapi if available
try:
    from . import nvapi as _nvapi_native
except ImportError:
    _nvapi_native = None

_THROTTLE_ABORT_CONSECUTIVE_POLLS: int = 3
# A hung GPU keeps its clock but its power falls far below what the run has
# already shown it draws (seen before the 0x116 TDR crash). Abort early.
_COLLAPSE_MIN_PEAK_W: float = 150.0
_COLLAPSE_POWER_RATIO: float = 0.45
_COLLAPSE_ABORT_CONSECUTIVE_POLLS: int = 5

_TELEMETRY_COLUMNS = [
    "utc", "temp_c", "hotspot_c", "vram_junction_c", "power_w",
    "clock_mhz", "mem_clock_mhz", "util_gpu", "voltage_mv",
    "throttle_reasons", "pstate", "perf_decrease", "topo_gpu_mw",
    "topo_total_mw", "measured_voltage_mv", "voltage_source",
]

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

def _next_collapse_streak(prev_streak: int, power_w: float, peak_power_w: float) -> int:
    collapsed = (
        peak_power_w >= _COLLAPSE_MIN_PEAK_W
        and power_w < peak_power_w * _COLLAPSE_POWER_RATIO
    )
    return prev_streak + 1 if collapsed else 0

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
        measure_voltage: bool = False,
        point_lock_bus: Optional[int] = None,
        point_voltage_mv: float = 0.0,
        point_clock_mhz: float = 0.0,
        point_voltage_tolerance_mv: float = 3.0,
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
        self.measure_voltage = measure_voltage
        self.point_lock_bus = point_lock_bus
        self._voltage_bus: Optional[int] = point_lock_bus
        self._voltage_read_errors = 0
        self._voltage_read_disabled = False
        self.point_voltage_mv = point_voltage_mv
        self.point_clock_mhz = point_clock_mhz
        self.point_voltage_tolerance_mv = point_voltage_tolerance_mv
        self._point_counts = (0, 0, 0)  # loaded samples, valid voltage, on-target
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
        self._actionable_throttle_streak: int = 0
        self._collapse_streak: int = 0
        self._collapse_peak_w: float = 0.0
        self._collapse_armed: bool = False
        self._started_monotonic: float = time.monotonic()
        self._sample_count: int = 0
        self._loaded_sample_count: int = 0
        self._measured_loaded_voltages_mv: List[float] = []
        self._voltage_streak_mv: List[Tuple[float, float]] = []
        self._sustained_min_voltage_mv: Optional[float] = None
        self._clock_samples: List[int] = []
        self._clock_sum_mhz: float = 0.0
        self._throttle_any_count: int = 0
        self._throttle_pwr_count: int = 0
        self._throttle_severe_count: int = 0
        self._throttle_label_counts: Dict[str, int] = {}

    def arm_collapse_check(self) -> None:
        """Start hang detection for one stress process, forgetting earlier runs' peak."""
        self._collapse_peak_w = 0.0
        self._collapse_streak = 0
        self._collapse_armed = True

    def disarm_collapse_check(self) -> None:
        """Stop hang detection; stress spin-up/spin-down power dips are expected."""
        self._collapse_armed = False
        self._collapse_streak = 0

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
            if self.measure_voltage and self.point_lock_bus is None:
                try:
                    pci = pynvml.nvmlDeviceGetPciInfo(self.handle)
                    if pci.domain == 0:
                        self._voltage_bus = int(pci.bus)
                except Exception:
                    self._voltage_bus = None
        except Exception:
            pynvml.nvmlShutdown()
            raise

        self._ensure_telemetry_header()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _ensure_telemetry_header(self) -> None:
        if not self.log_csv.exists() or self.log_csv.stat().st_size == 0:
            with self.log_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_TELEMETRY_COLUMNS)
            return
        with self.log_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header == _TELEMETRY_COLUMNS:
                return
            if header != _TELEMETRY_COLUMNS[:-2]:
                raise ValueError(f"Unrecognised telemetry columns in {self.log_csv}")
            temp = self.log_csv.with_suffix(".tmp")
            with temp.open("w", newline="", encoding="utf-8") as out:
                writer = csv.writer(out)
                writer.writerow(_TELEMETRY_COLUMNS)
                for row in reader:
                    writer.writerow(row + ["", "legacy_unknown"])
        os.replace(temp, self.log_csv)

    def _sample_voltage(self, clock_mhz: int, utilization: int) -> Tuple[Optional[float], Optional[float], str]:
        measured_mv = None
        bus = self._voltage_bus
        if bus is not None and not self._voltage_read_disabled:
            try:
                value = float(read_voltage_mv(self.gpu_index, bus))
                if 400 <= value <= 1500:
                    measured_mv = value
                    self._voltage_read_errors = 0
                else:
                    self._voltage_read_errors += 1
            except Exception:
                self._voltage_read_errors += 1
            if self._voltage_read_errors >= 3:
                if self.point_lock_bus is not None:
                    self.abort_reason = "POINT_VOLTAGE_UNAVAILABLE"
                    self.abort_event.set()
                else:
                    self._voltage_read_disabled = True
        # Coverage also matters for ordinary vlock lower-bin tests: a stress
        # pass at the anchor cannot validate a lower point the GPU never used.
        if self.point_lock_bus is not None or (self.point_voltage_mv > 0 and self.point_clock_mhz > 0):
            self._record_point_sample(measured_mv, clock_mhz, utilization)
        if measured_mv is not None:
            return measured_mv, measured_mv, "nvapi_measured"
        estimated_mv = self._estimate_voltage_mv_from_curve(clock_mhz)
        if estimated_mv is not None and 400 <= estimated_mv <= 2000:
            return estimated_mv, None, "curve_estimate"
        return None, None, "unavailable"

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

                voltage_mv, measured_voltage_mv, voltage_source = self._sample_voltage(clock, util)
                voltage_estimated = voltage_source == "curve_estimate"

                snap = MonitorSnapshot(
                    temp, hotspot, vram_junc, power, clock, mem_clock, util, throttle,
                    voltage_mv, voltage_estimated, pstate, perf_decrease, topo_gpu_mw, topo_total_mw,
                    measured_voltage_mv, voltage_source,
                )
                self.last_snapshot = snap
                self._sample_count += 1
                if util >= 35:
                    self._loaded_sample_count += 1
                    if measured_voltage_mv is not None:
                        self._measured_loaded_voltages_mv.append(measured_voltage_mv)
                self._record_sustained_voltage_sample(measured_voltage_mv, util, throttle)
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
                if self._collapse_armed:
                    self._collapse_peak_w = max(self._collapse_peak_w, power)
                    self._collapse_streak = _next_collapse_streak(
                        self._collapse_streak, power, self._collapse_peak_w
                    )
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
                         pstate_str, pdec_str, gpu_mw_str, tot_mw_str,
                         measured_voltage_mv if measured_voltage_mv is not None else "",
                         voltage_source]
                    )
                    f.flush()
                    os.fsync(f.fileno())

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
                elif self._collapse_armed and self._collapse_streak >= _COLLAPSE_ABORT_CONSECUTIVE_POLLS:
                    if not self.abort_event.is_set():
                        self.abort_reason = (
                            f"GPU_HANG_POWER_COLLAPSE_{power:.0f}W_PEAK_{self._collapse_peak_w:.0f}W"
                        )
                    self.abort_event.set()
                elif self.abort_on_throttle and self._actionable_throttle_streak >= _THROTTLE_ABORT_CONSECUTIVE_POLLS:
                    if not self.abort_event.is_set():
                        self.abort_reason = (
                            f"THROTTLE_{throttle_lbl or throttle}"
                            f"_STREAK_{self._actionable_throttle_streak}"
                        )
                    self.abort_event.set()

            except Exception as e:
                self._voltage_streak_mv.clear()
                self._consecutive_errors += 1
                if self._consecutive_errors >= 3:
                    if not self.abort_event.is_set():
                        self.driver_reset_detected = isinstance(
                            e, getattr(pynvml, "NVMLError_GpuIsLost", ())
                        )
                        self.abort_reason = (
                            "GPU_LOST" if self.driver_reset_detected else "TELEMETRY_UNAVAILABLE"
                        )
                        eprint(
                            f"\n  !! GPU monitor failed after repeated reads "
                            f"({type(e).__name__}: {e}). Aborting."
                        )
                    self.abort_event.set()
                self.stop_event.wait(timeout=self.poll_seconds)
                continue
            self._consecutive_errors = 0
            self.stop_event.wait(timeout=self.poll_seconds)

    def _record_sustained_voltage_sample(
        self, measured_mv: Optional[float], util: int, throttle: int,
        sample_time_s: Optional[float] = None,
    ) -> None:
        # The driver's Idle throttle flag can appear during a sustained loaded
        # workload, so require persistence rather than filtering that flag.
        if measured_mv is None or util < 35:
            self._voltage_streak_mv.clear()
            return
        sample_time_s = time.monotonic() if sample_time_s is None else sample_time_s
        if self._voltage_streak_mv and (
            sample_time_s - self._voltage_streak_mv[-1][0] > max(2.5 * self.poll_seconds, 2.5)
            or abs(measured_mv - self._voltage_streak_mv[0][1]) > 12.5
        ):
            self._voltage_streak_mv.clear()
        self._voltage_streak_mv.append((sample_time_s, measured_mv))
        if (len(self._voltage_streak_mv) >= 3
                and sample_time_s - self._voltage_streak_mv[0][0] >= 2.0):
            lowest = min(value for _, value in self._voltage_streak_mv)
            if self._sustained_min_voltage_mv is None or lowest < self._sustained_min_voltage_mv:
                self._sustained_min_voltage_mv = lowest

    def _clock_p95_mhz(self) -> Optional[float]:
        if not self._clock_samples:
            return None
        values = sorted(self._clock_samples)
        if len(values) == 1:
            return float(values[0])
        idx = int(round(0.95 * (len(values) - 1)))
        idx = max(0, min(idx, len(values) - 1))
        return float(values[idx])

    def _record_point_sample(self, voltage_mv, clock_mhz, utilization):
        if utilization < 35:
            return
        loaded, measured, matched = self._point_counts
        valid = voltage_mv is not None
        on_target = (valid
                     and abs(voltage_mv - self.point_voltage_mv) <= self.point_voltage_tolerance_mv
                     and abs(clock_mhz - self.point_clock_mhz) <= 15.0)
        self._point_counts = (loaded + 1, measured + int(valid), matched + int(on_target))

    def point_marker(self):
        return self._point_counts

    def point_coverage(self, marker=(0, 0, 0)):
        loaded, measured, matched = (v - old for v, old in zip(self._point_counts, marker))
        return {"loaded_samples": loaded, "measured_samples": measured,
                "matched_samples": matched,
                "coverage_pct": 100.0 * matched / loaded if loaded else 0.0,
                "target_voltage_mv": self.point_voltage_mv,
                "target_clock_mhz": self.point_clock_mhz,
                "voltage_tolerance_mv": self.point_voltage_tolerance_mv}

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
                "loaded_sample_count": 0.0,
                "measured_loaded_sample_count": 0.0,
                "measured_loaded_ratio_pct": 0.0,
                "min_measured_loaded_voltage_mv": 0.0,
                "sustained_min_measured_loaded_voltage_mv": 0.0,
                "max_measured_loaded_voltage_mv": 0.0,
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
            "loaded_sample_count": float(self._loaded_sample_count),
            "measured_loaded_sample_count": float(len(self._measured_loaded_voltages_mv)),
            "measured_loaded_ratio_pct": (
                100.0 * len(self._measured_loaded_voltages_mv) / self._loaded_sample_count
                if self._loaded_sample_count else 0.0
            ),
            "min_measured_loaded_voltage_mv": (
                min(self._measured_loaded_voltages_mv) if self._measured_loaded_voltages_mv else 0.0
            ),
            "sustained_min_measured_loaded_voltage_mv": self._sustained_min_voltage_mv or 0.0,
            "max_measured_loaded_voltage_mv": (
                max(self._measured_loaded_voltages_mv) if self._measured_loaded_voltages_mv else 0.0
            ),
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
