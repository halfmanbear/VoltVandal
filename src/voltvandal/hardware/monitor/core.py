import csv
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pynvml
except ImportError:
    pynvml = None

from ...core.models import MonitorSnapshot, CurvePoint
from ...core.utils import eprint, now_utc_iso
from ..point_lock import read_voltage_mv
from .decode import (
    _decode_perf_decrease, _decode_throttle, _next_collapse_streak, _next_throttle_streak,
)
from .display import LiveDisplayMixin
from .estimate import CurveEstimateMixin
from .stats import MonitorStatsMixin

# Import native nvapi if available
try:
    from .. import nvapi as _nvapi_native
except ImportError:
    _nvapi_native = None

_TELEMETRY_COLUMNS = [
    "utc", "temp_c", "hotspot_c", "vram_junction_c", "power_w",
    "clock_mhz", "mem_clock_mhz", "util_gpu", "voltage_mv",
    "throttle_reasons", "pstate", "perf_decrease", "topo_gpu_mw",
    "topo_total_mw", "measured_voltage_mv", "voltage_source",
]


class NvmlMonitor(MonitorStatsMixin, CurveEstimateMixin, LiveDisplayMixin):
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
                self._record_throttle_counts(throttle)

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
                    self._render_live_line(
                        temp, hotspot, vram_junc, pstate, clock, mem_clock, util,
                        voltage_mv, voltage_estimated, power, throttle_lbl, pdec_lbl,
                    )

                self._check_abort_limits(temp, hotspot, power, throttle, throttle_lbl)

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

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5.0)
        if self.live_display and self._live_line_len > 0:
            sys.stderr.write("\r" + " " * (self._live_line_len + 2) + "\r")
            sys.stderr.flush()
        try:
            if pynvml: pynvml.nvmlShutdown()
        except Exception: pass
