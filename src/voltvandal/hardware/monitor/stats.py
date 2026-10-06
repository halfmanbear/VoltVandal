"""Sample statistics, point coverage, metrics and abort thresholds."""

import time
from typing import Dict, Optional

from .decode import (
    _COLLAPSE_ABORT_CONSECUTIVE_POLLS, _THROTTLE_ABORT_CONSECUTIVE_POLLS,
    _THROTTLE_IDLE_BIT, _THROTTLE_LABELS, _THROTTLE_PWRCAP_BIT, _THROTTLE_SEVERE_BITS,
)


class MonitorStatsMixin:
    def _record_throttle_counts(self, throttle: int) -> None:
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

    def _check_abort_limits(
        self, temp: int, hotspot: float, power: float, throttle: int, throttle_lbl: str,
    ) -> None:
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
