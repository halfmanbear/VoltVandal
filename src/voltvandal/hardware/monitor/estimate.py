"""Curve-based voltage/frequency estimates for the live display and sampling."""

from typing import Optional

from ...core.curve import load_curve_csv


class CurveEstimateMixin:
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
