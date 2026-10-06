import csv
from types import SimpleNamespace

import voltvandal.hardware.monitor.core as monitor_module

from voltvandal.hardware.monitor.decode import (
    _has_actionable_throttle, _next_collapse_streak, _next_throttle_streak,
)
from voltvandal.hardware.monitor import NvmlMonitor


def test_has_actionable_throttle_ignores_pure_pwrcap():
    # Pure PwrCap should not be considered actionable for abort logic.
    assert _has_actionable_throttle(0x0000000000000004) is False
    # Idle + PwrCap should also be ignored.
    assert _has_actionable_throttle(0x0000000000000005) is False


def test_has_actionable_throttle_accepts_non_pwrcap_reasons():
    # SwTherm alone is actionable.
    assert _has_actionable_throttle(0x0000000000000020) is True
    # PwrCap + SwTherm is actionable (not pure PwrCap).
    assert _has_actionable_throttle(0x0000000000000024) is True


def test_next_throttle_streak_debounces_transients():
    streak = 0
    # Pure PwrCap does not count.
    streak = _next_throttle_streak(streak, 0x0000000000000004)
    assert streak == 0
    # Actionable reasons count up.
    streak = _next_throttle_streak(streak, 0x0000000000000020)  # SwTherm
    assert streak == 1
    streak = _next_throttle_streak(streak, 0x0000000000000024)  # PwrCap+SwTherm
    assert streak == 2
    # Non-actionable resets streak.
    streak = _next_throttle_streak(streak, 0x0000000000000001)  # Idle
    assert streak == 0


def point_monitor(tmp_path):
    return NvmlMonitor(0, 1.0, 90, 100, 15, 400, False, tmp_path / "telemetry.csv",
                       point_lock_bus=3, point_voltage_mv=900.0,
                       point_clock_mhz=1900.0, point_voltage_tolerance_mv=3.0)


def test_point_coverage_excludes_idle_but_counts_missing_voltage(tmp_path):
    monitor = point_monitor(tmp_path)
    monitor._record_point_sample(900, 1900, 0)
    monitor._record_point_sample(None, 1900, 90)
    monitor._record_point_sample(900, 1900, 90)
    monitor._record_point_sample(950, 1900, 90)
    monitor._record_point_sample(900, 1700, 90)
    coverage = monitor.point_coverage()
    assert coverage["loaded_samples"] == 4
    assert coverage["measured_samples"] == 3
    assert coverage["matched_samples"] == 1
    assert coverage["coverage_pct"] == 25.0


def test_each_workload_has_independent_point_coverage(tmp_path):
    monitor = point_monitor(tmp_path)
    for _ in range(10):
        monitor._record_point_sample(900, 1900, 90)
    marker = monitor.point_marker()
    monitor._record_point_sample(950, 1900, 90)
    coverage = monitor.point_coverage(marker)
    assert coverage["loaded_samples"] == 1
    assert coverage["coverage_pct"] == 0.0


def test_normal_monitor_records_only_real_voltage_as_measured(monkeypatch, tmp_path):
    monitor = NvmlMonitor(0, 1.0, 90, 100, 15, 400, False,
                          tmp_path / "telemetry.csv", measure_voltage=True)
    monitor._voltage_bus = 3
    monkeypatch.setattr("voltvandal.hardware.monitor.core.read_voltage_mv",
                        lambda gpu, bus: 887.5)
    assert monitor._sample_voltage(1740, 90) == (887.5, 887.5, "nvapi_measured")

    def unavailable(gpu, bus):
        raise RuntimeError("unavailable")

    monkeypatch.setattr("voltvandal.hardware.monitor.core.read_voltage_mv", unavailable)
    monkeypatch.setattr(monitor, "_estimate_voltage_mv_from_curve", lambda clock: 890)
    assert monitor._sample_voltage(1740, 90) == (890, None, "curve_estimate")
    assert not monitor.abort_event.is_set()


def test_unlocked_curve_point_coverage_uses_measured_loaded_samples(monkeypatch, tmp_path):
    monitor = NvmlMonitor(
        0, 0.5, 90, 100, 15, 400, False, tmp_path / "telemetry.csv",
        measure_voltage=True, point_voltage_mv=806.25,
        point_clock_mhz=1860.0, point_voltage_tolerance_mv=3.0,
    )
    monitor._voltage_bus = 3
    measured = [887.5, 806.25, 806.25]
    monkeypatch.setattr(
        "voltvandal.hardware.monitor.core.read_voltage_mv",
        lambda gpu, bus: measured.pop(0),
    )
    monitor._sample_voltage(1860, 50)
    monitor._sample_voltage(1860, 50)
    monitor._sample_voltage(1860, 10)  # Transition, not loaded coverage.
    coverage = monitor.point_coverage()
    assert coverage["loaded_samples"] == 2
    assert coverage["measured_samples"] == 2
    assert coverage["matched_samples"] == 1


def test_sustained_voltage_ignores_isolated_low_samples(tmp_path):
    monitor = point_monitor(tmp_path)
    for second in range(3):
        monitor._record_sustained_voltage_sample(1081.25, 75, 0, second)
    assert monitor._sustained_min_voltage_mv == 1081.25

    monitor._record_sustained_voltage_sample(768.75, 65, 1, 3)  # One low transition sample
    monitor._record_sustained_voltage_sample(881.25, 43, 0, 4)  # Another isolated reading
    monitor._record_sustained_voltage_sample(None, 0, 0, 5)
    assert monitor._sustained_min_voltage_mv == 1081.25

    for second, voltage in enumerate((800.0, 806.25, 800.0), start=6):
        monitor._record_sustained_voltage_sample(voltage, 75, 0, second)
    assert monitor._sustained_min_voltage_mv == 800.0


def test_sustained_voltage_accepts_loaded_samples_with_idle_flag(tmp_path):
    monitor = point_monitor(tmp_path)
    for second in range(3):
        monitor._record_sustained_voltage_sample(887.5, 50, 1, second)
    assert monitor._sustained_min_voltage_mv == 887.5


def test_fast_voltage_samples_do_not_count_as_sustained(tmp_path):
    monitor = point_monitor(tmp_path)
    for index in range(10):
        monitor._record_sustained_voltage_sample(800.0, 90, 0, index * 0.1)
    assert monitor._sustained_min_voltage_mv is None
    for index in range(10, 22):
        monitor._record_sustained_voltage_sample(800.0, 90, 0, index * 0.1)
    assert monitor._sustained_min_voltage_mv == 800.0


def test_existing_telemetry_rows_are_marked_unknown(tmp_path):
    path = tmp_path / "telemetry.csv"
    old_header = ["utc", "temp_c", "hotspot_c", "vram_junction_c", "power_w",
                  "clock_mhz", "mem_clock_mhz", "util_gpu", "voltage_mv",
                  "throttle_reasons", "pstate", "perf_decrease", "topo_gpu_mw",
                  "topo_total_mw"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(old_header)
        writer.writerow(["before"] + [""] * (len(old_header) - 1))
    monitor = NvmlMonitor(0, 1.0, 90, 100, 15, 400, False, path)
    monitor._ensure_telemetry_header()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["utc"] == "before"
    assert rows[0]["measured_voltage_mv"] == ""
    assert rows[0]["voltage_source"] == "legacy_unknown"


def test_monitor_csv_keeps_measured_voltage_separate_from_estimates(monkeypatch, tmp_path):
    path = tmp_path / "telemetry.csv"
    monitor = NvmlMonitor(0, 1.0, 90, 100, 15, 400, False, path,
                          live_display=False, measure_voltage=True)
    monitor.handle = object()
    monitor._voltage_bus = 3
    monitor._ensure_telemetry_header()
    fake_nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0, NVML_CLOCK_GRAPHICS=0, NVML_CLOCK_MEM=1,
        nvmlDeviceGetTemperature=lambda handle, sensor: 50,
        nvmlDeviceGetPowerUsage=lambda handle: 200000,
        nvmlDeviceGetClockInfo=lambda handle, kind: 1740 if kind == 0 else 9500,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=90),
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda handle: 0,
    )
    monkeypatch.setattr(monitor_module, "pynvml", fake_nvml)
    monkeypatch.setattr(monitor_module, "read_voltage_mv", lambda gpu, bus: 887.5)
    monkeypatch.setattr(monitor.stop_event, "wait", lambda timeout: monitor.stop_event.set())
    monitor._loop()
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    assert row["measured_voltage_mv"] == "887.5"
    assert row["voltage_mv"] == "887.5"
    assert row["voltage_source"] == "nvapi_measured"
    assert monitor.metrics()["min_measured_loaded_voltage_mv"] == 887.5
    assert monitor.metrics()["measured_loaded_ratio_pct"] == 100.0


def test_idle_gap_after_stress_does_not_signal_driver_reset(monkeypatch, tmp_path):
    monitor = NvmlMonitor(0, 1.0, 90, 100, 15, 400, False,
                          tmp_path / "telemetry.csv", live_display=False)
    monitor.handle = object()
    monitor._ensure_telemetry_header()
    sample = [0]
    fake_nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0, NVML_CLOCK_GRAPHICS=0, NVML_CLOCK_MEM=1,
        nvmlDeviceGetTemperature=lambda handle, sensor: 45,
        nvmlDeviceGetPowerUsage=lambda handle: 200000 if sample[0] == 0 else 10000,
        nvmlDeviceGetClockInfo=lambda handle, kind: (
            1800 if sample[0] == 0 else 210) if kind == 0 else 405,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(
            gpu=75 if sample[0] == 0 else 0),
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda handle: 0,
    )
    monkeypatch.setattr(monitor_module, "pynvml", fake_nvml)

    def next_sample(timeout):
        sample[0] += 1
        if sample[0] >= 5:
            monitor.stop_event.set()

    monkeypatch.setattr(monitor.stop_event, "wait", next_sample)
    monitor._loop()
    assert monitor.metrics()["sample_count"] == 5
    assert not monitor.driver_reset_detected
    assert not monitor.abort_event.is_set()


def test_repeated_nvml_failure_aborts_monitor(monkeypatch, tmp_path):
    monitor = NvmlMonitor(0, 1.0, 90, 100, 15, 400, False,
                          tmp_path / "telemetry.csv", live_display=False)
    monitor.handle = object()

    def unavailable(*args):
        raise RuntimeError("NVML unavailable")

    fake_nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        nvmlDeviceGetTemperature=unavailable,
    )
    monkeypatch.setattr(monitor_module, "pynvml", fake_nvml)
    monkeypatch.setattr(monitor.stop_event, "wait", lambda timeout: (
        monitor.stop_event.set() if monitor._consecutive_errors >= 3 else None
    ))
    monitor._loop()
    assert monitor.abort_event.is_set()
    assert monitor.abort_reason == "TELEMETRY_UNAVAILABLE"


def test_collapse_streak_detects_hung_gpu_power_drop():
    streak = 0
    for _ in range(5):
        streak = _next_collapse_streak(streak, 100.0, 310.0)
    assert streak == 5
    assert _next_collapse_streak(streak, 250.0, 310.0) == 0


def test_collapse_streak_ignores_low_power_runs():
    assert _next_collapse_streak(0, 20.0, 90.0) == 0
