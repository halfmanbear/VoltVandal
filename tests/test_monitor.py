from voltvandal.hardware.monitor import _has_actionable_throttle, _next_throttle_streak
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
