from voltvandal.hardware.monitor import _has_actionable_throttle, _next_throttle_streak


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
