import json

from voltvandal.core.anchors import (
    anchor_ceiling, build_curve, pick_anchors, plateau_starts,
    proven_gains_from_flightlog, search_safe_gain,
)
from voltvandal.core.models import CurvePoint

STEP = 15000


def _stock():
    freqs = [1500, 1500, 1530, 1560, 1560, 1560, 1620, 1650]
    return [CurvePoint(800000 + 6250 * i, f * 1000) for i, f in enumerate(freqs)]


def test_plateau_starts_skips_flat_bins_and_respects_range():
    pts = _stock()
    assert plateau_starts(pts, 0, 10**7) == [0, 2, 3, 6, 7]
    assert plateau_starts(pts, 806250, 825000) == [2, 3]


def test_pick_anchors_includes_ends_and_spreads():
    assert pick_anchors(list(range(10)), 3) == [0, 4, 9] or pick_anchors(list(range(10)), 3) == [0, 5, 9]
    assert pick_anchors([4, 7], 5) == [4, 7]
    assert pick_anchors([], 5) == []
    picks = pick_anchors(list(range(20)), 5)
    assert picks[0] == 0 and picks[-1] == 19 and len(picks) == 5


def test_search_safe_gain_stops_at_first_failure_and_never_steps_up():
    calls = []

    def test(gain):
        calls.append(gain)
        return gain <= 7 * STEP

    assert search_safe_gain(test, STEP, 30 * STEP) == 7 * STEP
    assert calls == [g * STEP for g in range(1, 9)]


def test_search_safe_gain_zero_when_first_step_fails_and_never_exceeds_ceiling():
    assert search_safe_gain(lambda g: False, STEP, 30 * STEP) == 0
    calls = []
    assert search_safe_gain(lambda g: calls.append(g) or True, STEP, 10 * STEP) == 10 * STEP
    assert max(calls) == 10 * STEP


def test_search_safe_gain_start_skips_known_safe_steps():
    calls = []
    search_safe_gain(lambda g: calls.append(g) or g < 6 * STEP, STEP, 30 * STEP, start_khz=4 * STEP + 5)
    assert calls == [4 * STEP, 5 * STEP, 6 * STEP]


def test_anchor_ceiling_uses_cap_until_gains_are_proven():
    assert anchor_ceiling([], 10 * STEP, STEP) == 10 * STEP
    assert anchor_ceiling([0, 4 * STEP], 10 * STEP, STEP) == 6 * STEP
    assert anchor_ceiling([9 * STEP], 10 * STEP, STEP) == 10 * STEP


def test_proven_gains_from_flightlog_uses_only_passes(tmp_path):
    stock = _stock()
    anchors = [0, 2]
    rows = [
        {"ev": "candidate_result", "label": "anchor_800mv_1560mhz_run1", "ok": True},
        {"ev": "candidate_result", "label": "anchor_800mv_1575mhz_run1", "ok": False},
        {"ev": "candidate_result", "label": "anchor_812mv_1590mhz_run1", "ok": True},
        {"ev": "candidate_begin", "label": "anchor_812mv_1650mhz_run1"},
        {"ev": "candidate_result", "label": "anchor_999mv_2000mhz_run1", "ok": True},
    ]
    path = tmp_path / "flight.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    assert proven_gains_from_flightlog(path, stock, anchors) == {0: 60000, 2: 60000}
    assert proven_gains_from_flightlog(tmp_path / "missing.jsonl", stock, anchors) == {}


def test_build_curve_interpolates_applies_margin_and_stays_monotonic():
    stock = _stock()
    out = build_curve(stock, {0: 4 * STEP, 7: 10 * STEP}, STEP, STEP)
    gains = [(o.freq_khz - s.freq_khz) // STEP for o, s in zip(out, stock)]
    assert gains[0] == 3 and gains[-1] == 9
    assert all(a.freq_khz <= b.freq_khz for a, b in zip(out, out[1:]))
    assert all(g >= 0 for g in gains)


def test_build_curve_monotonic_when_lower_anchor_gains_more():
    stock = _stock()
    out = build_curve(stock, {0: 10 * STEP, 7: 0}, 0, STEP)
    assert all(a.freq_khz <= b.freq_khz for a, b in zip(out, out[1:]))
    assert out[-1].freq_khz == stock[-1].freq_khz
