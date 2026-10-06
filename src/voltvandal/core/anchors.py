"""Anchor-based curve tuning: test a few real operating points, interpolate the rest."""

import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from .models import CurvePoint

ANCHOR_COUNT = 5
SAFE_MARGIN_STEPS = 4     # gain removed from every proven anchor result when building the curve
CEILING_SLACK_STEPS = 2   # how far past proven gains the next anchor may probe
_PASS_LABEL_RE = re.compile(r"^anchor_(\d+)mv_(\d+)mhz")


def plateau_starts(points: List[CurvePoint], min_uv: int, max_uv: int) -> List[int]:
    """Indices of the first bin of each distinct clock within [min_uv, max_uv].

    Higher bins on a flat stretch run at the first bin's voltage, so only these
    bins are real, separately observable operating points.
    """
    out: List[int] = []
    for i, p in enumerate(points):
        if not min_uv <= p.voltage_uv <= max_uv:
            continue
        if i == 0 or p.freq_khz != points[i - 1].freq_khz:
            out.append(i)
    return out


def pick_anchors(candidates: List[int], count: int = ANCHOR_COUNT) -> List[int]:
    """Evenly spread picks that always include the lowest and highest candidate."""
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return list(candidates)
    if count == 1:
        return [candidates[-1]]
    last = len(candidates) - 1
    picks = {round(k * last / (count - 1)) for k in range(count)}
    return [candidates[i] for i in sorted(picks)]


def search_safe_gain(
    test: Callable[[int], bool],
    step_khz: int,
    ceiling_khz: int,
    start_khz: int = 0,
) -> int:
    """Highest passing gain from a single-step ascent; 0 if nothing passed.

    Never steps up after a failure and never tests above ceiling_khz, so the
    first failing point is the last point tried. start_khz skips steps already
    known safe (rounded down to a step multiple, at least one step).
    """
    best = 0
    gain = max(step_khz, start_khz // step_khz * step_khz)
    while gain <= ceiling_khz:
        if not test(gain):
            break
        best = gain
        gain += step_khz
    return best


def anchor_ceiling(proven_khz: Iterable[int], cap_khz: int, step_khz: int) -> int:
    """Gain ceiling for the next anchor: the profile cap, tightened once gains are proven.

    Gains are not assumed monotonic in voltage, so proven results only limit how
    far beyond them the search may probe (CEILING_SLACK_STEPS steps).
    """
    proven = [g for g in proven_khz if g > 0]
    if not proven:
        return cap_khz
    return min(cap_khz, max(proven) + CEILING_SLACK_STEPS * step_khz)


def proven_gains_from_flightlog(path, stock: List[CurvePoint], anchors: List[int]) -> Dict[int, int]:
    """Highest gain each anchor passed, recovered from flight.jsonl pass records."""
    by_mv = {stock[i].voltage_uv // 1000: i for i in anchors}
    found: Dict[int, int] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return found
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        m = _PASS_LABEL_RE.match(str(ev.get("label", "")))
        if ev.get("ev") != "candidate_result" or not ev.get("ok") or not m or int(m[1]) not in by_mv:
            continue
        idx = by_mv[int(m[1])]
        gain = int(m[2]) * 1000 - stock[idx].freq_khz
        if gain > found.get(idx, 0):
            found[idx] = gain
    return found


def build_curve(
    stock: List[CurvePoint],
    anchor_gains_khz: Dict[int, int],
    margin_khz: int,
    step_khz: int,
) -> List[CurvePoint]:
    """Stock curve plus per-bin gains interpolated linearly in voltage.

    Each anchor gain is reduced by margin_khz. Outside the anchor range the
    nearest anchor's gain applies. Gains snap down to step_khz and the result
    is made monotonic by lowering (never raising) bins.
    """
    if not anchor_gains_khz:
        return [CurvePoint(p.voltage_uv, p.freq_khz) for p in stock]
    knots = sorted(
        (stock[i].voltage_uv, max(0, g - margin_khz)) for i, g in anchor_gains_khz.items()
    )
    out: List[CurvePoint] = []
    for p in stock:
        v = p.voltage_uv
        if v <= knots[0][0]:
            gain = knots[0][1]
        elif v >= knots[-1][0]:
            gain = knots[-1][1]
        else:
            gain = knots[-1][1]
            for (v0, g0), (v1, g1) in zip(knots, knots[1:]):
                if v0 <= v <= v1:
                    gain = g0 + (g1 - g0) * (v - v0) / (v1 - v0)
                    break
        gain = int(gain // step_khz) * step_khz
        out.append(CurvePoint(v, p.freq_khz + gain))
    for i in range(len(out) - 2, -1, -1):
        if out[i].freq_khz > out[i + 1].freq_khz:
            out[i] = CurvePoint(out[i].voltage_uv, out[i + 1].freq_khz)
    return out
