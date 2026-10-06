"""Choose a contiguous vlock sweep from measured, loaded voltage samples."""

from typing import List, Mapping, Optional

from .models import CurvePoint


MIN_LOADED_SAMPLES = 3
MIN_MEASURED_RATIO_PCT = 80.0
NEIGHBOUR_BINS = 2


def lower_bin_from_metrics(
    metrics: Optional[Mapping], stock_points: List[CurvePoint], anchor_idx: int
) -> Optional[int]:
    """Return a conservative lower bound, or None when coverage is inconclusive.

    All bins between the bound and anchor are tested; no intervening curve
    point is interpolated or silently changed.
    """
    if not metrics or anchor_idx <= 0:
        return None
    try:
        loaded = float(metrics["loaded_sample_count"])
        measured = float(metrics["measured_loaded_sample_count"])
        minimum_mv = float(metrics["sustained_min_measured_loaded_voltage_mv"])
    except (KeyError, TypeError, ValueError):
        return None
    if loaded < MIN_LOADED_SAMPLES or measured < MIN_LOADED_SAMPLES:
        return None
    if 100.0 * measured / loaded < MIN_MEASURED_RATIO_PCT or minimum_mv <= 0:
        return None
    observed_idx = min(
        range(len(stock_points)),
        key=lambda i: abs(stock_points[i].voltage_uv / 1000.0 - minimum_mv),
    )
    return max(0, min(anchor_idx, observed_idx) - NEIGHBOUR_BINS)
