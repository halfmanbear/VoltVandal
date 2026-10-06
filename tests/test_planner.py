from voltvandal.core.models import CurvePoint
from voltvandal.core.planner import lower_bin_from_metrics


def _metrics(minimum_mv, loaded=40, measured=40):
    return {
        "loaded_sample_count": loaded,
        "measured_loaded_sample_count": measured,
        "min_measured_loaded_voltage_mv": minimum_mv,
        "sustained_min_measured_loaded_voltage_mv": minimum_mv,
    }


def test_plan_includes_neighbours_and_every_intermediate_bin():
    points = [CurvePoint(mv * 1000, 1000000) for mv in (800, 825, 850, 875, 900, 925)]
    assert lower_bin_from_metrics(_metrics(875), points, 5) == 1
    assert lower_bin_from_metrics(_metrics(925), points, 5) == 3


def test_plan_falls_back_when_loaded_voltage_coverage_is_insufficient():
    points = [CurvePoint(mv * 1000, 1000000) for mv in (800, 850, 900)]
    assert lower_bin_from_metrics(_metrics(850, loaded=40, measured=31), points, 2) is None
    assert lower_bin_from_metrics(_metrics(850, loaded=2, measured=2), points, 2) is None
    assert lower_bin_from_metrics(None, points, 2) is None


def test_plan_ignores_raw_low_outlier_without_sustained_coverage():
    points = [CurvePoint(mv * 1000, 1000000) for mv in (750, 775, 800, 825, 850, 875, 900)]
    metrics = _metrics(775)
    metrics["sustained_min_measured_loaded_voltage_mv"] = 900
    assert lower_bin_from_metrics(metrics, points, 6) == 4
    del metrics["sustained_min_measured_loaded_voltage_mv"]
    assert lower_bin_from_metrics(metrics, points, 6) is None
    metrics["sustained_min_measured_loaded_voltage_mv"] = 0
    assert lower_bin_from_metrics(metrics, points, 6) is None


def test_effective_point_resolves_plateau_to_lowest_bin():
    from voltvandal.core.tuning import effective_point
    pts = [CurvePoint(875000, 1785000), CurvePoint(887500, 1875000),
           CurvePoint(900000, 1875000), CurvePoint(950000, 1875000),
           CurvePoint(981250, 1890000)]
    assert effective_point(pts, pts[3]) == pts[1]
    assert effective_point(pts, pts[0]) == pts[0]
    assert effective_point(pts, pts[4]) == pts[4]
