import csv
from pathlib import Path
from typing import List, Tuple
from .models import CurvePoint

def load_curve_csv(path: Path) -> List[CurvePoint]:
    points: List[CurvePoint] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        norm = {name.lower(): name for name in reader.fieldnames}
        required_lower = {"voltageuv", "frequencykhz"}
        if not required_lower.issubset(set(norm.keys())):
            raise ValueError(
                f"CSV header must include voltageUV and frequencyKHz (case-insensitive), got {reader.fieldnames}"
            )
        col_v = norm["voltageuv"]
        col_f = norm["frequencykhz"]
        for row in reader:
            v = int(row[col_v])
            fk = int(row[col_f])
            points.append(CurvePoint(voltage_uv=v, freq_khz=fk))
    if not points:
        raise ValueError(f"No curve points loaded from {path}")
    return points

def write_curve_csv(path: Path, points: List[CurvePoint]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["voltageUV", "frequencyKHz"])
        for p in points:
            writer.writerow([p.voltage_uv, p.freq_khz])

def mv_to_uv(mv: int) -> int:
    return mv * 1000

def mhz_to_khz(mhz: int) -> int:
    return mhz * 1000

def apply_offsets_to_bin(
    points: List[CurvePoint],
    bin_min_mv: int,
    bin_max_mv: int,
    offset_mv: int,
    offset_mhz: int,
) -> List[CurvePoint]:
    vmin_uv = mv_to_uv(bin_min_mv)
    vmax_uv = mv_to_uv(bin_max_mv)
    dv_uv = mv_to_uv(offset_mv)
    df_khz = mhz_to_khz(offset_mhz)
    new_points: List[CurvePoint] = []
    for p in points:
        if vmin_uv <= p.voltage_uv <= vmax_uv:
            new_v = max(0, p.voltage_uv + dv_uv)
            new_f = max(0, p.freq_khz + df_khz)
            new_points.append(CurvePoint(new_v, new_f))
        else:
            new_points.append(CurvePoint(p.voltage_uv, p.freq_khz))
    return new_points

def _build_vlock_curve(
    stock_points: List[CurvePoint],
    anchor_idx: int,
    anchor_voltage_uv: int,
    anchor_freq_khz: int,
    uv_offset_uv: int,
    cap_freq_khz: int = 0,
) -> List[CurvePoint]:
    effective_cap = cap_freq_khz if cap_freq_khz > 0 else anchor_freq_khz
    out: List[CurvePoint] = []
    for i, p in enumerate(stock_points):
        if i < anchor_idx:
            out.append(CurvePoint(max(p.voltage_uv - uv_offset_uv, 0), p.freq_khz))
        elif i == anchor_idx:
            out.append(CurvePoint(anchor_voltage_uv, min(anchor_freq_khz, effective_cap)))
        else:
            out.append(CurvePoint(p.voltage_uv, min(p.freq_khz, effective_cap)))
    return out

def _build_vlock_phase2_curves(
    stock_points: List[CurvePoint],
    last_good_points: List[CurvePoint],
    bin_idx: int,
    anchor_idx: int,
    anchor_voltage_uv: int,
    anchor_freq_khz: int,
    oc_gain_khz: int,
) -> Tuple[List[CurvePoint], List[CurvePoint]]:
    bin_target_freq = min(
        stock_points[bin_idx].freq_khz + oc_gain_khz, anchor_freq_khz
    )
    test_pts: List[CurvePoint] = []
    save_pts: List[CurvePoint] = []
    for i, p in enumerate(stock_points):
        if i < anchor_idx:
            if i == bin_idx:
                test_pts.append(CurvePoint(p.voltage_uv, bin_target_freq))
                save_pts.append(CurvePoint(p.voltage_uv, bin_target_freq))
            else:
                lg = last_good_points[i]
                test_pts.append(CurvePoint(lg.voltage_uv, lg.freq_khz))
                save_pts.append(CurvePoint(lg.voltage_uv, lg.freq_khz))
        elif i == anchor_idx:
            test_pts.append(CurvePoint(anchor_voltage_uv, anchor_freq_khz))
            save_pts.append(CurvePoint(anchor_voltage_uv, anchor_freq_khz))
        else:
            test_pts.append(CurvePoint(p.voltage_uv, min(p.freq_khz, anchor_freq_khz)))
            save_pts.append(CurvePoint(p.voltage_uv, min(p.freq_khz, anchor_freq_khz)))
    return test_pts, save_pts
