import csv
import json
import shutil
import threading
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import SessionState, CurvePoint
from ..utils import eprint, now_utc_iso, ensure_dir
from ..curve import load_curve_csv, write_curve_csv, mv_to_uv
from ..session import save_session
from ...hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from .confident import evaluate_candidate_confident
from .recovery import revert_to_last_good, _check_for_manual_recovery


def _mvscan_candidates_mvs(
    stock_points: List[CurvePoint],
    min_mv: int,
    max_mv: int,
) -> List[int]:
    lo = min(min_mv, max_mv)
    hi = max(min_mv, max_mv)
    bins = sorted(
        {
            int(round(p.voltage_uv / 1000.0))
            for p in stock_points
            if lo <= int(round(p.voltage_uv / 1000.0)) <= hi
        },
        reverse=True,
    )
    if not bins:
        raise ValueError(
            f"No stock VF bins found in range {lo}-{hi} mV. "
            "Adjust --bin-min-mv / --bin-max-mv."
        )
    return bins

def _build_mvscan_cap_curve(
    stock_points: List[CurvePoint],
    target_mv: int,
) -> Tuple[List[CurvePoint], int, int]:
    target_uv = mv_to_uv(target_mv)
    anchor_idx = min(range(len(stock_points)), key=lambda i: abs(stock_points[i].voltage_uv - target_uv))
    anchor_uv = stock_points[anchor_idx].voltage_uv
    anchor_freq_khz = stock_points[anchor_idx].freq_khz
    capped: List[CurvePoint] = []
    for p in stock_points:
        if p.voltage_uv >= anchor_uv:
            capped.append(CurvePoint(p.voltage_uv, anchor_freq_khz))
        else:
            capped.append(CurvePoint(p.voltage_uv, p.freq_khz))
    return capped, anchor_uv, anchor_freq_khz

def _mvscan_rank_key(row: dict, objective: str) -> Tuple[float, float, float, float, float]:
    p95 = float(row.get("p95_clock_mhz", 0.0) or 0.0)
    avg = float(row.get("avg_clock_mhz", 0.0) or 0.0)
    severe = float(row.get("throttle_severe_ratio_pct", 0.0) or 0.0)
    pwr = float(row.get("throttle_pwr_ratio_pct", 0.0) or 0.0)
    mv = float(row.get("target_mv", 0.0) or 0.0)
    if objective == "max-clock":
        return (p95, avg, -severe, -pwr, -mv)
    if objective == "min-cap":
        return (-severe, -pwr, p95, avg, -mv)
    score = p95 - (2.0 * severe) - (0.5 * pwr)
    return (score, p95, -severe, -pwr, -mv)

def _safe_metric(metrics: Optional[dict], key: str) -> float:
    if not metrics:
        return 0.0
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0

def _row_ok(row: dict) -> bool:
    v = row.get("ok")
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() == "true"

def run_mvscan_session(
    state: SessionState,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
) -> None:
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    last_good_csv = Path(state.last_good_curve_csv)
    candidates_mvs = _mvscan_candidates_mvs(stock_points, state.bin_min_mv, state.bin_max_mv)
    total = len(candidates_mvs)
    start_idx = max(0, min(state.current_step, total))

    print("\n=== VoltVandal - mvscan mode ===")
    print(
        f"  Objective: {state.mvscan_objective} | "
        f"Candidates: {total} bins (high->low) in {min(state.bin_min_mv, state.bin_max_mv)}-{max(state.bin_min_mv, state.bin_max_mv)} mV"
    )
    if start_idx > 0:
        print(f"  Resuming from candidate {start_idx + 1}/{total}")

    results_csv = out_dir / "mvscan_results.csv"
    steps_log = out_dir / "steps.jsonl"
    rows: List[dict] = []
    if start_idx > 0 and results_csv.exists():
        try:
            with results_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
    consecutive_failures = 0

    for idx in range(start_idx, total):
        target_mv = candidates_mvs[idx]
        label = f"mvscan_step{idx:03d}_{target_mv}mv"
        _check_for_manual_recovery(state, label, manual_recovery_event)
        if interrupted_event.is_set():
            eprint("\nInterrupted - reverting...")
            try:
                revert_to_last_good(state)
            except Exception as ex:
                eprint(f"WARNING: revert failed: {ex}")
            raise KeyboardInterrupt("User pressed Ctrl+C")

        curve_points, anchor_uv, anchor_freq_khz = _build_mvscan_cap_curve(stock_points, target_mv)
        candidate_csv = out_dir / f"{label}.csv"
        write_curve_csv(candidate_csv, curve_points)

        print(f"\n== {label} ==")
        print(f"  Cap: {anchor_uv//1000} mV | Plateau: {anchor_freq_khz//1000} MHz")
        result = evaluate_candidate_confident(
            state,
            candidate_csv,
            label,
            interrupted_event,
            manual_recovery_event,
            max_freq_mhz=anchor_freq_khz // 1000,
        )
        if result.ok:
            print("Result: PASS")
        else:
            print(f"Result: FAIL | {result.reason}")

        if result.ok:
            shutil.copyfile(candidate_csv, last_good_csv)
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            revert_to_last_good(state)

        row = {
            "utc": now_utc_iso(),
            "step": idx,
            "target_mv": int(target_mv),
            "anchor_mv": int(anchor_uv // 1000),
            "plateau_mhz": int(anchor_freq_khz // 1000),
            "ok": bool(result.ok),
            "reason": result.reason,
            "avg_clock_mhz": _safe_metric(result.metrics, "avg_clock_mhz"),
            "p95_clock_mhz": _safe_metric(result.metrics, "p95_clock_mhz"),
            "max_clock_mhz": _safe_metric(result.metrics, "max_clock_mhz"),
            "throttle_any_ratio_pct": _safe_metric(result.metrics, "throttle_any_ratio_pct"),
            "throttle_pwr_ratio_pct": _safe_metric(result.metrics, "throttle_pwr_ratio_pct"),
            "throttle_severe_ratio_pct": _safe_metric(result.metrics, "throttle_severe_ratio_pct"),
            "sample_count": _safe_metric(result.metrics, "sample_count"),
            "telemetry_max_temp_c": result.telemetry_max_temp_c or "",
            "telemetry_max_power_w": result.telemetry_max_power_w or "",
        }
        rows.append(row)
        with steps_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(result), "utc": row["utc"], "label": label, "step": idx}) + "\n")

        state.current_step = idx + 1
        save_session(state)

        if not result.ok and "GPU_DRIVER_RESET_DETECTED" in result.reason:
            print("Stopping mvscan early: GPU driver reset detected.")
            break
        if consecutive_failures >= 3:
            print("Stopping mvscan early: 3 consecutive failures.")
            break

    fieldnames = [
        "utc",
        "step",
        "target_mv",
        "anchor_mv",
        "plateau_mhz",
        "ok",
        "reason",
        "avg_clock_mhz",
        "p95_clock_mhz",
        "max_clock_mhz",
        "throttle_any_ratio_pct",
        "throttle_pwr_ratio_pct",
        "throttle_severe_ratio_pct",
        "sample_count",
        "telemetry_max_temp_c",
        "telemetry_max_power_w",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    stable_rows = [r for r in rows if _row_ok(r)]
    if not stable_rows:
        print("\n=== mvscan complete: no stable candidates found (kept last good curve) ===")
        return

    best = max(stable_rows, key=lambda r: _mvscan_rank_key(r, state.mvscan_objective))
    best_idx = int(best["step"])
    best_mv = int(best["target_mv"])
    best_label = f"mvscan_step{best_idx:03d}_{best_mv}mv"
    best_csv = out_dir / f"{best_label}.csv"
    if best_csv.exists():
        shutil.copyfile(best_csv, last_good_csv)
        try:
            nvapi_apply_curve_safe(state.gpu, best_csv, timeout_seconds=12.0)
        except Exception:
            pass
        shutil.copyfile(best_csv, out_dir / "mvscan_best_curve.csv")

    print("\n=== mvscan complete ===")
    print(
        f"Best candidate: {best_mv} mV | "
        f"P95 {float(best['p95_clock_mhz']):.0f} MHz | "
        f"SevereCap {float(best['throttle_severe_ratio_pct']):.1f}% | "
        f"PwrCap {float(best['throttle_pwr_ratio_pct']):.1f}%"
    )
    print(f"Results saved: {results_csv}")
