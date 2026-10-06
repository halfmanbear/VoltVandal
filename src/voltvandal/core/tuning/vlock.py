import shutil
import threading
from pathlib import Path
from typing import List, Optional

from ..models import SessionState, CurvePoint
from ..utils import ensure_dir
from ..curve import (
    load_curve_csv, write_curve_csv, mv_to_uv, mhz_to_khz,
    _build_vlock_curve, _build_vlock_phase2_curves,
)
from ..session import save_session
from ...hardware.nvapi import apply_curve_safe as nvapi_apply_curve_safe
from .autoplan import (
    _AUTO_PLAN_MODES, _AUTO_PLAN_PROBE_SECONDS, _AUTO_PLAN_FINAL_SECONDS,
    _run_auto_plan_suite, _update_auto_plan_bound,
)
from .confident import evaluate_candidate_confident
from .recovery import revert_to_last_good, _check_for_manual_recovery


def _snap_down_to_stock_bin_khz(stock_points: List[CurvePoint], requested_khz: int) -> int:
    """
    Snap a requested frequency down to the nearest stock VF-bin frequency.

    If the request is below all known bins, clamp to the minimum stock bin.
    """
    bins = sorted({p.freq_khz for p in stock_points if p.freq_khz > 0})
    if not bins:
        return requested_khz
    lower_or_equal = [f for f in bins if f <= requested_khz]
    if lower_or_equal:
        return lower_or_equal[-1]
    return bins[0]


def _next_lower_stock_bin_khz(
    stock_points: List[CurvePoint],
    current_khz: int,
    min_khz: int,
) -> Optional[int]:
    """
    Return the next lower stock VF-bin frequency below current_khz, clamped to min_khz.
    """
    bins = sorted({p.freq_khz for p in stock_points if p.freq_khz > 0})
    lower_bins = [f for f in bins if min_khz <= f < current_khz]
    return lower_bins[-1] if lower_bins else None


def run_vlock_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    # Ported from voltvandal.py with necessary adjustments
    if state.active_candidate_label:
        raise RuntimeError(f"Previous run ended during {state.active_candidate_label}; refusing automatic retry")
    if state.vlock_phase == "failed":
        raise RuntimeError("Vlock session failed; start a new session after reviewing the failure")
    if state.vlock_phase == "inconclusive":
        raise RuntimeError("Vlock lower-bin coverage was inconclusive; start a new session")
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    target_uv = mv_to_uv(state.vlock_target_mv)
    anchor_idx = min(range(len(stock_points)), key=lambda i: abs(stock_points[i].voltage_uv - target_uv))
    anchor_v_uv = stock_points[anchor_idx].voltage_uv
    anchor_stock_f_khz = stock_points[anchor_idx].freq_khz
    oc_base_freq_khz = state.vlock_oc_base_freq_khz or anchor_stock_f_khz
    requested_start_khz = mhz_to_khz(state.vlock_start_freq_mhz) if state.vlock_start_freq_mhz > 0 else oc_base_freq_khz
    oc_start_freq_khz = _snap_down_to_stock_bin_khz(stock_points, requested_start_khz)
    step_khz = mhz_to_khz(state.step_mhz)
    coarse_mult = 2
    if state.vlock_anchor_freq_khz <= 0:
        state.vlock_anchor_freq_khz = anchor_stock_f_khz

    print(f"\n=== VoltVandal — vlock mode ===\n  Anchor: {anchor_v_uv//1000} mV | Stock: {anchor_stock_f_khz//1000} MHz")
    if state.vlock_start_freq_mhz > 0:
        if oc_start_freq_khz != requested_start_khz:
            print(
                f"  Phase 1 start frequency adjust: {requested_start_khz//1000} MHz "
                f"-> {oc_start_freq_khz//1000} MHz "
                "(auto adjusted to closest round tuning bin)."
            )
        else:
            print(f"  Phase 1 start frequency override: {oc_start_freq_khz//1000} MHz")

    if state.auto_plan and not state.auto_plan_probe_done:
        if state.vlock_phase != "oc" or state.current_step != 0:
            state.auto_plan_fallback_full = True
            state.auto_plan_min_bin_idx = 0
            state.auto_plan_modes = state.doloming_modes or state.doloming_mode
            print("Existing vlock progress has no stock probe; using the full sweep.")
        else:
            print("\n== Automatic stock-curve coverage probe ==")
            state.active_candidate_label = "vlock_plan_stock_probe"
            save_session(state)
            probe = _run_auto_plan_suite(
                state, Path(state.stock_curve_csv), "vlock_plan_stock_probe",
                _AUTO_PLAN_PROBE_SECONDS, interrupted_event, manual_recovery_event,
            )
            if not probe.ok:
                state.vlock_phase = "failed"
                save_session(state)
                raise RuntimeError(f"Stock coverage probe stopped: {probe.reason}")
            state.active_candidate_label = ""
            state.auto_plan_modes = _AUTO_PLAN_MODES
            _update_auto_plan_bound(state, probe.metrics, stock_points, anchor_idx)
        state.auto_plan_probe_done = True
        save_session(state)
    
    # Phase 1: OC search
    if state.vlock_phase == "oc":
        while state.vlock_phase == "oc":
            _check_for_manual_recovery(state, "vlock_p1", manual_recovery_event)
            step = state.current_step
            if step > state.max_steps:
                state.vlock_phase = "uv"
                state.vlock_uv_bin_idx = anchor_idx - 1
                state.current_step = 0
                state.vlock_last_fail_step = -1
                save_session(state)
                break

            cand_freq = oc_start_freq_khz + mhz_to_khz(state.step_mhz * step)
            cand_pts = _build_vlock_curve(stock_points, anchor_idx, anchor_v_uv, cand_freq, 0)
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, cand_pts)
            
            _p1_mode = "coarse" if state.vlock_last_fail_step < 0 else "fine"
            label = f"vlock_p1_step{step:03d}_{cand_freq//1000}mhz_{_p1_mode}"
            print(f"\n== {label} ==")
            point_options = {"target_point": CurvePoint(anchor_v_uv, cand_freq)} if state.point_lock else {}
            state.active_candidate_label = label
            save_session(state)
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, max_freq_mhz=cand_freq//1000, **point_options)
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
                if any(token in result.reason for token in (
                    "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT", "CUDA_RUNTIME_ERROR", "STRESS_ERROR"
                )):
                    state.vlock_phase = "failed"
                    save_session(state)
                    raise RuntimeError(f"Vlock tuning stopped: {result.reason}")
            state.active_candidate_label = ""
            save_session(state)
            
            if result.ok:
                shutil.copyfile(cand_csv, Path(state.last_good_curve_csv))
                state.vlock_anchor_freq_khz = cand_freq
                if state.auto_plan and not state.auto_plan_fallback_full and not state.point_lock:
                    _update_auto_plan_bound(state, result.metrics, stock_points, anchor_idx)
                if state.vlock_last_fail_step >= 0:
                    next_step = step + 1
                    if next_step >= state.vlock_last_fail_step:
                        state.vlock_phase = "uv"
                        state.vlock_uv_bin_idx = anchor_idx - 1
                        state.current_step = 0
                        state.vlock_last_fail_step = -1
                    else:
                        state.current_step = next_step
                else:
                    state.current_step = step + coarse_mult
                save_session(state)
            else:
                revert_to_last_good(state)
                if state.vlock_last_fail_step >= 0:
                    state.vlock_phase = "uv"
                    state.vlock_uv_bin_idx = anchor_idx - 1
                    state.current_step = 0
                    state.vlock_last_fail_step = -1
                    save_session(state)
                    break

                prev_coarse_step = max(0, step - coarse_mult)
                fine_start_step = prev_coarse_step + 1
                if fine_start_step >= step:
                    failed_freq_khz = cand_freq
                    lowered_start_khz = _next_lower_stock_bin_khz(
                        stock_points,
                        cand_freq,
                        anchor_stock_f_khz,
                    )
                    if lowered_start_khz is not None:
                        print(
                            f"Phase 1 start {cand_freq//1000} MHz failed immediately; "
                            f"lowering start to {lowered_start_khz//1000} MHz and retrying."
                        )
                        oc_start_freq_khz = lowered_start_khz
                        state.vlock_start_freq_mhz = lowered_start_khz // 1000
                        # Keep subsequent search below the original immediate-fail
                        # ceiling to avoid jumping to higher coarse points.
                        delta_khz = max(0, failed_freq_khz - lowered_start_khz)
                        step_at_failed = (delta_khz + step_khz - 1) // step_khz
                        state.vlock_last_fail_step = max(1, int(step_at_failed + 1))
                        state.current_step = 0
                        state.vlock_anchor_freq_khz = anchor_stock_f_khz
                        save_session(state)
                        continue
                    state.vlock_phase = "uv"
                    state.vlock_uv_bin_idx = anchor_idx - 1
                    state.current_step = 0
                    state.vlock_last_fail_step = -1
                    save_session(state)
                    break

                state.vlock_last_fail_step = step
                state.current_step = fine_start_step
                _lo_freq = oc_start_freq_khz + mhz_to_khz(state.step_mhz * prev_coarse_step)
                _hi_freq = cand_freq
                print(f"Refining Phase 1 between {_lo_freq//1000} and {_hi_freq//1000} MHz using {state.step_mhz} MHz steps.")
                save_session(state)
    
    # Phase 2: UV/OC Shift Sweep
    if state.vlock_phase == "uv":
        oc_gain = state.vlock_anchor_freq_khz - anchor_stock_f_khz
        uv_failed = False
        while True:
            bin_idx = state.vlock_uv_bin_idx
            plan_boundary_reached = (
                state.auto_plan and not state.auto_plan_fallback_full
                and bin_idx < state.auto_plan_min_bin_idx
            )
            if bin_idx < 0 or plan_boundary_reached:
                if state.auto_plan:
                    print("\n== Automatic final-curve validation ==")
                    state.active_candidate_label = "vlock_plan_final"
                    save_session(state)
                    final = _run_auto_plan_suite(
                        state, Path(state.last_good_curve_csv), "vlock_plan_final",
                        _AUTO_PLAN_FINAL_SECONDS, interrupted_event, manual_recovery_event,
                        include_gpuburn=True,
                        modes=state.auto_plan_modes or _AUTO_PLAN_MODES,
                    )
                    if not final.ok:
                        nvapi_apply_curve_safe(state.gpu, Path(state.stock_curve_csv), timeout_seconds=12.0)
                        shutil.copyfile(state.stock_curve_csv, state.last_good_curve_csv)
                        state.vlock_phase = "failed"
                        state.active_candidate_label = ""
                        save_session(state)
                        print(f"Final curve failed ({final.reason}); stock curve restored.")
                        uv_failed = True
                        break
                    state.active_candidate_label = ""
                    save_session(state)
                    if not state.auto_plan_fallback_full:
                        old_bound = state.auto_plan_min_bin_idx
                        _update_auto_plan_bound(state, final.metrics, stock_points, anchor_idx)
                        if state.auto_plan_fallback_full:
                            if bin_idx < 0:
                                state.vlock_uv_bin_idx = anchor_idx - 1
                            save_session(state)
                            continue
                        if state.auto_plan_min_bin_idx < old_bound:
                            save_session(state)
                            continue
                state.vlock_phase = "done"
                save_session(state)
                print("\n=== vlock tuning complete ===")
                break
            _check_for_manual_recovery(state, f"vlock_p2_bin{bin_idx}", manual_recovery_event)
            
            last_good_pts = load_curve_csv(Path(state.last_good_curve_csv))
            test_pts, save_pts = _build_vlock_phase2_curves(stock_points, last_good_pts, bin_idx, anchor_idx, anchor_v_uv, state.vlock_anchor_freq_khz, oc_gain)
            
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, test_pts)
            
            label = f"vlock_p2_bin{bin_idx:03d}_{stock_points[bin_idx].voltage_uv//1000}mv"
            print(f"\n== {label} ==")
            point_options = {"target_point": test_pts[bin_idx]}
            state.active_candidate_label = label
            save_session(state)
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, **point_options)
            if not result.ok and "INCONCLUSIVE_BIN_COVERAGE" in result.reason:
                print(f"Result: INCONCLUSIVE | {result.reason}")
                revert_to_last_good(state)
                state.vlock_phase = "inconclusive"
                state.active_candidate_label = ""
                save_session(state)
                print("Lower-bin sweep stopped; the unexercised candidate was not saved.")
                return
            if result.ok:
                print("Result: PASS")
            else:
                print(f"Result: FAIL | {result.reason}")
                if any(token in result.reason for token in (
                    "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT", "CUDA_RUNTIME_ERROR", "STRESS_ERROR"
                )):
                    state.vlock_phase = "failed"
                    save_session(state)
                    raise RuntimeError(f"Vlock tuning stopped: {result.reason}")
            state.active_candidate_label = ""
            save_session(state)
            
            if result.ok:
                write_curve_csv(Path(state.last_good_curve_csv), save_pts)
                if state.auto_plan and not state.auto_plan_fallback_full and not state.point_lock:
                    _update_auto_plan_bound(state, result.metrics, stock_points, anchor_idx)
            else:
                revert_to_last_good(state)
                uv_failed = True
                save_session(state)
                break
            
            state.vlock_uv_bin_idx -= 1
            save_session(state)

        if uv_failed and state.vlock_phase != "failed":
            print("\n=== vlock tuning stopped on failure (reverted to last good curve) ===")
