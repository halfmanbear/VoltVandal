import json
import re
import shutil
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from .models import SessionState, CandidateResult, CurvePoint
from .utils import eprint, now_utc_iso, ensure_dir
from .curve import (
    load_curve_csv, write_curve_csv, mv_to_uv, mhz_to_khz,
    _build_vlock_curve, _build_vlock_phase2_curves, apply_offsets_to_bin
)
from .session import save_session, session_paths
from ..hardware.nvapi import apply_curve as nvapi_apply_curve, dump_curve as nvapi_dump_curve
from ..hardware.monitor import NvmlMonitor
from ..stress.runner import run_doloming, run_gpuburn

def _doloming_mode_tag(mode: str) -> str:
    return mode.upper().replace("-", "_")

def _extract_summary_value(text: str, label: str) -> Optional[str]:
    m = re.search(rf"^{re.escape(label)}\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else None

def _extract_first_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else None

def _parse_doloming_stability(out_text: str, mode: str) -> Tuple[bool, Optional[str]]:
    mode_tag = _doloming_mode_tag(mode)
    summary_idx = out_text.rfind("Test Summary:")
    scan_text = out_text[summary_idx:] if summary_idx >= 0 else out_text
    if re.search(r"Error during (?:stress )?test:", scan_text, re.I):
        return False, f"DOLOMING_{mode_tag}_STRESS_ERROR"
    if re.search(r"cudaerror\w*|illegal memory access|device-side assert|unspecified launch failure|launch timeout|driver shutting down", out_text, re.I):
        return False, f"DOLOMING_{mode_tag}_CUDA_RUNTIME_ERROR"

    status = _extract_summary_value(scan_text, "Status")
    if status and re.search(r"\b(unstable|fail(?:ed|ure)?|error)\b", status, re.I):
        return False, f"DOLOMING_{mode_tag}_UNSTABLE_STATUS"

    avg_util = _extract_first_float(_extract_summary_value(scan_text, "Average Utilization"))
    if avg_util is not None and avg_util < 35.0:
        return False, f"DOLOMING_{mode_tag}_LOW_UTILIZATION"

    if mode == "frequency-max":
        score_gamma = _extract_first_float(_extract_summary_value(out_text, "Score_gamma"))
        if score_gamma is not None and score_gamma < 96.0:
            return False, f"DOLOMING_{mode_tag}_LOW_SCORE_GAMMA"
        avg_max_ratio = _extract_first_float(_extract_summary_value(out_text, "Avg/Max ratio"))
        if avg_max_ratio is not None and avg_max_ratio < 0.96:
            return False, f"DOLOMING_{mode_tag}_LOW_FREQ_RATIO"
    return True, None

def evaluate_candidate(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
) -> CandidateResult:
    out_dir = Path(state.out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    try:
        nvapi_apply_curve(state.gpu, candidate_csv)
        interrupted_event.wait(timeout=2.0)
        if interrupted_event.is_set():
            raise KeyboardInterrupt("User pressed Ctrl+C")
    except Exception as ex:
        return CandidateResult(ok=False, reason=f"APPLY_FAILED: {ex}")

    monitor_log = logs_dir / "telemetry.csv"
    monitor: Optional[NvmlMonitor] = None
    abort_event = threading.Event()

    _modes_raw = (state.doloming_modes or "").strip()
    _multi_modes = [m.strip() for m in _modes_raw.split(",") if m.strip()] if _modes_raw else []
    _use_multi = bool(_multi_modes)
    _run_modes = _multi_modes if _use_multi else [state.doloming_mode]
    _secs_each = state.multi_stress_seconds if _use_multi else state.stress_seconds
    _expected_test_seconds = 0
    if state.doloming: _expected_test_seconds += _secs_each * len(_run_modes)
    if state.gpuburn: _expected_test_seconds += state.stress_seconds

    try:
        monitor = NvmlMonitor(
            gpu_index=state.gpu,
            poll_seconds=state.poll_seconds,
            temp_limit_c=state.temp_limit_c,
            hotspot_limit_c=state.hotspot_limit_c,
            hotspot_offset_c=state.hotspot_offset_c,
            power_limit_w=state.power_limit_w,
            abort_on_throttle=state.abort_on_throttle,
            log_csv=monitor_log,
            curve_csv=candidate_csv,
            expected_test_seconds=_expected_test_seconds if _expected_test_seconds > 0 else None,
            live_display=state.live_display,
        )
        monitor.start()
        abort_event = monitor.abort_event
    except Exception as e:
        eprint(f"Failed to start monitor: {e}")
        return CandidateResult(ok=False, reason=f"MONITOR_START_FAILED: {e}")

    stress_exit_codes = {}
    if state.doloming:
        _dolo_timeout = state.stress_timeout if state.stress_timeout is not None else max(_secs_each * 5, 300)
        for _mode in _run_modes:
            dololog = logs_dir / f"{candidate_label}_doloming_{_mode}.log"
            rc, out_text = run_doloming(
                state.doloming, state.gpu, _mode, _secs_each, None, dololog,
                abort_event, manual_recovery_event, interrupted_event,
                stress_timeout=_dolo_timeout, max_freq_mhz=max_freq_mhz,
            )
            _key = f"doloming_{_mode}" if _use_multi else "doloming"
            stress_exit_codes[_key] = rc
            if rc != 0:
                if monitor: monitor.stop()
                _reason = "MANUAL_RECOVERY_REQUESTED" if rc == 997 else f"DOLOMING_{_mode.upper().replace('-', '_')}_RC_{rc}"
                return CandidateResult(False, _reason, monitor.max_temp if monitor else None, monitor.max_power if monitor else None, monitor.any_throttle if monitor else None, stress_exit_codes)
            if re.search(r"\boom\b|\bout of memory\b|\bfailed\b|errors?\s*[:=]\s*[1-9]|cudaerror\w*|illegal memory access|device-side assert|unspecified launch failure", out_text, re.I):
                if monitor: monitor.stop()
                return CandidateResult(False, f"DOLOMING_{_mode.upper().replace('-', '_')}_OUTPUT_ERROR_KEYWORD", monitor.max_temp if monitor else None, monitor.max_power if monitor else None, monitor.any_throttle if monitor else None, stress_exit_codes)
            _stable, _instability_reason = _parse_doloming_stability(out_text, _mode)
            if not _stable:
                if monitor: monitor.stop()
                return CandidateResult(False, _instability_reason, monitor.max_temp if monitor else None, monitor.max_power if monitor else None, monitor.any_throttle if monitor else None, stress_exit_codes)

    if state.gpuburn:
        burnlog = logs_dir / f"{candidate_label}_gpuburn.log"
        _burn_timeout = state.stress_timeout if state.stress_timeout is not None else state.stress_seconds * 5
        rc, out_text, parsed_ok = run_gpuburn(
            state.gpuburn, state.stress_seconds, None, burnlog,
            abort_event, manual_recovery_event, interrupted_event,
            stress_timeout=_burn_timeout,
        )
        stress_exit_codes["gpuburn"] = rc
        if rc != 0 or not parsed_ok or re.search(r"\bnan\b|\bfailed\b|errors?\s*[:=]\s*[1-9]", out_text, re.I):
            if monitor: monitor.stop()
            _reason = f"GPUBURN_RC_{rc}" if rc != 0 else ("GPUBURN_ERRORS_DETECTED" if not parsed_ok else "GPUBURN_OUTPUT_ERROR_KEYWORD")
            return CandidateResult(False, _reason, monitor.max_temp if monitor else None, monitor.max_power if monitor else None, monitor.any_throttle if monitor else None, stress_exit_codes)

    if monitor:
        if monitor.abort_event.is_set():
            monitor.stop()
            return CandidateResult(False, "GPU_DRIVER_RESET_DETECTED" if monitor.driver_reset_detected else "MONITOR_ABORT_THRESHOLD", monitor.max_temp, monitor.max_power, monitor.any_throttle, stress_exit_codes)
        monitor.stop()

    return CandidateResult(True, "PASS", monitor.max_temp if monitor else None, monitor.max_power if monitor else None, monitor.any_throttle if monitor else None, stress_exit_codes)

def evaluate_candidate_confident(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    interrupted_event: threading.Event,
    manual_recovery_event: threading.Event,
    max_freq_mhz: int = 0,
    passes_required: int = 2,
    max_runs: int = 3,
    warmup: bool = True,
) -> CandidateResult:
    all_results: List[CandidateResult] = []
    hard_fail_prefixes = ("APPLY_FAILED", "GPU_DRIVER_RESET_DETECTED", "MONITOR_ABORT_THRESHOLD", "DOLOMING_", "GPUBURN_")

    def _is_hard_fail(reason: str) -> bool:
        return any(reason.startswith(p) for p in hard_fail_prefixes) or "CUDA_ERROR" in reason.upper()

    orig_stress = state.stress_seconds
    orig_multi = state.multi_stress_seconds
    try:
        if warmup:
            state.stress_seconds = max(8, min(20, orig_stress // 3))
            state.multi_stress_seconds = max(6, min(15, orig_multi // 2))
            w = evaluate_candidate(state, candidate_csv, f"{candidate_label}_warmup", interrupted_event, manual_recovery_event, max_freq_mhz)
            all_results.append(w)
            if not w.ok and _is_hard_fail(w.reason): return _merge_results(all_results, False, f"WARMUP_HARD_FAIL:{w.reason}")

        state.stress_seconds, state.multi_stress_seconds = orig_stress, orig_multi
        passes = 0
        fails = 0
        fail_reasons = []
        for i in range(1, max_runs + 1):
            r = evaluate_candidate(state, candidate_csv, f"{candidate_label}_run{i}", interrupted_event, manual_recovery_event, max_freq_mhz)
            all_results.append(r)
            if r.ok:
                passes += 1
                if passes >= passes_required: return _merge_results(all_results, True, f"PASS_{passes}OF{max_runs}")
            else:
                fails += 1
                fail_reasons.append(r.reason)
                if _is_hard_fail(r.reason) or fails > (max_runs - passes_required): return _merge_results(all_results, False, f"HARD_FAIL:{r.reason}" if _is_hard_fail(r.reason) else f"FAIL_{fails}OF{max_runs}:{fail_reasons[-1]}")
        return _merge_results(all_results, passes >= passes_required, f"PASS_{passes}OF{max_runs}" if passes >= passes_required else f"FAIL_{fails}OF{max_runs}:{fail_reasons[-1]}")
    finally:
        state.stress_seconds, state.multi_stress_seconds = orig_stress, orig_multi

def _merge_results(results: List[CandidateResult], ok: bool, reason: str) -> CandidateResult:
    temps = [r.telemetry_max_temp_c for r in results if r.telemetry_max_temp_c is not None]
    powers = [r.telemetry_max_power_w for r in results if r.telemetry_max_power_w is not None]
    throttles = [r.telemetry_any_throttle for r in results if r.telemetry_any_throttle is not None]
    codes = {}
    for r in results:
        if r.stress_exit_codes: codes.update(r.stress_exit_codes)
    return CandidateResult(ok, reason, max(temps) if temps else None, max(powers) if powers else None, any(throttles) if throttles else None, codes or None)

def revert_to_last_good(state: SessionState) -> None:
    nvapi_apply_curve(state.gpu, Path(state.last_good_curve_csv))

def _check_for_manual_recovery(state: SessionState, label: str, manual_recovery_event: threading.Event) -> None:
    if not manual_recovery_event.is_set(): return
    manual_recovery_event.clear()
    eprint(f"\nManual recovery requested ({label}) — reverting...")
    try:
        revert_to_last_good(state)
        eprint("Revert complete.")
    except Exception as ex:
        eprint(f"WARNING: revert failed: {ex}")
    raise KeyboardInterrupt("Manual recovery hotkey")

def run_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    last_good_csv = Path(state.last_good_curve_csv)

    print(f"Starting from step {state.current_step}/{state.max_steps} mode={state.mode}")
    while state.current_step < state.max_steps:
        _check_for_manual_recovery(state, "main tuning loop", manual_recovery_event)
        if interrupted_event.is_set():
            eprint("\nInterrupted — reverting...")
            try: revert_to_last_good(state)
            except Exception as ex: eprint(f"WARNING: revert failed: {ex}")
            raise KeyboardInterrupt("User pressed Ctrl+C")

        step = state.current_step + 1
        if state.mode == "uv": offset_mv, offset_mhz = -(state.step_mv * step), 0
        elif state.mode == "oc": offset_mv, offset_mhz = 0, state.step_mhz * step
        elif state.mode == "hybrid":
            phase = getattr(state, "hybrid_phase", "uv")
            if phase == "uv": offset_mv, offset_mhz = -(state.step_mv * step), 0
            else:
                offset_mv = getattr(state, "hybrid_locked_mv", 0)
                oc_step = step - getattr(state, "hybrid_oc_start_step", 0)
                offset_mhz = state.step_mhz * oc_step
        else: raise ValueError(f"Unknown mode: {state.mode}")

        label = f"step{step:03d}_mv{offset_mv}_mhz{offset_mhz}"
        candidate_csv = out_dir / "candidate.csv"
        points_candidate = apply_offsets_to_bin(stock_points, state.bin_min_mv, state.bin_max_mv, offset_mv, offset_mhz)
        write_curve_csv(candidate_csv, points_candidate)

        print(f"\n== Candidate {label} ==\n  Offset: {offset_mv} mV, {offset_mhz} MHz")
        result = evaluate_candidate(state, candidate_csv, label, interrupted_event, manual_recovery_event)
        print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")
        _check_for_manual_recovery(state, label, manual_recovery_event)

        steps_log = out_dir / "steps.jsonl"
        with steps_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**asdict(result), "utc": now_utc_iso(), "label": label, "step": step}) + "\n")

        if result.ok:
            shutil.copyfile(candidate_csv, last_good_csv)
            state.current_step, state.current_offset_mv, state.current_offset_mhz = step, offset_mv, offset_mhz
            save_session(state)
        else:
            revert_to_last_good(state)
            if state.mode == "hybrid" and state.hybrid_phase == "uv":
                state.hybrid_phase, state.hybrid_locked_mv, state.hybrid_oc_start_step = "oc", state.current_offset_mv, step
                state.current_step = step - 1
                save_session(state)
                continue
            state.current_step = step - 1
            save_session(state)
            break

def run_vlock_session(state: SessionState, interrupted_event: threading.Event, manual_recovery_event: threading.Event) -> None:
    # Ported from voltvandal.py with necessary adjustments
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))
    target_uv = mv_to_uv(state.vlock_target_mv)
    anchor_idx = min(range(len(stock_points)), key=lambda i: abs(stock_points[i].voltage_uv - target_uv))
    anchor_v_uv = stock_points[anchor_idx].voltage_uv
    anchor_stock_f_khz = stock_points[anchor_idx].freq_khz
    oc_base_freq_khz = state.vlock_oc_base_freq_khz or anchor_stock_f_khz

    print(f"\n=== VoltVandal — vlock mode ===\n  Anchor: {anchor_v_uv//1000} mV | Stock: {anchor_stock_f_khz//1000} MHz")
    
    # Phase 1: OC search
    if state.vlock_phase == "oc":
        while state.current_step <= state.max_steps:
            _check_for_manual_recovery(state, "vlock_p1", manual_recovery_event)
            step = state.current_step
            cand_freq = oc_base_freq_khz + mhz_to_khz(state.step_mhz * step)
            cand_pts = _build_vlock_curve(stock_points, anchor_idx, anchor_v_uv, cand_freq, 0)
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, cand_pts)
            
            label = f"vlock_p1_step{step:03d}_{cand_freq//1000}mhz"
            print(f"\n== {label} ==")
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event, max_freq_mhz=cand_freq//1000)
            print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")
            
            if result.ok:
                shutil.copyfile(cand_csv, Path(state.last_good_curve_csv))
                state.vlock_anchor_freq_khz = cand_freq
                state.current_step += 1
                save_session(state)
            else:
                revert_to_last_good(state)
                state.vlock_phase = "uv"
                state.vlock_uv_bin_idx = anchor_idx - 1
                state.current_step = 0
                save_session(state)
                break
    
    # Phase 2: UV/OC Shift Sweep
    if state.vlock_phase == "uv":
        oc_gain = state.vlock_anchor_freq_khz - anchor_stock_f_khz
        while state.vlock_uv_bin_idx >= 0:
            bin_idx = state.vlock_uv_bin_idx
            _check_for_manual_recovery(state, f"vlock_p2_bin{bin_idx}", manual_recovery_event)
            
            last_good_pts = load_curve_csv(Path(state.last_good_curve_csv))
            test_pts, save_pts = _build_vlock_phase2_curves(stock_points, last_good_pts, bin_idx, anchor_idx, anchor_v_uv, state.vlock_anchor_freq_khz, oc_gain)
            
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, test_pts)
            
            label = f"vlock_p2_bin{bin_idx:03d}_{stock_points[bin_idx].voltage_uv//1000}mv"
            print(f"\n== {label} ==")
            result = evaluate_candidate_confident(state, cand_csv, label, interrupted_event, manual_recovery_event)
            print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")
            
            if result.ok:
                write_curve_csv(Path(state.last_good_curve_csv), save_pts)
            else:
                revert_to_last_good(state)
                break
            
            state.vlock_uv_bin_idx -= 1
            save_session(state)
        
        state.vlock_phase = "done"
        save_session(state)
        print("\n=== vlock tuning complete ===")
