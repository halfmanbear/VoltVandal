import json
import shutil
import threading
from dataclasses import asdict
from pathlib import Path

from ..models import SessionState
from ..utils import eprint, now_utc_iso, ensure_dir
from ..curve import load_curve_csv, write_curve_csv, apply_offsets_to_bin
from ..session import save_session
from .evaluate import evaluate_candidate
from .recovery import revert_to_last_good, _check_for_manual_recovery


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
        if result.ok:
            print("Result: PASS")
        else:
            print(f"Result: FAIL | {result.reason}")
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
