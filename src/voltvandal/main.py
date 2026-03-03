import shutil
import signal
import sys
import threading
from pathlib import Path

from .cli import parse_args
from .core.models import SessionState
from .core.utils import ensure_dir, warn_if_not_admin, now_utc_iso
from .core.session import session_paths, save_session, load_session
from .core.tuning import run_session, run_vlock_session
from .hardware.nvapi import dump_curve as nvapi_dump_curve, apply_curve as nvapi_apply_curve
from .hardware.profiles import get_profile

interrupted = threading.Event()
manual_recovery = threading.Event()

def signal_handler(sig, frame):
    interrupted.set()
    print("\nCtrl+C received — stopping...")

def main():
    warn_if_not_admin()
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    
    if args.command == "dump":
        out_dir = Path(args.out)
        ensure_dir(out_dir)
        stock_csv, last_good_csv, checkpoint = session_paths(out_dir)
        print(f"Dumping curve to {stock_csv}...")
        nvapi_dump_curve(args.gpu, stock_csv)
        if not last_good_csv.exists():
            shutil.copyfile(stock_csv, last_good_csv)
        print("Done.")
        
    elif args.command == "run":
        out_dir = Path(args.out)
        ensure_dir(out_dir)
        stock_csv, last_good_csv, checkpoint = session_paths(out_dir)
        
        # Profile application
        profile = get_profile(args.gpu_profile) if args.gpu_profile else {}
        
        state = SessionState(
            gpu=args.gpu,
            out_dir=str(out_dir),
            stock_curve_csv=str(stock_csv),
            last_good_curve_csv=str(last_good_csv),
            checkpoint_json=str(checkpoint),
            mode=args.mode,
            bin_min_mv=args.bin_min_mv if hasattr(args, 'bin_min_mv') else profile.get('bin_min_mv', 850),
            bin_max_mv=args.bin_max_mv if hasattr(args, 'bin_max_mv') else profile.get('bin_max_mv', 1050),
            step_mv=args.step_mv,
            step_mhz=args.step_mhz,
            max_steps=args.max_steps,
            stress_seconds=args.stress_seconds,
            doloming="integrated",
            doloming_mode="simple",
            doloming_modes=args.doloming_modes or "",
            multi_stress_seconds=args.multi_stress_seconds,
            gpuburn=None,
            poll_seconds=1.0,
            temp_limit_c=args.temp_limit_c,
            hotspot_limit_c=args.hotspot_limit_c or profile.get('hotspot_limit_c', 95),
            hotspot_offset_c=15,
            power_limit_w=args.power_limit_w,
            abort_on_throttle=True,
            vlock_target_mv=args.target_voltage_mv or profile.get('target_voltage_mv', 950),
            started_utc=now_utc_iso()
        )
        
        if not stock_csv.exists():
            print(f"Baseline dump (gpu={state.gpu}) -> {stock_csv} ...")
            nvapi_dump_curve(state.gpu, stock_csv)
            shutil.copyfile(stock_csv, last_good_csv)
        
        save_session(state)
        
        if state.mode == "vlock":
            run_vlock_session(state, interrupted, manual_recovery)
        else:
            run_session(state, interrupted, manual_recovery)
            
    elif args.command == "resume":
        out_dir = Path(args.out)
        state = load_session(out_dir)
        if state.mode == "vlock":
            run_vlock_session(state, interrupted, manual_recovery)
        else:
            run_session(state, interrupted, manual_recovery)

if __name__ == "__main__":
    main()
