import argparse
import sys
from .hardware.profiles import list_profiles


class _VVHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    pass


def _format_flag_summary(title: str, parser: argparse.ArgumentParser) -> str:
    lines = [title]
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.help is argparse.SUPPRESS:
            continue
        opts = ", ".join(action.option_strings)
        help_text = (action.help or "").strip().replace("%%", "%")
        lines.append(f"  {opts:<42} {help_text}")
    return "\n".join(lines)


def create_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="voltvandal",
        description="VoltVandal — Professional GPU Undervolting & Overclocking CLI",
        formatter_class=_VVHelpFormatter,
    )
    
    sub = ap.add_subparsers(dest="command", help="Command to run")

    # Dump command
    p_dump = sub.add_parser("dump", help="Dump current VF curve to CSV")
    p_dump.add_argument("--gpu", type=int, default=0, help="GPU index")
    p_dump.add_argument("--out", type=str, default="artifacts", help="Output directory")

    # Run command
    p_run = sub.add_parser("run", help="Start a new tuning session")
    p_run.add_argument("--gpu", type=int, default=0, help="GPU index")
    p_run.add_argument("--out", type=str, default="artifacts", help="Output directory")
    p_run.add_argument("--mode", choices=["uv", "oc", "hybrid", "vlock", "mvscan"], required=True, help="Tuning mode: uv, oc, hybrid, vlock, or mvscan")
    p_run.add_argument("--gpu-profile", type=str, help="Use a reference profile (e.g. rtx30, rtx40)")
    
    # Tuning params
    p_run.add_argument("--target-voltage-mv", type=int, default=argparse.SUPPRESS, help="Target voltage for vlock mode (profile default if omitted)")
    p_run.add_argument("--step-mv", type=int, default=argparse.SUPPRESS, help="Voltage step size (profile default if omitted)")
    p_run.add_argument("--step-mhz", type=int, default=argparse.SUPPRESS, help="Frequency step size (profile default if omitted)")
    p_run.add_argument("--max-steps", type=int, default=argparse.SUPPRESS, help="Maximum number of steps (profile default if omitted)")
    p_run.add_argument("--vlock-start-freq-mhz", type=int, default=0, help="Phase 1 start frequency for vlock search (0 = start at base/stock)")
    p_run.add_argument("--mvscan-objective", choices=["balanced", "max-clock", "min-cap"], default="balanced", help="Objective used by mvscan to rank stable voltage caps")
    
    # Stress params
    p_run.add_argument("--stress-seconds", type=int, default=argparse.SUPPRESS, help="Stress duration per step (profile default if omitted)")
    p_run.add_argument("--doloming-mode", type=str, default="simple", help="Stress mode(s): single mode or comma-separated list (simple,matrix,ray,frequency-max)")
    p_run.add_argument("--multi-stress-seconds", type=int, default=argparse.SUPPRESS, help="Duration per mode in multi-stress (profile default if omitted)")
    p_run.add_argument("--gpuburn", type=str, help="Path to gpu-burn executable (optional)")
    p_run.add_argument("--stress-timeout", type=int, help="Hard timeout per stress run in seconds")
    
    # Limits
    p_run.add_argument("--bin-min-mv", type=int, default=argparse.SUPPRESS, help="Voltage bin sweep minimum (profile default if omitted)")
    p_run.add_argument("--bin-max-mv", type=int, default=argparse.SUPPRESS, help="Voltage bin sweep maximum (profile default if omitted)")
    p_run.add_argument("--temp-limit-c", type=int, default=argparse.SUPPRESS, help="GPU edge temp limit (profile default if omitted)")
    p_run.add_argument("--hotspot-limit-c", type=int, default=argparse.SUPPRESS, help="Hotspot temp limit (profile default if omitted)")
    p_run.add_argument("--power-limit-w", type=float, default=argparse.SUPPRESS, help="Power limit (W, profile default if omitted)")

    # Monitor / behavior
    p_run.add_argument("--poll-seconds", type=float, default=1.0, help="Telemetry poll interval")
    p_run.add_argument("--live-display", dest="live_display", action="store_true", help="Enable live telemetry display (default)")
    p_run.add_argument("--no-live-display", dest="live_display", action="store_false", help="Disable live telemetry display")
    p_run.set_defaults(live_display=True)
    p_run.add_argument("--abort-on-throttle", dest="abort_on_throttle", action="store_true", help="Abort when actionable throttle reasons are detected")
    p_run.add_argument("--allow-throttle", dest="abort_on_throttle", action="store_false", help="Do not abort on throttle reasons (default)")
    p_run.set_defaults(abort_on_throttle=False)

    # Power / fan controls/state
    p_run.add_argument("--power-limit-pct", type=int, default=100, help="Power limit as %% of GPU default TDP (100 = unchanged)")
    p_run.add_argument("--gpu-throttle-temp-c", type=int, default=0, help="GPU throttle temp target via nvidia-smi -gtt (0 = unchanged)")
    p_run.add_argument("--fan-mode", choices=["auto", "manual"], default="auto", help="Fan control mode (manual uses NVML APIs where supported)")
    p_run.add_argument("--fan-speed-pct", type=int, default=0, help="Fan speed percent; non-zero automatically forces --fan-mode=manual")

    # Resume command
    p_res = sub.add_parser("resume", help="Resume a tuning session from checkpoint")
    p_res.add_argument("--out", type=str, default="artifacts", help="Output directory")

    # Profiles command
    p_prof = sub.add_parser("profiles", help="List available GPU profiles")

    ap.epilog = "\n\n".join(
        [
            _format_flag_summary("Dump Flags:", p_dump),
            _format_flag_summary("Run Flags:", p_run),
            _format_flag_summary("Resume Flags:", p_res),
            "Safe Example (conservative baseline for most GPUs):\n"
            "  python voltvandal.py run --mode uv --gpu 0 --out artifacts "
            "--step-mv 5 --max-steps 4 --stress-seconds 30 "
            "--power-limit-pct 85 --gpu-throttle-temp-c 80 --fan-speed-pct 50",
            "Tip: for full run-specific help, use:\n"
            "  python voltvandal.py run -h",
        ]
    )

    return ap

def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command == "profiles":
        list_profiles()
        sys.exit(0)
        
    if args.command is None:
        parser.print_help()
        sys.exit(0)
        
    return args
