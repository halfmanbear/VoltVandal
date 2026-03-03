import argparse
import sys
from pathlib import Path
from .hardware.profiles import list_profiles, get_profile

def create_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="voltvandal",
        description="VoltVandal — Professional GPU Undervolting & Overclocking CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    p_run.add_argument("--mode", choices=["uv", "oc", "hybrid", "vlock"], required=True)
    p_run.add_argument("--gpu-profile", type=str, help="Use a reference profile (e.g. rtx30, rtx40)")
    
    # Tuning params
    p_run.add_argument("--target-voltage-mv", type=int, help="Target voltage for vlock mode")
    p_run.add_argument("--step-mv", type=int, default=5, help="Voltage step size")
    p_run.add_argument("--step-mhz", type=int, default=15, help="Frequency step size")
    p_run.add_argument("--max-steps", type=int, default=30, help="Maximum number of steps")
    
    # Stress params
    p_run.add_argument("--stress-seconds", type=int, default=60, help="Stress duration per step")
    p_run.add_argument("--doloming-modes", type=str, help="Comma-separated stress modes (ray,matrix,frequency-max)")
    p_run.add_argument("--multi-stress-seconds", type=int, default=30, help="Duration per mode in multi-stress")
    
    # Limits
    p_run.add_argument("--bin-min-mv", type=int, help="Voltage bin sweep minimum")
    p_run.add_argument("--bin-max-mv", type=int, help="Voltage bin sweep maximum")
    p_run.add_argument("--temp-limit-c", type=int, default=83, help="GPU temp limit")
    p_run.add_argument("--hotspot-limit-c", type=int, help="Hotspot temp limit")
    p_run.add_argument("--power-limit-w", type=float, default=400, help="Power limit (W)")

    # Resume command
    p_res = sub.add_parser("resume", help="Resume a tuning session from checkpoint")
    p_res.add_argument("--out", type=str, default="artifacts", help="Output directory")

    # Profiles command
    p_prof = sub.add_parser("profiles", help="List available GPU profiles")

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
