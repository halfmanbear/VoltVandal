import shutil
import json
import signal
import time
import threading
from pathlib import Path

from .cli import parse_args
from .core import flightlog, faultmap
from .core.models import SessionState
from .core.utils import ensure_dir, warn_if_not_admin, now_utc_iso
from .core.session import session_paths, save_session, load_session
from .core.tuning import run_anchor_session, run_mvscan_session, run_session, run_vlock_session, revert_to_last_good
from .stress.runner import terminate_all_active_processes
from .hardware.nvapi import (
    dump_curve as nvapi_dump_curve,
    reset_curve_safe as nvapi_reset_curve_safe,
)
from .hardware.profiles import get_profile
from .hardware.point_lock import (
    PointLockError, inspect_point_lock, restore_point_lock, read_voltage_mv,
    temporary_point_lock,
    check_monitor_identity,
)
from .hardware.runtime_controls import (
    apply_fan_control,
    apply_gpu_throttle_temp,
    apply_power_limit_percent,
    apply_power_limit_max,
    read_gpu_target_temp,
    reset_gpu_throttle_temp,
    reset_power_limit_default,
)

interrupted = threading.Event()
manual_recovery = threading.Event()

def signal_handler(sig, frame):
    interrupted.set()
    print("\nCtrl+C received — stopping...")

def _anchor_cap_mhz(args, profile: dict, step_mhz: int, max_steps: int) -> int:
    """Anchor gain ceiling: explicit flag, else --aggressive range, else the profile's safe cap (0 = built-in default)."""
    if args.max_gain_mhz:
        return args.max_gain_mhz
    if args.aggressive:
        return step_mhz * max_steps
    return int(profile.get("safe_cap_mhz", 0))


def _resolve_arg(args, profile: dict, arg_name: str, profile_key: str, fallback):
    if hasattr(args, arg_name):
        return getattr(args, arg_name)
    if profile_key in profile:
        return profile[profile_key]
    return fallback


def _apply_pre_tune_controls(state: SessionState) -> None:
    if state.gpu_throttle_temp_c > 0 and state.gpu_throttle_temp_restore_c <= 0:
        prev_target = read_gpu_target_temp(state.gpu)
        if prev_target is not None:
            state.gpu_throttle_temp_restore_c = prev_target
            print(f"Captured pre-run GPU throttle temp target: {prev_target} C.")
    power_msg = (
        apply_power_limit_max(state.gpu) if state.power_limit_max
        else apply_power_limit_percent(state.gpu, state.power_limit_pct)
    )
    for msg in (
        power_msg,
        apply_gpu_throttle_temp(state.gpu, state.gpu_throttle_temp_c),
        apply_fan_control(state.gpu, state.fan_mode, state.fan_speed_pct),
    ):
        if msg:
            print(msg)


def _normalize_state_controls(state: SessionState) -> None:
    if state.fan_speed_pct > 0 and state.fan_mode != "manual":
        state.fan_mode = "manual"
        print("Fan speed requested; forcing fan mode to manual.")


def _resume_safety_reason(state: SessionState) -> str:
    if state.mode == "vlock" and getattr(state, "active_candidate_label", ""):
        return (
            f"Previous run ended during {state.active_candidate_label}; "
            "automatic retry is disabled."
        )
    if state.mode == "vlock" and state.vlock_phase == "failed":
        return "Vlock session stopped after a fatal failure; automatic retry is disabled."
    if state.mode == "vlock" and state.vlock_phase == "inconclusive":
        return "Vlock lower-bin coverage was inconclusive; automatic retry is disabled."
    return ""

def _parse_doloming_mode_arg(raw_mode: str) -> tuple[str, str]:
    """
    Parse --doloming-mode which may be a single value or comma-separated list.
    Returns (primary_mode, multi_modes_csv) where multi_modes_csv is empty when
    only one mode is supplied.
    """
    allowed = {"simple", "matrix", "ray", "frequency-max", "canary"}
    modes = [m.strip().lower() for m in str(raw_mode).split(",") if m.strip()]
    if not modes:
        raise ValueError("--doloming-mode must specify at least one mode")
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(
            f"--doloming-mode contains invalid mode(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )
    if len(modes) == 1:
        return modes[0], ""
    return modes[0], ",".join(modes)


def _run_tuning_with_hotkey(state: SessionState) -> None:
    interrupted.clear()
    manual_recovery.clear()
    unfinished = flightlog.start(state.out_dir)
    if unfinished:
        print(f"WARNING: the previous run ended without a clean exit (likely a crash/BSOD). Last flight-log entry: {unfinished}")
    crashed = faultmap.recover(state.out_dir)
    if crashed:
        print(f"Blocked crash point {crashed['voltage_uv'] / 1000:g} mV @ {crashed['freq_khz'] / 1000:g} MHz "
              f"({crashed['label']}); it and anything harsher will not be tried again.")
    try:
        if state.mode == "vlock":
            run_vlock_session(state, interrupted, manual_recovery)
        elif state.mode == "anchor":
            run_anchor_session(state, interrupted, manual_recovery)
        elif state.mode == "mvscan":
            run_mvscan_session(state, interrupted, manual_recovery)
        else:
            run_session(state, interrupted, manual_recovery)
    except KeyboardInterrupt as ex:
        _msg = str(ex).lower()
        _is_ctrl_c = interrupted.is_set() or "ctrl+c" in _msg or "ctrl-c" in _msg
        if _is_ctrl_c:
            print("\nInterrupted by user (Ctrl+C).")
            _reset_board_to_factory_defaults(state)
            if state.mode == "vlock" and getattr(state, "active_candidate_label", ""):
                state.active_candidate_label = ""
                save_session(state)
            print("Exiting cleanly.")
        else:
            print("\nManual recovery requested. Attempting safe revert...")
            try:
                revert_to_last_good(state)
                print("Revert complete. Exiting cleanly.")
            except Exception as ex:
                print(f"WARNING: Revert failed during interrupt handling: {ex}")
    except PointLockError:
        # Do not shrink the search boundary for an unexercised/unsupported point.
        # A failed lock restore retains its journal; leave recovery explicit.
        if not (Path(state.out_dir) / "point_lock_recovery.json").exists():
            revert_to_last_good(state)
        raise
    except Exception as ex:
        print(f"\nFatal error during tuning: {ex}")
        _reset_board_to_factory_defaults(state)
        print("Exited after emergency factory reset.")
    finally:
        flightlog.log("session_end")
        _killed = terminate_all_active_processes()
        if _killed > 0:
            print(f"Terminated {_killed} lingering stress process(es).")


def _reset_board_to_factory_defaults(state: SessionState) -> None:
    print("Resetting board controls to factory defaults...")
    failures = []
    try:
        nvapi_reset_curve_safe(state.gpu, timeout_seconds=12.0)
        print("VF curve reset to default.")
    except Exception as ex:
        failures.append(f"VF curve reset failed: {ex}")

    try:
        print(reset_power_limit_default(state.gpu))
    except Exception as ex:
        failures.append(f"Power-limit reset failed: {ex}")

    if state.gpu_throttle_temp_c > 0:
        try:
            print(
                reset_gpu_throttle_temp(
                    state.gpu,
                    restore_temp_c=state.gpu_throttle_temp_restore_c or None,
                )
            )
        except Exception as ex:
            failures.append(f"Throttle-temp reset failed: {ex}")

    try:
        msg = apply_fan_control(state.gpu, "auto", 0)
        if msg:
            print(msg)
    except Exception as ex:
        failures.append(f"Fan-mode reset failed: {ex}")

    if failures:
        print("Factory reset completed with warnings:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("Factory reset complete.")

def _hold_and_sample(gpu: int, bus: int, seconds: int, stress_mode: str = "") -> None:
    import pynvml
    from .stress.runner import (
        _build_official_stress_cmd, _reader_thread, start_process, terminate_process_tree,
    )
    proc = None
    out_lines: list[str] = []
    if stress_mode:
        proc = start_process(_build_official_stress_cmd(gpu, stress_mode, seconds + 120))
        # Drain stdout, otherwise a full pipe blocks the stress process.
        threading.Thread(target=_reader_thread, args=(proc.stdout, out_lines, threading.Event()),
                         daemon=True).start()
        print(f"Started '{stress_mode}' stress; waiting up to 90s for it to calibrate to >= 50% util...")
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            waited = 0
            while waited < 90 and proc.poll() is None:
                if pynvml.nvmlDeviceGetUtilizationRates(handle).gpu >= 50:
                    break
                time.sleep(1.0)
                waited += 1
            print(f"Load ready after {waited}s." if waited < 90 else "Load never reached 50% util.")
        finally:
            pynvml.nvmlShutdown()
    else:
        print(f"Holding lock for {seconds}s; start your GPU load now.")
    print("Sampling once per second...")
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        rows = []
        for _ in range(seconds):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            try:
                mv = read_voltage_mv(gpu, bus)
            except PointLockError:
                mv = None
            rows.append((util, clock, mv))
            print(f"  util {util:3d}% | clock {clock} MHz | voltage {mv if mv is None else f'{mv:.1f}'} mV")
            time.sleep(1.0)
    finally:
        pynvml.nvmlShutdown()
        if proc is not None:
            exited = proc.poll()
            terminate_process_tree(proc)
            tail = "".join(out_lines[-40:]).replace(chr(13), chr(10)).strip().splitlines()[-4:]
            print(f"Stress process exit code at end: {exited}; last output:")
            for line in tail:
                print(f"  | {line[:160]}")
    loaded = [r for r in rows if r[0] >= 50 and r[2] is not None]
    print(f"Samples: {len(rows)} total, {len(loaded)} loaded (util >= 50%).")
    if loaded:
        volts = [r[2] for r in loaded]
        clocks = [r[1] for r in loaded]
        print(f"Loaded voltage: min {min(volts):.1f} / max {max(volts):.1f} mV | "
              f"clock: min {min(clocks)} / max {max(clocks)} MHz")

def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    try:
        args = parse_args()
        if args.command in ("run", "resume"):
            journal = Path(args.out) / "point_lock_recovery.json"
            if journal.exists():
                raise PointLockError(f"Recovery pending. Run recover-point-lock --out {args.out}")
        if args.command in ("dump", "run", "resume", "recover-point-lock"):
            warn_if_not_admin()

        if args.command == "point-lock-info":
            info = inspect_point_lock(args.gpu)
            info.pop("snapshot", None)
            info["uuid"] = check_monitor_identity(args.gpu, info["bus"])
            try:
                info["observed_voltage_mv"] = read_voltage_mv(args.gpu, info["bus"])
            except PointLockError as ex:
                info["voltage_read_error"] = str(ex)
            print(json.dumps(info, indent=2))
            print("Getter succeeded and setter resolved; writes and loaded operation are untested.")
            return 0
        if args.command == "point-lock-test":
            Path(args.out).mkdir(parents=True, exist_ok=True)
            with temporary_point_lock(args.gpu, round(args.voltage_mv * 1000),
                                      Path(args.out) / "point_lock_recovery.json") as bus:
                print(f"Setter accepted and readback matched {args.voltage_mv} mV.")
                try:
                    print(f"Observed voltage while locked (idle): {read_voltage_mv(args.gpu, bus):.1f} mV")
                except PointLockError as ex:
                    print(f"Voltage read failed: {ex}")
                hold = args.hold_seconds or (30 if args.stress_mode else 0)
                if hold > 0:
                    _hold_and_sample(args.gpu, bus, hold, args.stress_mode)
            print("Previous lock state restored and verified.")
            return 0
        if args.command == "recover-point-lock":
            restore_point_lock(Path(args.out) / "point_lock_recovery.json")
            print("Previous point-lock state restored and verified.")
            return 0
        
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
            stock_csv, last_good_csv, checkpoint = session_paths(out_dir)
            if any(path.exists() for path in (stock_csv, last_good_csv, checkpoint)):
                raise ValueError(
                    f"{out_dir} already contains a tuning session or baseline; "
                    "choose a new --out directory for a fresh run"
                )
            ensure_dir(out_dir)
            dolo_mode, dolo_modes_csv = _parse_doloming_mode_arg(args.doloming_mode)
            
            # Profile application
            profile = get_profile(args.gpu_profile) if args.gpu_profile else {}
            
            state = SessionState(
                gpu=args.gpu,
                out_dir=str(out_dir),
                stock_curve_csv=str(stock_csv),
                last_good_curve_csv=str(last_good_csv),
                checkpoint_json=str(checkpoint),
                mode=args.mode,
                bin_min_mv=_resolve_arg(args, profile, "bin_min_mv", "bin_min_mv", 850),
                bin_max_mv=_resolve_arg(args, profile, "bin_max_mv", "bin_max_mv", 1050),
                step_mv=_resolve_arg(args, profile, "step_mv", "step_mv", 5),
                step_mhz=_resolve_arg(args, profile, "step_mhz", "step_mhz", 15),
                max_steps=_resolve_arg(args, profile, "max_steps", "max_steps", 30),
                stress_seconds=_resolve_arg(args, profile, "stress_seconds", "stress_seconds", 60),
                doloming="auto",
                doloming_mode=dolo_mode,
                doloming_modes=dolo_modes_csv,
                multi_stress_seconds=_resolve_arg(args, profile, "multi_stress_seconds", "multi_stress_seconds", 30),
                gpuburn=getattr(args, "gpuburn", None),
                poll_seconds=args.poll_seconds,
                temp_limit_c=_resolve_arg(args, profile, "temp_limit_c", "temp_limit_c", 83),
                hotspot_limit_c=_resolve_arg(args, profile, "hotspot_limit_c", "hotspot_limit_c", 95),
                hotspot_offset_c=15,
                power_limit_w=_resolve_arg(args, profile, "power_limit_w", "power_limit_w", 400.0),
                abort_on_throttle=args.abort_on_throttle,
                stress_timeout=args.stress_timeout,
                vlock_target_mv=_resolve_arg(args, profile, "target_voltage_mv", "target_voltage_mv", 950),
                vlock_start_freq_mhz=args.vlock_start_freq_mhz,
                point_lock=args.point_lock,
                anchor_max_gain_mhz=_anchor_cap_mhz(args, profile, _resolve_arg(args, profile, "step_mhz", "step_mhz", 15),
                                                    _resolve_arg(args, profile, "max_steps", "max_steps", 30)),
                anchor_finish=args.anchor_finish,
                point_util_pct=args.point_util_pct,
                auto_plan=args.auto_plan,
                mvscan_objective=args.mvscan_objective,
                power_limit_pct=args.power_limit_pct,
                power_limit_max=args.power_limit_max,
                gpu_throttle_temp_c=args.gpu_throttle_temp_c,
                fan_mode=args.fan_mode,
                fan_speed_pct=args.fan_speed_pct,
                live_display=args.live_display,
                started_utc=now_utc_iso()
            )
            _normalize_state_controls(state)

            if state.vlock_start_freq_mhz < 0:
                raise ValueError("--vlock-start-freq-mhz must be >= 0")
            if state.power_limit_pct <= 0:
                raise ValueError("--power-limit-pct must be > 0")
            if state.gpu_throttle_temp_c < 0:
                raise ValueError("--gpu-throttle-temp-c must be >= 0")
            if state.fan_speed_pct < 0 or state.fan_speed_pct > 100:
                raise ValueError("--fan-speed-pct must be in range 0-100")
            if state.fan_mode == "manual" and state.fan_speed_pct == 0:
                raise ValueError("--fan-speed-pct must be > 0 when --fan-mode=manual")
            
            if state.mode == "anchor":
                # Gains are relative to the baseline, so it must be factory, not a leftover tune.
                nvapi_reset_curve_safe(state.gpu, timeout_seconds=12.0)
                print("VF curve reset to factory default before taking the baseline.")
            if not stock_csv.exists():
                print(f"Baseline dump (gpu={state.gpu}) -> {stock_csv} ...")
                nvapi_dump_curve(state.gpu, stock_csv)
                shutil.copyfile(stock_csv, last_good_csv)
            
            _apply_pre_tune_controls(state)
            save_session(state)
            _run_tuning_with_hotkey(state)
                
        elif args.command == "resume":
            out_dir = Path(args.out)
            state = load_session(out_dir)
            reason = _resume_safety_reason(state)
            if reason:
                print(reason)
                _reset_board_to_factory_defaults(state)
                return 1
            if args.anchor_finish:
                state.anchor_finish = True
            if args.max_gain_mhz:
                state.anchor_max_gain_mhz = args.max_gain_mhz
            _normalize_state_controls(state)
            _apply_pre_tune_controls(state)
            _run_tuning_with_hotkey(state)
    except PointLockError as ex:
        print(f"Point test stopped: {ex}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
        return 130

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
