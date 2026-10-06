"""Hold a point lock while sampling utilization, clock and measured voltage."""

import threading
import time

from .hardware.point_lock import PointLockError, read_voltage_mv


def hold_and_sample(gpu: int, bus: int, seconds: int, stress_mode: str = "") -> None:
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
