#!/usr/bin/env python3
"""
VoltVandal integrated GPU stress runner.

This is a lightweight in-repo replacement for the external doloMing wrapper
flow. It keeps mode names and summary text compatible with VoltVandal's parser.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pynvml

try:
    import cupy as cp  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: cupy is required for integrated stress tests. "
        "Install with: python -m pip install cupy-cuda12x"
    ) from exc


@dataclass
class Metrics:
    util_avg: float = 0.0
    util_max: float = 0.0
    freq_avg: float = 0.0
    freq_max: float = 0.0
    temp_max: float = 0.0
    power_max: float = 0.0
    samples: int = 0


def _nvml_handle(gpu_index: int):
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(gpu_index)


def _read_metrics(handle) -> Dict[str, float]:
    util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    freq = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
    try:
        power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
    except pynvml.NVMLError:
        power = 0.0
    return {"util": util, "freq": freq, "temp": temp, "power": power}


def _accumulate(m: Metrics, sample: Dict[str, float]) -> None:
    m.samples += 1
    m.util_avg += sample["util"]
    m.freq_avg += sample["freq"]
    m.util_max = max(m.util_max, sample["util"])
    m.freq_max = max(m.freq_max, sample["freq"])
    m.temp_max = max(m.temp_max, sample["temp"])
    m.power_max = max(m.power_max, sample["power"])


def _finalize(m: Metrics) -> Metrics:
    if m.samples > 0:
        m.util_avg /= m.samples
        m.freq_avg /= m.samples
    return m


def _print_status(mode: str, remaining: int, sample: Dict[str, float]) -> None:
    print(
        f"[{mode}] t-{remaining:3d}s | util={sample['util']:5.1f}% "
        f"freq={sample['freq']:6.0f}MHz temp={sample['temp']:5.1f}C "
        f"power={sample['power']:6.1f}W"
    )


def _work_matrix(arr_a, arr_b, out, loops: int = 3) -> None:
    for _ in range(loops):
        cp.matmul(arr_a, arr_b, out=out)


def _work_simple(vec_a, vec_b, loops: int = 24) -> None:
    x = vec_a
    y = vec_b
    for _ in range(loops):
        x = cp.tanh(x * 1.0003 + y * 0.9997) + cp.sqrt(cp.abs(y) + 1e-3)
        y = cp.sin(y + x * 0.017) + cp.cos(x * 0.013)
    _ = cp.mean(x + y)


def _work_ray(origins, dirs, spheres_center, spheres_radius, loops: int = 2) -> None:
    o = origins
    d = dirs
    for _ in range(loops):
        oc = o[:, None, :] - spheres_center[None, :, :]
        b = 2.0 * cp.sum(oc * d[:, None, :], axis=2)
        c = cp.sum(oc * oc, axis=2) - spheres_radius[None, :] ** 2
        disc = b * b - 4.0 * c
        hit = disc > 0
        _ = cp.count_nonzero(hit)


def _run_mode(
    mode: str,
    seconds: int,
    gpu_index: int,
    max_freq_mhz: int = 0,
    target_percent: float = 50.0,
) -> Metrics:
    handle = _nvml_handle(gpu_index)
    m = Metrics()

    with cp.cuda.Device(gpu_index):
        cp.random.seed(1337)
        a = cp.random.random((2048, 2048), dtype=cp.float32)
        b = cp.random.random((2048, 2048), dtype=cp.float32)
        out = cp.zeros((2048, 2048), dtype=cp.float32)
        v1 = cp.random.random((8_000_000,), dtype=cp.float32)
        v2 = cp.random.random((8_000_000,), dtype=cp.float32)

        n_rays = 400_000
        n_spheres = 64
        origins = cp.random.random((n_rays, 3), dtype=cp.float32)
        dirs = cp.random.random((n_rays, 3), dtype=cp.float32) - 0.5
        dirs /= cp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6
        spheres_center = cp.random.random((n_spheres, 3), dtype=cp.float32) * 2.0 - 1.0
        spheres_radius = cp.random.random((n_spheres,), dtype=cp.float32) * 0.4 + 0.1

        end_t = time.time() + max(1, int(seconds))
        next_log = 0.0
        while time.time() < end_t:
            if mode == "matrix":
                _work_matrix(a, b, out, loops=2)
            elif mode == "ray":
                _work_ray(origins, dirs, spheres_center, spheres_radius, loops=1)
            elif mode == "frequency-max":
                # Heavier mixed workload to hold sustained high clocks.
                _work_matrix(a, b, out, loops=3)
                _work_simple(v1, v2, loops=20)
            else:
                _work_simple(v1, v2, loops=18)

            cp.cuda.runtime.deviceSynchronize()
            sample = _read_metrics(handle)
            _accumulate(m, sample)

            # For util-targeted modes, lightly modulate pacing to stay near target.
            if mode in ("ray", "matrix", "simple"):
                err = sample["util"] - target_percent
                if err > 10:
                    time.sleep(0.02)
                elif err < -10:
                    time.sleep(0.002)

            if time.time() >= next_log:
                remaining = int(max(0.0, end_t - time.time()))
                _print_status(mode, remaining, sample)
                next_log = time.time() + 1.0

        if mode == "frequency-max" and max_freq_mhz > 0:
            # Keep key naming compatible with prior parser/reporting expectations.
            score_gamma = 100.0 * (m.freq_avg / float(max_freq_mhz))
            score_gamma = max(0.0, min(120.0, score_gamma))
            print(f"Score_gamma : {score_gamma:.2f}")
            print(f"Avg/Max ratio : {m.freq_avg / float(max_freq_mhz):.4f}")

    return _finalize(m)


def main() -> int:
    ap = argparse.ArgumentParser(description="VoltVandal integrated GPU stress runner")
    ap.add_argument("--mode", choices=["matrix", "simple", "ray", "frequency-max"], default="simple")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--target-percent", type=float, default=50.0)
    ap.add_argument("--max-freq-mhz", type=int, default=0)
    args = ap.parse_args()

    print(f"Starting integrated stress mode={args.mode} gpu={args.gpu} duration={args.seconds}s")
    print(f"Target utilization: {args.target_percent:.1f}%")

    failure: Optional[str] = None
    result: Optional[Metrics] = None
    try:
        result = _run_mode(
            mode=args.mode,
            seconds=args.seconds,
            gpu_index=args.gpu,
            max_freq_mhz=args.max_freq_mhz,
            target_percent=args.target_percent,
        )
    except Exception as exc:
        failure = str(exc)
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    print("\nTest Summary:")
    if failure is not None:
        print(f"Error during test: {failure}")
        return 1

    assert result is not None
    util_std = 0.0  # lightweight runner does not keep full history
    status = "Successfully maintain" if result.util_avg >= 35.0 else "Unstable"
    print(
        f"Status              : {status}\n"
        f"Average Utilization : {result.util_avg:.2f}%\n"
        f"Max Utilization     : {result.util_max:.2f}%\n"
        f"Average Frequency   : {result.freq_avg:.2f} MHz\n"
        f"Max Frequency       : {result.freq_max:.2f} MHz\n"
        f"Max Temperature     : {result.temp_max:.2f} C\n"
        f"Max Power           : {result.power_max:.2f} W\n"
        f"Util Stddev         : {util_std:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
