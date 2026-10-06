#!/usr/bin/env python3
"""VoltVandal canary: a stress workload that checks its own results.

The other modes only fail when the GPU crashes or hangs. This one runs deterministic kernels and
verifies the answers, so a point that is about to become unstable shows up as a clean data error
(`CANARY_DATA_ERROR`) before it becomes a hang. Design rules:

* every kernel finishes in tens of milliseconds, far below the 2 s Windows TDR timeout;
* each check is a known-answer test: integer kernels are compared to a host reference, and every
  kernel is run twice and compared bit for bit;
* the load ramps up over the first seconds instead of starting at full power;
* the first mismatch ends the run at once (fail fast), exit code 1, `Error during test:` line.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Optional

import numpy as np

KERNEL_SRC = r"""
extern "C" __global__ void int_chain(const unsigned* in, unsigned* out, int n, int iters) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = in[i];
    for (int k = 0; k < iters; ++k) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        x = x * 1664525u + 1013904223u + (unsigned)k;
    }
    out[i] = x;
}
extern "C" __global__ void fma_chain(const float* in, float* out, int n, int iters) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float a = in[i];
    float b = in[(i + 1) % n];
    for (int k = 0; k < iters; ++k) {
        a = fmaf(a, 1.0000001f, b * 0.5f);
        b = fmaf(b, 0.9999999f, a * 0.25f);
        if (fabsf(a) > 1.0e6f) a *= 1.0e-6f;
        if (fabsf(b) > 1.0e6f) b *= 1.0e-6f;
    }
    out[i] = a + b;
}
extern "C" __global__ void fill_pattern(unsigned* buf, size_t n, unsigned seed) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = (unsigned)(i * 2654435761u) ^ seed;
}
extern "C" __global__ void check_pattern(const unsigned* buf, size_t n, unsigned seed, unsigned* bad) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    if (buf[i] != ((unsigned)(i * 2654435761u) ^ seed)) atomicAdd(bad, 1u);
}
extern "C" __global__ void cmp_words(const unsigned* a, const unsigned* b, size_t n, unsigned* bad) {
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i]) atomicAdd(bad, 1u);
}
"""

THREADS = 256
TARGET_KERNEL_S = 0.04      # per-launch duration the iteration count is tuned towards
MAX_KERNEL_S = 0.25         # hard ceiling, still far below the 2 s TDR timeout
RAMP_SECONDS = 10.0
REF_WORDS = 1024            # slice of the integer kernel checked against the host reference


def int_chain_reference(values: np.ndarray, iters: int) -> np.ndarray:
    """Host implementation of the int_chain kernel (uint32 arithmetic wraps like the GPU's)."""
    x = values.astype(np.uint32).copy()
    with np.errstate(over="ignore"):
        for k in range(iters):
            x ^= x << np.uint32(13)
            x ^= x >> np.uint32(17)
            x ^= x << np.uint32(5)
            x = x * np.uint32(1664525) + np.uint32(1013904223) + np.uint32(k)
    return x


def ramp_duty(elapsed_s: float, target_fraction: float, ramp_s: float = RAMP_SECONDS) -> float:
    """Fraction of time the GPU is kept busy: ramps from 25% of target to target over ramp_s."""
    target = min(1.0, max(0.05, target_fraction))
    if ramp_s <= 0 or elapsed_s >= ramp_s:
        return target
    return target * (0.25 + 0.75 * elapsed_s / ramp_s)


class CanaryDataError(RuntimeError):
    pass


def _blocks(n: int) -> int:
    return (n + THREADS - 1) // THREADS


def run(gpu: int, seconds: int, target_percent: float, seed: int = 1337) -> Dict[str, float]:
    import cupy as cp
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    stats = {"util_sum": 0.0, "util_max": 0.0, "freq_sum": 0.0, "freq_max": 0.0, "temp_max": 0.0, "power_max": 0.0, "n": 0}
    target_fraction = 1.0 if target_percent <= 0 or target_percent >= 95 else target_percent / 100.0

    with cp.cuda.Device(gpu):
        module = cp.RawModule(code=KERNEL_SRC)
        k_int, k_fma = module.get_function("int_chain"), module.get_function("fma_chain")
        k_fill, k_check = module.get_function("fill_pattern"), module.get_function("check_pattern")
        k_cmp = module.get_function("cmp_words")

        rng = np.random.default_rng(seed)
        n = 1 << 22
        host_in = rng.integers(0, 2**32, size=n, dtype=np.uint32)
        d_in = cp.asarray(host_in)
        d_f = cp.asarray(rng.random(n, dtype=np.float32) + 0.5)
        outs = {name: [cp.zeros(n, dtype=dtype) for _ in range(2)]
                for name, dtype in (("int_chain", cp.uint32), ("fma_chain", cp.float32))}
        mem_words = 1 << 26  # 256 MB pattern buffer
        d_mem = cp.zeros(mem_words, dtype=cp.uint32)
        d_bad = cp.zeros(1, dtype=cp.uint32)
        mat_a = cp.asarray(rng.random((1024, 1024), dtype=np.float32)).astype(cp.float16)
        mat_b = cp.asarray(rng.random((1024, 1024), dtype=np.float32)).astype(cp.float16)

        iters = {"int_chain": 64, "fma_chain": 64}
        start = time.time()
        end_t = start + max(1, int(seconds))
        next_log = 0.0
        loop = 0
        mem_seed = seed

        bad = cp.zeros(4, dtype=cp.uint32)   # int, fma, fp16 matmul, memory pattern mismatch counters
        names = ("int_chain run_vs_run", "fma_chain run_vs_run", "fp16_matmul run_vs_run", "memory_pattern")

        def cmp_into(slot: int, a, b, words: int) -> None:
            k_cmp((_blocks(words),), (THREADS,), (a, b, np.uint64(words), bad[slot:slot + 1]))

        while time.time() < end_t:
            t0 = time.perf_counter()
            bad[...] = 0
            for slot, (name, kern, src) in enumerate((("int_chain", k_int, d_in), ("fma_chain", k_fma, d_f))):
                a, b = outs[name]
                it = np.int32(iters[name])
                kern((_blocks(n),), (THREADS,), (src, a, np.int32(n), it))
                kern((_blocks(n),), (THREADS,), (src, b, np.int32(n), it))
                cmp_into(slot, a, b, n)
            for _ in range(3):
                c1 = cp.matmul(mat_a, mat_b)
                c2 = cp.matmul(mat_a, mat_b)
                cmp_into(2, c1.view(cp.uint32).ravel(), c2.view(cp.uint32).ravel(), c1.size // 2)
            if loop % 8 == 0:
                mem_seed = (mem_seed * 1103515245 + 12345) & 0xFFFFFFFF
                k_fill((_blocks(mem_words),), (THREADS,), (d_mem, np.uint64(mem_words), np.uint32(mem_seed)))
                k_check((_blocks(mem_words),), (THREADS,), (d_mem, np.uint64(mem_words), np.uint32(mem_seed), bad[3:4]))
            cp.cuda.runtime.deviceSynchronize()
            busy = time.perf_counter() - t0
            counts = bad.get()
            for slot, count in enumerate(counts):
                if count:
                    raise CanaryDataError(f"CANARY_DATA_ERROR kernel={names[slot]} mismatches={int(count)} loop={loop}")
            if loop % 8 == 0:
                # Known-answer check: a slice of the integer kernel against the host implementation.
                ref = int_chain_reference(host_in[:REF_WORDS], int(iters["int_chain"]))
                got = outs["int_chain"][0][:REF_WORDS].get()
                wrong = int(np.count_nonzero(ref != got))
                if wrong:
                    raise CanaryDataError(f"CANARY_DATA_ERROR kernel=int_chain host_reference mismatches={wrong} loop={loop}")
            # Keep each launch near the target duration (the 2 s TDR timeout is the hard limit).
            per_launch = busy / 8.0
            for name in iters:
                if per_launch > MAX_KERNEL_S:
                    iters[name] = max(8, int(iters[name] * MAX_KERNEL_S / per_launch))
                elif per_launch < TARGET_KERNEL_S * 0.5:
                    iters[name] = min(4096, int(iters[name] * 1.5) + 1)

            duty = ramp_duty(time.time() - start, target_fraction)
            if duty < 0.999 and busy > 0:
                time.sleep(busy * (1.0 / duty - 1.0))
            loop += 1

            now = time.time()
            if now >= next_log:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                freq = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS))
                temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                try:
                    power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except pynvml.NVMLError:
                    power = 0.0
                stats["n"] += 1
                stats["util_sum"] += util
                stats["freq_sum"] += freq
                stats["util_max"] = max(stats["util_max"], util)
                stats["freq_max"] = max(stats["freq_max"], freq)
                stats["temp_max"] = max(stats["temp_max"], temp)
                stats["power_max"] = max(stats["power_max"], power)
                print(f"[canary] t-{int(max(0.0, end_t - now)):3d}s | util={util:5.1f}% freq={freq:6.0f}MHz "
                      f"temp={temp:5.1f}C power={power:6.1f}W loops={loop} duty={duty:.2f}")
                next_log = now + 1.0

    pynvml.nvmlShutdown()
    stats["loops"] = loop
    return stats


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="VoltVandal self-checking GPU canary")
    ap.add_argument("--mode", default="canary")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--target-percent", type=float, default=0.0, help="duty-cycle target; 0 or >=95 = full load")
    args = ap.parse_args(argv)

    print(f"Starting canary stress gpu={args.gpu} duration={args.seconds}s")
    try:
        stats = run(args.gpu, args.seconds, args.target_percent)
    except CanaryDataError as exc:
        print("\nTest Summary:")
        print(f"Error during test: {exc}")
        print("Status              : FAILED (data errors)")
        return 1
    except Exception as exc:
        print("\nTest Summary:")
        print(f"Error during test: {exc}")
        return 1

    n = max(1, int(stats["n"]))
    print("\nTest Summary:")
    print(
        f"Status              : Successfully maintain\n"
        f"Average Utilization : {stats['util_sum'] / n:.2f}%\n"
        f"Max Utilization     : {stats['util_max']:.2f}%\n"
        f"Average Frequency   : {stats['freq_sum'] / n:.2f} MHz\n"
        f"Max Frequency       : {stats['freq_max']:.2f} MHz\n"
        f"Max Temperature     : {stats['temp_max']:.2f} C\n"
        f"Max Power           : {stats['power_max']:.2f} W\n"
        f"Canary loops        : {int(stats['loops'])}\n"
        f"Data errors         : 0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
