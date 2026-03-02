#!/usr/bin/env python3
"""
VoltVandal v1.1 (single-file prototype)
--------------------------------
A safety-first Windows CLI harness for:
  - dumping NVIDIA VF curve via nvapi-cmd
  - making small edits (per-voltage-bin range)
  - applying candidate curve
  - running stress tools (doloMing + gpu-burn)
  - monitoring telemetry (NVML via pynvml)
  - auto-reverting to last-known-good on any failure
  - checkpoint/resume

DISCLAIMERS
- Overclocking/undervolting can crash your system/driver and may risk hardware.
- Use at your own risk. Start conservative. Keep temps under control.
- Many curve tools require Admin. Run this in an elevated terminal.

Dependencies:
  pip install pynvml

VF-curve backend (auto-selected, first available wins):
  1. nvapi_curve.py  — pure-Python ctypes wrapper (included, no extra install)
  2. nvapi-cmd.exe   — legacy subprocess fallback (pass via --nvapi-cmd)

Optional stress tools (place alongside this script or specify paths):
  - doloMing stress script (python) OR packaged exe
  - gpu-burn.exe   (optional)

Typical usage:
  # Dump and backup stock curve (GPU 0) — nvapi_curve.py used automatically
  python voltvandal.py dump --gpu 0 --out artifacts

  # Explicit legacy backend
  python voltvandal.py dump --gpu 0 --nvapi-cmd .\nvapi-cmd.exe --out artifacts

  # Run a simple UV sweep on a voltage bin range, stress with doloMing "ray"
  python voltvandal.py run --gpu 0 --out artifacts ^
    --mode uv --bin-min-mv 850 --bin-max-mv 950 --step-mv 5 --stress-seconds 90 ^
    --doloming .\doloMing\stress.py --doloming-mode ray

  # vlock: anchor at 987 mV, find max freq there, then UV-sweep all lower bins
  python voltvandal.py run --gpu 0 --out artifacts ^
    --mode vlock --target-voltage-mv 987 --step-mhz 5 --step-mv 5 ^
    --stress-seconds 120 --max-steps 30 ^
    --doloming .\doloMing\stress.py --doloming-mode frequency-max

  # Multi-mode sweep: 30s of ray + matrix + frequency-max at each step
  python voltvandal.py run --gpu 0 --out artifacts ^
    --mode uv --bin-min-mv 850 --bin-max-mv 950 --step-mv 5 ^
    --doloming .\doloMing\stress.py --doloming-modes ray,matrix,frequency-max --multi-stress-seconds 30

  # Resume from checkpoint
  python voltvandal.py resume --out artifacts

Notes:
- The VF curve CSV format expected:
    voltageUV,frequencyKHz
    450000,210000
    ...
- microvolts (uV) and kilohertz (kHz)

This is a prototype: keep it simple, transparent, and safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ----------------------------
# NVML / monitoring
# ----------------------------
try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None  # allow dump/apply without NVML

# VF-curve backend: prefer the native Python ctypes module; fall back to
# spawning nvapi-cmd.exe as a subprocess (legacy / compatibility path).
try:
    import nvapi_curve as _nvapi_native  # type: ignore
    _NVAPI_BACKEND = "native"
except Exception:
    _nvapi_native = None  # type: ignore[assignment]
    _NVAPI_BACKEND = "subprocess"

# GPU series reference profiles (optional companion module).
try:
    import gpu_profiles as _gpu_profiles  # type: ignore
except Exception:
    _gpu_profiles = None  # type: ignore[assignment]

# Matplotlib for post-run VF curve plots (optional — graceful fallback if absent).
try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")  # non-interactive; safe for background/headless use
    import matplotlib.pyplot as plt  # type: ignore
    _matplotlib_available = True
except Exception:
    plt = None  # type: ignore[assignment]
    _matplotlib_available = False


@dataclass
class CurvePoint:
    voltage_uv: int
    freq_khz: int


@dataclass
class CandidateResult:
    ok: bool
    reason: str
    telemetry_max_temp_c: Optional[int] = None
    telemetry_max_power_w: Optional[float] = None
    telemetry_any_throttle: Optional[bool] = None
    stress_exit_codes: Optional[dict] = None


@dataclass
class SessionState:
    gpu: int
    nvapi_cmd: str
    out_dir: str

    stock_curve_csv: str
    last_good_curve_csv: str
    checkpoint_json: str

    mode: str  # "uv" | "oc" | "hybrid"
    bin_min_mv: int
    bin_max_mv: int

    # steps
    step_mv: int
    step_mhz: int
    max_steps: int

    # stress
    stress_seconds: int
    doloming: Optional[str]
    doloming_mode: str
    gpuburn: Optional[str]

    # monitor thresholds
    poll_seconds: float
    temp_limit_c: int
    hotspot_limit_c: Optional[int]
    hotspot_offset_c: int
    power_limit_w: float
    abort_on_throttle: bool

    # multi-mode stress (runs each doloMing mode in sequence per step)
    doloming_modes: str = ""          # comma-separated modes, e.g. "ray,matrix,frequency-max"; empty = use doloming_mode
    multi_stress_seconds: int = 30    # per-mode duration when doloming_modes is set

    # progress
    current_step: int = 0
    stress_timeout: Optional[int] = None  # hard wall-clock cap on each stress run (seconds)
    current_offset_mv: int = 0
    current_offset_mhz: int = 0

    # hybrid mode state (survives resume)
    hybrid_phase: str = "uv"           # "uv" or "oc"
    hybrid_locked_mv: int = 0          # UV offset locked after phase 1
    hybrid_oc_start_step: int = 0      # step number where OC phase began

    # vlock mode state (survives resume)
    vlock_target_mv: int = 0           # user-supplied anchor voltage
    vlock_anchor_freq_khz: int = 0     # peak freq confirmed stable in phase 1
    vlock_uv_offset_mv: int = 0        # last confirmed UV offset (for display / legacy)
    vlock_uv_bin_idx: int = -1         # Phase 2 outer loop: which sub-anchor bin is being tuned
                                       # -1 = not started; counts down from anchor_idx-1 to 0
    vlock_p2_current_gain_khz: int = 0 # Phase 2: gain currently being tested for the active bin
                                       # 0 = use full oc_gain on next bin entry
    vlock_phase: str = "oc"            # "oc" | "uv" | "done"
    vlock_oc_base_freq_khz: int = 0    # freq floor found by floor-search (0 = use stock)
    vlock_start_freq_mhz: int = 0     # user-supplied OC start freq; baseline skipped when > 0

    # power limit
    power_limit_pct: int = 100        # % of GPU default TDP to apply before run (100 = no change)

    # display
    live_display: bool = True         # print live GPU metrics line during stress
    no_plot: bool = False             # skip post-run VF curve PNG generation

    # bookkeeping
    started_utc: str = ""
    updated_utc: str = ""


# ----------------------------
# Utility
# ----------------------------
def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_windows() -> bool:
    return os.name == "nt"


def warn_if_not_admin() -> None:
    """On Windows, emit a warning when the process is not elevated.

    NVAPI VF-curve operations require Administrator rights regardless of
    whether the native Python backend or nvapi-cmd.exe subprocess is used.
    """
    if not is_windows():
        return
    try:
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():  # type: ignore[attr-defined]
            eprint(
                "WARNING: Not running as Administrator. "
                "NVAPI VF-curve operations require elevated privileges. "
                "Re-run from an elevated terminal if commands fail."
            )
    except Exception:
        pass  # can't determine; continue anyway


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


# ----------------------------
# Curve IO
# ----------------------------
def load_curve_csv(path: Path) -> List[CurvePoint]:
    points: List[CurvePoint] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        # Normalise header names to lowercase for case-insensitive matching.
        norm = {name.lower(): name for name in reader.fieldnames}
        required_lower = {"voltageuv", "frequencykhz"}
        if not required_lower.issubset(set(norm.keys())):
            raise ValueError(
                f"CSV header must include voltageUV and frequencyKHz (case-insensitive), got {reader.fieldnames}"
            )
        col_v = norm["voltageuv"]
        col_f = norm["frequencykhz"]
        for row in reader:
            v = int(row[col_v])
            fk = int(row[col_f])
            points.append(CurvePoint(voltage_uv=v, freq_khz=fk))
    if not points:
        raise ValueError(f"No curve points loaded from {path}")
    return points


def write_curve_csv(path: Path, points: List[CurvePoint]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["voltageUV", "frequencyKHz"])
        for p in points:
            writer.writerow([p.voltage_uv, p.freq_khz])


def mv_to_uv(mv: int) -> int:
    return mv * 1000


def mhz_to_khz(mhz: int) -> int:
    return mhz * 1000


def apply_offsets_to_bin(
    points: List[CurvePoint],
    bin_min_mv: int,
    bin_max_mv: int,
    offset_mv: int,
    offset_mhz: int,
) -> List[CurvePoint]:
    """
    Make a candidate curve by:
      - shifting voltage (UV mode) for points within bin range
      - shifting frequency (OC mode) for points within bin range
    For safety:
      - keep voltages >= 0
      - keep frequencies >= 0
      - keep points sorted by voltage (we do not reorder; we only adjust values)
    """
    vmin_uv = mv_to_uv(bin_min_mv)
    vmax_uv = mv_to_uv(bin_max_mv)

    dv_uv = mv_to_uv(offset_mv)
    df_khz = mhz_to_khz(offset_mhz)

    new_points: List[CurvePoint] = []
    for p in points:
        if vmin_uv <= p.voltage_uv <= vmax_uv:
            new_v = p.voltage_uv + dv_uv
            new_f = p.freq_khz + df_khz
            new_v = max(0, new_v)
            new_f = max(0, new_f)
            new_points.append(CurvePoint(new_v, new_f))
        else:
            new_points.append(CurvePoint(p.voltage_uv, p.freq_khz))

    return new_points


def _build_vlock_curve(
    stock_points: List[CurvePoint],
    anchor_idx: int,
    anchor_voltage_uv: int,
    anchor_freq_khz: int,
    uv_offset_uv: int,
    cap_freq_khz: int = 0,
) -> List[CurvePoint]:
    """
    Build a candidate VF curve for vlock mode.

    Layout:
      - Index < anchor_idx  : voltage reduced by uv_offset_uv, freq unchanged (stock)
      - Index == anchor_idx : fixed at (anchor_voltage_uv, anchor_freq_khz)
      - Index > anchor_idx  : voltage unchanged, freq CAPPED at effective_cap

    cap_freq_khz (optional):
        0 (default) — cap all bins at/above anchor_idx at anchor_freq_khz.
                      Standard Phase 1 behaviour: GPU ceiling = confirmed OC freq.
        > 0         — override cap to this frequency instead (rarely needed).

    Capping above-anchor bins prevents the GPU from boosting beyond the test
    point during Phase 1.  Phase 2 uses _build_vlock_phase2_curves instead,
    which lets the GPU boost freely up to the anchor during stress so that
    doloMing's frequency-max mode produces meaningful scores.
    """
    effective_cap = cap_freq_khz if cap_freq_khz > 0 else anchor_freq_khz
    out: List[CurvePoint] = []
    for i, p in enumerate(stock_points):
        if i < anchor_idx:
            out.append(CurvePoint(max(p.voltage_uv - uv_offset_uv, 0), p.freq_khz))
        elif i == anchor_idx:
            out.append(CurvePoint(anchor_voltage_uv, min(anchor_freq_khz, effective_cap)))
        else:
            # Cap frequency so the GPU cannot boost above the test point.
            # Voltage is left at stock so the driver's safety margins remain intact.
            out.append(CurvePoint(p.voltage_uv, min(p.freq_khz, effective_cap)))
    return out


def _build_vlock_phase2_curves(
    stock_points: List[CurvePoint],
    last_good_points: List[CurvePoint],
    bin_idx: int,
    anchor_idx: int,
    anchor_voltage_uv: int,
    anchor_freq_khz: int,
    oc_gain_khz: int,
) -> Tuple[List[CurvePoint], List[CurvePoint]]:
    """
    Build (test_pts, save_pts) for Phase 2 per-bin OC-shift testing.

    Phase 2 applies the same frequency gain achieved in Phase 1 (oc_gain_khz)
    to sub-anchor bins individually while keeping their stock voltages.  Each
    bin is tested one at a time (highest → lowest).

    oc_gain_khz = anchor_achieved_freq - anchor_stock_freq  (e.g. +165 MHz)

    For bin_idx the target frequency is min(stock_freq + oc_gain, anchor_freq).
    Stock voltages are unchanged (no undervolting in Phase 2).

    test_pts  — curve applied to the GPU during the stress run:
      • bin_idx: stock voltage, shifted frequency (stock_freq + oc_gain_khz,
        capped at anchor_freq_khz).
      • All other sub-anchor bins: taken from last_good_points (already-
        confirmed bins have their shifted freqs; unconfirmed bins remain at
        stock).  NOT hard-capped at bin_target_freq — the GPU can boost freely
        up to the anchor point, letting doloMing's 3 modes exercise the full
        curve naturally:
          - ray/matrix (50% util target): GPU naturally runs in the sub-anchor
            voltage range, exercising lower bins under real moderate load.
          - frequency-max: GPU boosts to anchor_freq, giving a meaningful
            Score_gamma (not corrupted by a global MHz cap).
      • Anchor: (anchor_voltage_uv, anchor_freq_khz) — full confirmed OC freq.
      • Above-anchor: stock voltage, capped at anchor_freq_khz.

    save_pts  — curve written to last_good_curve.csv on a passing test:
      • bin_idx: stock voltage, shifted frequency at its own (uncapped) value.
      • All other sub-anchor bins: last_good settings (already-confirmed bins
        keep their shifted freqs; unconfirmed bins stay at stock).
      • Anchor: (anchor_voltage_uv, anchor_freq_khz) — confirmed OC ceiling.
      • Above-anchor: stock voltage, capped at anchor_freq_khz.
    """
    bin_target_freq = min(
        stock_points[bin_idx].freq_khz + oc_gain_khz, anchor_freq_khz
    )

    test_pts: List[CurvePoint] = []
    save_pts: List[CurvePoint] = []

    for i, p in enumerate(stock_points):
        if i < anchor_idx:
            if i == bin_idx:
                # Bin under test: apply the OC gain.
                test_pts.append(CurvePoint(p.voltage_uv, bin_target_freq))
                save_pts.append(CurvePoint(p.voltage_uv, bin_target_freq))
            else:
                # Other sub-anchor bins: last_good (no cap at bin_target_freq).
                # The GPU will exercise these naturally during ray/matrix load.
                lg = last_good_points[i]
                test_pts.append(CurvePoint(lg.voltage_uv, lg.freq_khz))
                save_pts.append(CurvePoint(lg.voltage_uv, lg.freq_khz))
        elif i == anchor_idx:
            # Anchor: use full confirmed OC frequency in BOTH test and save.
            # This lets frequency-max mode reach anchor_freq for a meaningful score.
            test_pts.append(CurvePoint(anchor_voltage_uv, anchor_freq_khz))
            save_pts.append(CurvePoint(anchor_voltage_uv, anchor_freq_khz))
        else:
            # Above anchor: stock voltage, capped at anchor_freq in both curves.
            test_pts.append(CurvePoint(p.voltage_uv, min(p.freq_khz, anchor_freq_khz)))
            save_pts.append(CurvePoint(p.voltage_uv, min(p.freq_khz, anchor_freq_khz)))

    return test_pts, save_pts


# ----------------------------
# Tool adapters
# ----------------------------
def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        capture_output=capture,
        text=True,
        check=False,
    )


_NVAPI_TIMEOUT_S = 30  # seconds before nvapi-cmd.exe is considered hung (subprocess path only)


def nvapi_dump_curve(nvapi_cmd: Optional[str], gpu: int, out_csv: Path) -> None:
    if _nvapi_native is not None:
        _nvapi_native.dump_curve(gpu, out_csv)
        if not out_csv.exists() or out_csv.stat().st_size < 10:
            raise RuntimeError(f"nvapi_curve.dump_curve did not produce a valid file: {out_csv}")
        return
    # subprocess fallback
    if not nvapi_cmd:
        raise RuntimeError(
            "No VF-curve backend available: nvapi_curve.py not importable "
            "and --nvapi-cmd not specified."
        )
    cmd = [nvapi_cmd, "-curve", str(gpu), "-1", str(out_csv)]
    cp = run_cmd(cmd, capture=True, timeout=_NVAPI_TIMEOUT_S)
    if cp.returncode != 0:
        raise RuntimeError(
            f"nvapi-cmd dump failed rc={cp.returncode}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )
    if not out_csv.exists() or out_csv.stat().st_size < 10:
        raise RuntimeError(f"nvapi-cmd dump did not produce a valid file: {out_csv}")


def nvapi_apply_curve(nvapi_cmd: Optional[str], gpu: int, in_csv: Path) -> None:
    if _nvapi_native is not None:
        _nvapi_native.apply_curve(gpu, in_csv)
        return
    # subprocess fallback
    if not nvapi_cmd:
        raise RuntimeError(
            "No VF-curve backend available: nvapi_curve.py not importable "
            "and --nvapi-cmd not specified."
        )
    cmd = [nvapi_cmd, "-curve", str(gpu), "1", str(in_csv)]
    cp = run_cmd(cmd, capture=True, timeout=_NVAPI_TIMEOUT_S)
    if cp.returncode != 0:
        raise RuntimeError(
            f"nvapi-cmd apply failed rc={cp.returncode}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}"
        )


_interrupted = threading.Event()


def _install_signal_handlers() -> None:
    """Install signal handlers so Ctrl+C reliably sets the _interrupted flag."""
    def _handler(sig, frame):
        _interrupted.set()
        eprint("\nCtrl+C received — stopping after current operation...")
    signal.signal(signal.SIGINT, _handler)
    if is_windows():
        try:
            signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def _reader_thread(pipe, lines: List[str], done: threading.Event) -> None:
    """Background thread to drain a pipe without blocking the main thread."""
    try:
        for line in iter(pipe.readline, ""):
            lines.append(line)
        pipe.close()
    except Exception:
        pass
    finally:
        done.set()


def start_process(
    cmd: List[str],
    cwd: Optional[Path] = None,
) -> subprocess.Popen:
    creationflags = 0
    if is_windows():
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    return subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )


def terminate_process_tree(p: subprocess.Popen, gentle_seconds: float = 1.0) -> None:
    if p.poll() is not None:
        return
    # On Windows, use taskkill for reliable tree kill
    if is_windows():
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
            return
        except Exception:
            pass
    else:
        try:
            p.terminate()
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < gentle_seconds:
        if p.poll() is not None:
            return
        time.sleep(0.05)

    try:
        p.kill()
    except Exception:
        pass


# ----------------------------
# Stress runners
# ----------------------------
def run_doloming(
    doloming_path: str,
    mode: str,
    seconds: int,
    workdir: Optional[Path],
    log_path: Path,
    abort_event: threading.Event,
    stress_timeout: Optional[int] = None,
    max_freq_mhz: int = 0,
) -> Tuple[int, str]:
    exe_is_py = doloming_path.lower().endswith(".py")
    base_cmd = [sys.executable, doloming_path] if exe_is_py else [doloming_path]

    # For frequency-max mode pass the target ceiling so doloMing's Score_gamma
    # and Avg/Max ratio are computed against the actual test freq, not the
    # GPU's absolute silicon max (e.g. 2010 MHz anchor vs 2100 MHz hardware max).
    _freq_args = (["--max-freq-mhz", str(max_freq_mhz)]
                  if mode == "frequency-max" and max_freq_mhz > 0 else [])

    candidates = [
        base_cmd + ["--mode", mode, "--seconds", str(seconds)] + _freq_args,
        base_cmd + [mode, str(seconds)] + _freq_args,
    ]

    last_output = ""
    for cmd in candidates:
        p = start_process(cmd, cwd=workdir)
        output_lines: List[str] = []
        reader_done = threading.Event()
        reader = threading.Thread(
            target=_reader_thread, args=(p.stdout, output_lines, reader_done), daemon=True
        )
        reader.start()
        deadline: Optional[float] = time.time() + stress_timeout if stress_timeout else None
        try:
            while not reader_done.is_set():
                if _interrupted.is_set():
                    terminate_process_tree(p)
                    raise KeyboardInterrupt("User pressed Ctrl+C")
                if abort_event.is_set():
                    terminate_process_tree(p)
                    reader.join(timeout=3.0)
                    log_path.write_text(
                        "".join(output_lines) + "\n--- TRUNCATED (killed by monitor abort) ---\n",
                        encoding="utf-8", errors="replace",
                    )
                    return (999, "ABORTED_BY_MONITOR")
                if deadline and time.time() > deadline:
                    terminate_process_tree(p)
                    reader.join(timeout=3.0)
                    log_path.write_text(
                        "".join(output_lines) + "\n--- TRUNCATED (killed by VoltVandal stress-timeout) ---\n",
                        encoding="utf-8", errors="replace",
                    )
                    return (998, "STRESS_TIMEOUT")
                reader_done.wait(timeout=0.25)
        finally:
            reader.join(timeout=3.0)
        p.wait()

        out_text = "".join(output_lines)
        last_output = out_text
        log_path.write_text(out_text, encoding="utf-8", errors="replace")

        if (
            p.returncode not in (0, None)
            and len(out_text) < 400
            and re.search(r"usage|help|unknown option|invalid", out_text, re.I)
        ):
            continue

        return (p.returncode or 0, out_text)

    return ((p.returncode or 1), last_output)


def run_gpuburn(
    gpuburn_path: str,
    seconds: int,
    workdir: Optional[Path],
    log_path: Path,
    abort_event: threading.Event,
    stress_timeout: Optional[int] = None,
) -> Tuple[int, str, bool]:
    cmd = [gpuburn_path, str(seconds)]
    p = start_process(cmd, cwd=workdir)
    output_lines: List[str] = []
    reader_done = threading.Event()
    reader = threading.Thread(
        target=_reader_thread, args=(p.stdout, output_lines, reader_done), daemon=True
    )
    reader.start()
    deadline: Optional[float] = time.time() + stress_timeout if stress_timeout else None
    try:
        while not reader_done.is_set():
            if _interrupted.is_set():
                terminate_process_tree(p)
                raise KeyboardInterrupt("User pressed Ctrl+C")
            if abort_event.is_set():
                terminate_process_tree(p)
                return (999, "ABORTED_BY_MONITOR", False)
            if deadline and time.time() > deadline:
                terminate_process_tree(p)
                return (998, "STRESS_TIMEOUT", False)
            reader_done.wait(timeout=0.25)
    finally:
        reader.join(timeout=3.0)
    p.wait()

    out_text = "".join(output_lines)
    log_path.write_text(out_text, encoding="utf-8", errors="replace")

    # Prefer an explicit numeric error counter if present.
    m = re.search(r"errors?\s*[:=]\s*(\d+)", out_text, re.I)
    if m:
        ok = int(m.group(1)) == 0
    else:
        # No explicit counter found: treat as ok unless failure keywords are present.
        ok = True
        if re.search(r"\bfail(ed)?\b|\berror\b", out_text, re.I):
            ok = False

    return (p.returncode or 0, out_text, ok)


# ----------------------------
# NVML monitoring
# ----------------------------
@dataclass
class MonitorSnapshot:
    temp_c: int
    hotspot_c: float          # real NvAPI hotspot, or estimated (edge + offset)
    vram_junction_c: Optional[float]  # real NvAPI VRAM junction, or None
    power_w: float
    clock_mhz: int
    mem_clock_mhz: int        # GPU memory clock
    util_gpu: int
    throttle_reasons: int
    voltage_mv: Optional[int] = None   # core voltage via NvAPI if available
    pstate: Optional[int] = None       # current P-state (0=P0 max perf, 8=P8 idle)
    perf_decrease: Optional[int] = None  # NvAPI_GPU_GetPerfDecreaseInfo bitmask
    topo_gpu_mw: Optional[int] = None  # GPU die power in mW (power topology)
    topo_total_mw: Optional[int] = None  # total board power in mW (power topology)


# ----------------------------
# GPU driver-reset (TDR) detection
# ----------------------------
# A real GPU TDR collapses the graphics clock to base-clock territory in a
# single poll interval — the driver resets the device and all P-state/clock
# machinery drops instantly.  This is a hardware-observable fact, not a
# utilisation heuristic:
#   • _TDR_MIN_BOOST_MHZ  — clock must reach this before the detector is
#                           armed (prevents false fires at session start when
#                           the GPU is still idling).
#   • _TDR_BASE_CLOCK_MHZ — if armed and clock drops at/below this the
#                           driver has reset.  600 MHz sits well above the
#                           typical polling noise floor but well below any
#                           real boost clock under load.
_TDR_MIN_BOOST_MHZ:  int = 1200   # arm threshold  — "we are under load"
_TDR_BASE_CLOCK_MHZ: int = 600    # collapse threshold — "driver just reset"

# NVML throttle-reason bitmask → short human-readable label
_THROTTLE_LABELS = {
    0x0000000000000001: "Idle",
    0x0000000000000002: "AppClk",
    0x0000000000000004: "PwrCap",
    0x0000000000000008: "HwSlowdn",
    0x0000000000000010: "SyncBst",
    0x0000000000000020: "SwTherm",
    0x0000000000000040: "HwTherm",
    0x0000000000000080: "PwrBrake",
    0x0000000000000100: "DispClk",
}

def _decode_throttle(reasons: int) -> str:
    """Return a compact string describing active NVML throttle reasons."""
    if reasons == 0:
        return ""
    active = [lbl for bit, lbl in _THROTTLE_LABELS.items() if reasons & bit]
    return "+".join(active) if active else f"0x{reasons:X}"


class NvmlMonitor:
    def __init__(
        self,
        gpu_index: int,
        poll_seconds: float,
        temp_limit_c: int,
        hotspot_limit_c: Optional[int],
        hotspot_offset_c: int,
        power_limit_w: float,
        abort_on_throttle: bool,
        log_csv: Path,
        live_display: bool = True,
    ):
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.temp_limit_c = temp_limit_c
        self.hotspot_limit_c = hotspot_limit_c
        self.hotspot_offset_c = hotspot_offset_c
        self.power_limit_w = power_limit_w
        self.abort_on_throttle = abort_on_throttle
        self.log_csv = log_csv
        self.live_display = live_display
        self._live_line_len: int = 0  # track printed width so we can erase cleanly

        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.max_temp: Optional[int] = None
        self.max_hotspot: Optional[float] = None
        self.max_power: Optional[float] = None
        self.any_throttle: bool = False
        self.last_snapshot: Optional[MonitorSnapshot] = None
        self._consecutive_errors: int = 0

        # GPU driver-reset (TDR) detection state
        self.driver_reset_detected: bool = False
        self._had_boost_clock: bool = False  # True once clock ≥ _TDR_MIN_BOOST_MHZ

    def start(self) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml not installed. `pip install pynvml`")
        pynvml.nvmlInit()
        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            pynvml.nvmlShutdown()
            raise

        first = not self.log_csv.exists()
        with self.log_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(
                    [
                        "utc",
                        "temp_c",
                        "hotspot_c",
                        "vram_junction_c",
                        "power_w",
                        "clock_mhz",
                        "mem_clock_mhz",
                        "util_gpu",
                        "voltage_mv",
                        "throttle_reasons",
                        "pstate",
                        "perf_decrease",
                        "topo_gpu_mw",
                        "topo_total_mw",
                    ]
                )

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set() and not _interrupted.is_set():
            try:
                temp = int(
                    pynvml.nvmlDeviceGetTemperature(
                        self.handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
                power = float(pynvml.nvmlDeviceGetPowerUsage(self.handle)) / 1000.0
                clock = int(
                    pynvml.nvmlDeviceGetClockInfo(
                        self.handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                )
                mem_clock = int(
                    pynvml.nvmlDeviceGetClockInfo(
                        self.handle, pynvml.NVML_CLOCK_MEM
                    )
                )
                util = int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                throttle = int(
                    pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
                )

                # ── NvAPI supplemental readings ───────────────────────────────
                hotspot: float = temp + self.hotspot_offset_c  # fallback estimate
                vram_junc: Optional[float] = None
                voltage_mv: Optional[int] = None
                pstate: Optional[int] = None
                perf_decrease: Optional[int] = None
                topo_gpu_mw: Optional[int] = None
                topo_total_mw: Optional[int] = None

                if _nvapi_native is not None:
                    try:
                        thr = _nvapi_native.get_thermal_sensors(self.gpu_index)
                        if thr["hotspot_c"] is not None:
                            hotspot = thr["hotspot_c"]
                        vram_junc = thr.get("vram_junction_c")
                    except Exception:
                        pass  # fall back to estimate
                    try:
                        voltage_mv = _nvapi_native.get_current_voltage_mv(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        pstate = _nvapi_native.get_current_pstate(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        perf_decrease = _nvapi_native.get_perf_decrease_info(self.gpu_index)
                    except Exception:
                        pass
                    try:
                        topo = _nvapi_native.get_power_topology_mw(self.gpu_index)
                        if topo:
                            topo_gpu_mw   = topo.get("gpu_mw")
                            topo_total_mw = topo.get("total_mw")
                    except Exception:
                        pass

                snap = MonitorSnapshot(
                    temp, hotspot, vram_junc, power, clock, mem_clock, util, throttle,
                    voltage_mv, pstate, perf_decrease, topo_gpu_mw, topo_total_mw,
                )
                self.last_snapshot = snap

                self.max_temp = (
                    temp if self.max_temp is None else max(self.max_temp, temp)
                )
                self.max_hotspot = (
                    hotspot if self.max_hotspot is None else max(self.max_hotspot, hotspot)
                )
                self.max_power = (
                    power if self.max_power is None else max(self.max_power, power)
                )
                if throttle != 0:
                    self.any_throttle = True

                # ── CSV logging ───────────────────────────────────────────────
                vram_str    = f"{vram_junc:.1f}" if vram_junc is not None else ""
                volt_str    = str(voltage_mv)   if voltage_mv is not None else ""
                pstate_str  = str(pstate)        if pstate    is not None else ""
                pdec_str    = f"0x{perf_decrease:X}" if perf_decrease is not None else ""
                gpu_mw_str  = str(topo_gpu_mw)  if topo_gpu_mw   is not None else ""
                tot_mw_str  = str(topo_total_mw) if topo_total_mw is not None else ""
                with self.log_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [now_utc_iso(), temp, f"{hotspot:.1f}", vram_str,
                         f"{power:.1f}", clock, mem_clock, util, volt_str, throttle,
                         pstate_str, pdec_str, gpu_mw_str, tot_mw_str]
                    )

                # ── Live display ──────────────────────────────────────────────
                if self.live_display:
                    parts = [
                        f"GPU {temp}°C",
                        f"Hot {hotspot:.0f}°C",
                    ]
                    if vram_junc is not None:
                        parts.append(f"VRAMJnc {vram_junc:.0f}°C")
                    if pstate is not None:
                        parts.append(f"P{pstate}")
                    parts += [
                        f"Core {clock} MHz",
                        f"Mem {mem_clock} MHz",
                        f"Load {util}%",
                    ]
                    if voltage_mv is not None:
                        parts.append(f"Volt {voltage_mv} mV")
                    if topo_total_mw is not None:
                        parts.append(f"BrdPwr {topo_total_mw/1000:.1f}W")
                    elif topo_gpu_mw is not None:
                        parts.append(f"GPUPwr {topo_gpu_mw/1000:.1f}W")
                    else:
                        parts.append(f"Pwr {power:.0f}W")
                    throttle_lbl = _decode_throttle(throttle)
                    if throttle_lbl and throttle_lbl != "Idle":
                        parts.append(f"Throt:{throttle_lbl}")
                    if self.driver_reset_detected:
                        parts.append("!! DRIVER RESET !!")
                    line = "  " + "  |  ".join(parts)
                    # Pad to fixed width so previous longer lines are fully erased
                    self._live_line_len = max(self._live_line_len, len(line))
                    sys.stderr.write(f"\r{line:<{self._live_line_len}}")
                    sys.stderr.flush()

                # ── Abort checks ──────────────────────────────────────────────
                if temp >= self.temp_limit_c:
                    self.abort_event.set()
                if self.hotspot_limit_c is not None and hotspot >= self.hotspot_limit_c:
                    self.abort_event.set()
                if power >= self.power_limit_w:
                    self.abort_event.set()
                if self.abort_on_throttle and throttle != 0:
                    self.abort_event.set()

                # ── GPU driver-reset (TDR) detection ──────────────────────────
                # Arm once we've seen the GPU under load; then a clock collapse
                # to base-clock territory is an unambiguous driver reset signal.
                if clock >= _TDR_MIN_BOOST_MHZ:
                    self._had_boost_clock = True
                if (self._had_boost_clock
                        and clock <= _TDR_BASE_CLOCK_MHZ
                        and not self.driver_reset_detected):
                    self.driver_reset_detected = True
                    eprint(
                        f"\n  !! GPU DRIVER RESET (TDR): clock collapsed to "
                        f"{clock} MHz after boost — GPU driver has reset. "
                        f"Aborting test and marking FAIL."
                    )
                    self.abort_event.set()

            except Exception as e:
                self._consecutive_errors += 1
                eprint(f"NVML poll error ({self._consecutive_errors}): {e}")
                if self._consecutive_errors >= 3:
                    self.abort_event.set()
                self.stop_event.wait(timeout=self.poll_seconds)
                continue
            self._consecutive_errors = 0

            self.stop_event.wait(timeout=self.poll_seconds)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5.0)
        if self.live_display and self._live_line_len > 0:
            sys.stderr.write("\r" + " " * (self._live_line_len + 2) + "\r")
            sys.stderr.flush()
            if self.thread.is_alive():
                # Thread didn't exit in time; skip nvmlShutdown to avoid calling
                # NVML from the main thread while the monitor thread may still be using it.
                eprint("WARNING: monitor thread did not stop within 5s; skipping nvmlShutdown.")
                return
        try:
            if pynvml is not None:
                pynvml.nvmlShutdown()
        except Exception:
            pass


# ----------------------------
# Session / persistence
# ----------------------------
def session_paths(out_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        out_dir / "stock_curve.csv",
        out_dir / "last_good_curve.csv",
        out_dir / "session.json",
    )


def save_session(state: SessionState) -> None:
    state.updated_utc = now_utc_iso()
    Path(state.checkpoint_json).write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )


def load_session(out_dir: Path) -> SessionState:
    _, _, checkpoint = session_paths(out_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"No session checkpoint found: {checkpoint}")
    data = json.loads(checkpoint.read_text(encoding="utf-8"))
    # Drop any keys not in SessionState (forward compat) and let defaults
    # fill in any missing keys (backward compat with older session files).
    valid_fields = {f.name for f in fields(SessionState)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return SessionState(**filtered)


# ----------------------------
# GPU power limit helpers
# ----------------------------
def nvml_read_default_power_w(gpu_index: int) -> Optional[float]:
    """Return the GPU's factory default TDP in watts via NVML, or None."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return float(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)) / 1000.0
    except Exception:
        return None


def nvml_apply_power_limit(gpu_index: int, target_w: float) -> bool:
    """Set GPU power limit to target_w watts.  Returns True on success."""
    if pynvml is None:
        return False
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(target_w * 1000))
        return True
    except Exception as e:
        eprint(f"[power-limit] Failed to apply {target_w:.0f}W: {e}")
        return False


def nvml_restore_power_limit(gpu_index: int, original_w: float) -> None:
    """Restore GPU power limit to original_w watts (best-effort)."""
    if pynvml is None:
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, int(original_w * 1000))
    except Exception as e:
        eprint(f"[power-limit] Failed to restore {original_w:.0f}W: {e}")


# ----------------------------
# VF curve plot
# ----------------------------
def plot_vf_curve(
    stock_csv: Path,
    last_good_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    """
    Generate a VF curve PNG comparing stock vs last-good curve.
    Returns the output path on success, None if matplotlib is unavailable or
    either CSV is missing.
    """
    if not _matplotlib_available:
        return None
    if not stock_csv.exists() or not last_good_csv.exists():
        return None

    def _load(p: Path):
        import csv as _csv
        rows = []
        with p.open(newline="") as f:
            for row in _csv.reader(f):
                if len(row) >= 2:
                    try:
                        rows.append((int(row[0]) / 1000, int(row[1]) / 1000))  # µV→mV, kHz→MHz
                    except ValueError:
                        pass  # skip header / non-numeric
        return rows

    stock = _load(stock_csv)
    last_good = _load(last_good_csv)
    if not stock or not last_good:
        return None

    s_v, s_f = zip(*stock)
    g_v, g_f = zip(*last_good)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(s_v, s_f, color="silver", linestyle="--", linewidth=1.5, label="Stock curve")
    ax.plot(g_v, g_f, color="#2196F3", linewidth=2, label="Last-good (tuned)")

    # Shade the difference between the two curves
    import numpy as np  # type: ignore
    common_v = sorted(set(s_v) | set(g_v))
    s_interp = np.interp(common_v, s_v, s_f)
    g_interp = np.interp(common_v, g_v, g_f)
    ax.fill_between(
        common_v, s_interp, g_interp,
        where=(g_interp >= s_interp), alpha=0.15, color="#4CAF50", label="Freq gain (OC)"
    )
    ax.fill_between(
        common_v, s_interp, g_interp,
        where=(g_interp < s_interp), alpha=0.15, color="#F44336", label="Freq reduction"
    )

    ax.set_xlabel("Voltage (mV)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("VoltVandal — VF Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


# ----------------------------
# doloMing output metric parsing
# ----------------------------
def _parse_doloming_stability(out_text: str, mode: str) -> Tuple[bool, Optional[str]]:
    """
    Parse doloMing stdout for actual error conditions in the Test Summary.

    IMPORTANT: "Unstable" and "Successfully maintain" are per-sample progress
    bar labels printed every ~100 ms while the test runs (they reflect whether
    the current utilisation sample is within tolerance of the 50% target).
    They are NOT final verdicts and appear in the output of every run —
    searching the full text for "Unstable" would produce a false FAIL on any
    run where utilisation ever deviated from target, which is essentially all
    of them.

    The doloMing Test Summary section (after "Test Summary:") never contains
    "Unstable".  Real failures are reported as:
        "Error during stress test: <exc>"   (ray / simple modes)
        "Error during test: <exc>"          (matrix / frequency-max modes)
    in the summary when an uncaught exception terminated the workload.

    Returns (is_stable, reason_if_unstable).
    """
    mode_key = mode.upper().replace("-", "_")

    # Only inspect the Test Summary section — progress bars are noise.
    summary_idx = out_text.rfind("Test Summary:")
    scan_text = out_text[summary_idx:] if summary_idx >= 0 else out_text

    if re.search(r"Error during (?:stress )?test:", scan_text, re.I):
        return False, f"DOLOMING_{mode_key}_STRESS_ERROR"

    return True, None


# ----------------------------
# Candidate evaluation
# ----------------------------
def evaluate_candidate(
    state: SessionState,
    candidate_csv: Path,
    candidate_label: str,
    max_freq_mhz: int = 0,
) -> CandidateResult:
    out_dir = Path(state.out_dir)
    logs_dir = out_dir / "logs"
    ensure_dir(logs_dir)

    try:
        nvapi_apply_curve(state.nvapi_cmd, state.gpu, candidate_csv)
        # Give the driver a moment to propagate the new curve before stressing
        _interrupted.wait(timeout=2.0)
        if _interrupted.is_set():
            raise KeyboardInterrupt("User pressed Ctrl+C")
    except Exception as ex:
        return CandidateResult(ok=False, reason=f"APPLY_FAILED: {ex}")

    monitor_log = logs_dir / "telemetry.csv"
    monitor: Optional[NvmlMonitor] = None
    abort_event = threading.Event()

    if pynvml is not None:
        monitor = NvmlMonitor(
            gpu_index=state.gpu,
            poll_seconds=state.poll_seconds,
            temp_limit_c=state.temp_limit_c,
            hotspot_limit_c=state.hotspot_limit_c,
            hotspot_offset_c=state.hotspot_offset_c,
            power_limit_w=state.power_limit_w,
            abort_on_throttle=state.abort_on_throttle,
            log_csv=monitor_log,
            live_display=state.live_display,
        )
        monitor.start()
        abort_event = monitor.abort_event

    stress_exit_codes = {}

    if state.doloming:
        # Multi-mode: run each mode in sequence for multi_stress_seconds each.
        # Single-mode: run doloming_mode for the full stress_seconds.
        _modes_raw = (state.doloming_modes or "").strip()
        _multi_modes = [m.strip() for m in _modes_raw.split(",") if m.strip()] if _modes_raw else []
        _use_multi = bool(_multi_modes)
        _run_modes = _multi_modes if _use_multi else [state.doloming_mode]
        _secs_each = state.multi_stress_seconds if _use_multi else state.stress_seconds

        # Effective hard timeout: user value if set, otherwise 5× the requested
        # test duration, with a floor of 300s to absorb long warm-up phases
        # (e.g. doloMing ray mode never stabilises at 50% on high-end GPUs and
        # can spend 2–3 minutes in warm-up before the timed test even starts).
        _dolo_timeout = state.stress_timeout if state.stress_timeout is not None else max(_secs_each * 5, 300)

        for _mode in _run_modes:
            dololog = logs_dir / f"{candidate_label}_doloming_{_mode}.log"
            rc, out_text = run_doloming(
                state.doloming,
                _mode,
                _secs_each,
                None,
                dololog,
                abort_event,
                stress_timeout=_dolo_timeout,
                max_freq_mhz=max_freq_mhz,
            )
            _key = f"doloming_{_mode}" if _use_multi else "doloming"
            stress_exit_codes[_key] = rc
            if rc != 0:
                if monitor:
                    monitor.stop()
                if rc == 998:
                    _src = "explicit --stress-timeout" if state.stress_timeout is not None else "auto (5× test duration)"
                    print(
                        f"\n  [STRESS_TIMEOUT] doloMing '{_mode}' was killed after "
                        f"{_dolo_timeout}s (timeout source: {_src}; "
                        f"test requested {_secs_each}s).\n"
                        f"  The stress tool's warm-up phase likely exceeded the timeout.\n"
                        f"  Fix: pass --stress-timeout with a larger value, "
                        f"or remove it to use auto (5× = {_secs_each * 5}s).\n"
                    )
                    _reason = f"DOLOMING_{_mode.upper().replace('-', '_')}_STRESS_TIMEOUT"
                else:
                    _reason = f"DOLOMING_{_mode.upper().replace('-', '_')}_RC_{rc}"
                return CandidateResult(
                    False,
                    _reason,
                    monitor.max_temp if monitor else None,
                    monitor.max_power if monitor else None,
                    monitor.any_throttle if monitor else None,
                    stress_exit_codes,
                )
            # Deliberately narrow: "no errors" and "error code: 0" must not trigger.
            # Match "oom", "out of memory", "failed", or "errors:" followed by a non-zero digit.
            if re.search(
                r"\boom\b|\bout of memory\b|\bfailed\b|errors?\s*[:=]\s*[1-9]", out_text, re.I
            ):
                if monitor:
                    monitor.stop()
                return CandidateResult(
                    False,
                    f"DOLOMING_{_mode.upper().replace('-', '_')}_OUTPUT_ERROR_KEYWORD",
                    monitor.max_temp if monitor else None,
                    monitor.max_power if monitor else None,
                    monitor.any_throttle if monitor else None,
                    stress_exit_codes,
                )
            # Parse doloMing's own stability verdict ("Successfully maintain" / "Unstable").
            _stable, _instability_reason = _parse_doloming_stability(out_text, _mode)
            if not _stable:
                if monitor:
                    monitor.stop()
                return CandidateResult(
                    False,
                    _instability_reason,
                    monitor.max_temp if monitor else None,
                    monitor.max_power if monitor else None,
                    monitor.any_throttle if monitor else None,
                    stress_exit_codes,
                )

    if state.gpuburn:
        burnlog = logs_dir / f"{candidate_label}_gpuburn.log"
        _burn_timeout = state.stress_timeout if state.stress_timeout is not None else state.stress_seconds * 5
        rc, out_text, parsed_ok = run_gpuburn(
            state.gpuburn, state.stress_seconds, None, burnlog, abort_event,
            stress_timeout=_burn_timeout,
        )
        stress_exit_codes["gpuburn"] = rc
        if rc != 0:
            if monitor:
                monitor.stop()
            if rc == 998:
                _src = "explicit --stress-timeout" if state.stress_timeout is not None else "auto (5× test duration)"
                print(
                    f"\n  [STRESS_TIMEOUT] gpu-burn was killed after "
                    f"{_burn_timeout}s (timeout source: {_src}; "
                    f"test requested {state.stress_seconds}s).\n"
                    f"  Fix: pass --stress-timeout with a larger value, "
                    f"or remove it to use auto (5× = {state.stress_seconds * 5}s).\n"
                )
                _burn_reason = "GPUBURN_STRESS_TIMEOUT"
            else:
                _burn_reason = f"GPUBURN_RC_{rc}"
            return CandidateResult(
                False,
                _burn_reason,
                monitor.max_temp if monitor else None,
                monitor.max_power if monitor else None,
                monitor.any_throttle if monitor else None,
                stress_exit_codes,
            )
        if not parsed_ok:
            if monitor:
                monitor.stop()
            return CandidateResult(
                False,
                "GPUBURN_ERRORS_DETECTED",
                monitor.max_temp if monitor else None,
                monitor.max_power if monitor else None,
                monitor.any_throttle if monitor else None,
                stress_exit_codes,
            )
        if re.search(r"\bnan\b|\bfailed\b|errors?\s*[:=]\s*[1-9]", out_text, re.I):
            if monitor:
                monitor.stop()
            return CandidateResult(
                False,
                "GPUBURN_OUTPUT_ERROR_KEYWORD",
                monitor.max_temp if monitor else None,
                monitor.max_power if monitor else None,
                monitor.any_throttle if monitor else None,
                stress_exit_codes,
            )

    if monitor:
        if monitor.abort_event.is_set():
            monitor.stop()
            _abort_reason = (
                "GPU_DRIVER_RESET_DETECTED"
                if monitor.driver_reset_detected
                else "MONITOR_ABORT_THRESHOLD"
            )
            return CandidateResult(
                False,
                _abort_reason,
                monitor.max_temp,
                monitor.max_power,
                monitor.any_throttle,
                stress_exit_codes,
            )
        monitor.stop()

    return CandidateResult(
        True,
        "PASS",
        monitor.max_temp if monitor else None,
        monitor.max_power if monitor else None,
        monitor.any_throttle if monitor else None,
        stress_exit_codes,
    )


def revert_to_last_good(state: SessionState) -> None:
    nvapi_apply_curve(state.nvapi_cmd, state.gpu, Path(state.last_good_curve_csv))


# ----------------------------
# vlock tuning loop
# ----------------------------
def run_vlock_session(state: SessionState) -> None:
    """
    Voltage-lock mode — two-phase automated curve finder.

    Phase 1 (OC): At the user's chosen voltage bin, step frequency up from
                  stock in +step_mhz increments until the stress test fails.
                  The last passing frequency becomes the anchor.

    Phase 2 (OC shift): Apply the same frequency gain from Phase 1 to each
                  sub-anchor bin (highest first), keeping stock voltages.
                  Each bin is tested individually with the GPU capped at its
                  shifted frequency.  Passing bins keep the gain; failing
                  bins are left at stock.  Above-anchor bins are clamped at
                  anchor_freq_khz in the saved curve.

    Produces a curve that:
      - Maximises clock speed at the chosen voltage (anchor point)
      - Extends the same OC gain to lower-voltage bins where silicon allows
    This matches the classic MSI Afterburner undervolt shape.
    """
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)
    last_good_csv = Path(state.last_good_curve_csv)
    stock_points = load_curve_csv(Path(state.stock_curve_csv))

    target_uv = mv_to_uv(state.vlock_target_mv)

    # Find the stock-curve bin whose voltage is closest to the target.
    anchor_idx = min(
        range(len(stock_points)),
        key=lambda i: abs(stock_points[i].voltage_uv - target_uv),
    )
    anchor_v_uv = stock_points[anchor_idx].voltage_uv   # actual bin voltage
    anchor_v_mv = anchor_v_uv // 1000
    anchor_stock_f_khz = stock_points[anchor_idx].freq_khz

    # OC-phase base frequency: normally stock, but may be lowered by floor-search
    # if the stock clock is unstable at the chosen voltage.  Persisted so resumes
    # after a floor-search start from the correct base frequency.
    oc_base_freq_khz = state.vlock_oc_base_freq_khz if state.vlock_oc_base_freq_khz else anchor_stock_f_khz

    if anchor_v_mv != state.vlock_target_mv:
        print(
            f"[vlock] Note: no bin at {state.vlock_target_mv} mV — "
            f"using nearest bin at {anchor_v_mv} mV."
        )

    _start_freq_note = (
        f"  OC start freq  : {state.vlock_start_freq_mhz} MHz  (baseline will be skipped)\n"
        if state.vlock_start_freq_mhz > 0 else ""
    )
    print(
        f"\n=== VoltVandal — vlock mode ===\n"
        f"  Anchor voltage : {anchor_v_mv} mV  "
        f"(stock freq at this bin: {anchor_stock_f_khz // 1000} MHz)\n"
        f"  Bins below anchor: {anchor_idx}  "
        f"(indices 0 … {max(anchor_idx - 1, 0)})\n"
        f"{_start_freq_note}"
        f"  Resuming at phase : {state.vlock_phase}\n"
    )

    steps_log = out_dir / "steps.jsonl"

    def _log(result: CandidateResult, label: str, extra: dict) -> None:
        with steps_log.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({**asdict(result), "utc": now_utc_iso(),
                            "label": label, **extra}) + "\n"
            )

    # Pre-compute a human-readable stress duration label used in both phase banners.
    _dolo_multi_modes = [m.strip() for m in (state.doloming_modes or "").split(",") if m.strip()]
    if _dolo_multi_modes:
        _stress_label = (
            f"{state.multi_stress_seconds}s × {len(_dolo_multi_modes)} modes"
            f" = {state.multi_stress_seconds * len(_dolo_multi_modes)}s total"
        )
    else:
        _stress_label = f"{state.stress_seconds}s"

    # ── Phase 1: OC search at anchor voltage ─────────────────────────────────
    if state.vlock_phase == "oc":

        # If --start-freq-mhz was given, skip the baseline and jump straight to
        # the requested step.  The value is rounded to the nearest step-mhz
        # boundary above stock; stock is assumed stable (user's responsibility).
        if state.vlock_start_freq_mhz > 0 and state.current_step == 0:
            _sf_khz   = mhz_to_khz(state.vlock_start_freq_mhz)
            _sf_step  = max(1, round((_sf_khz - anchor_stock_f_khz) / mhz_to_khz(state.step_mhz)))
            _sf_actual = (anchor_stock_f_khz + mhz_to_khz(state.step_mhz * _sf_step)) // 1000
            print(
                f"  Baseline skipped (--start-freq-mhz {state.vlock_start_freq_mhz} MHz).\n"
                f"  Starting OC search at step {_sf_step}: {_sf_actual} MHz.\n"
            )
            state.vlock_anchor_freq_khz = anchor_stock_f_khz
            state.current_step = _sf_step
            save_session(state)

        # Step 0: verify that stock frequency at the target voltage is stable.
        # This is the minimum bar — if it fails the voltage is simply too low.
        if state.current_step == 0:
            cand_pts = _build_vlock_curve(
                stock_points, anchor_idx, anchor_v_uv, anchor_stock_f_khz, 0
            )
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, cand_pts)
            label = f"vlock_oc_step000_{anchor_v_mv}mv_{anchor_stock_f_khz // 1000}mhz_baseline"
            print(
                f"\n== {label} ==\n"
                f"  Verifying stock frequency at anchor voltage before stepping up.\n"
            )
            result = evaluate_candidate(state, cand_csv, label,
                                        max_freq_mhz=anchor_stock_f_khz // 1000)
            _log(result, label, {"phase": "oc", "anchor_v_mv": anchor_v_mv,
                                 "freq_khz": anchor_stock_f_khz, "uv_offset_mv": 0})
            print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")

            if not result.ok:
                # Auto floor-search: step frequency DOWN from stock until stable.
                # The first passing frequency becomes the OC-phase base for Phase 1.
                print(
                    f"\n  Stock frequency ({anchor_stock_f_khz // 1000} MHz) is unstable at "
                    f"{anchor_v_mv} mV.\n"
                    f"  Auto-searching for highest stable frequency "
                    f"(stepping down {state.step_mhz} MHz per trial, "
                    f"up to {state.max_steps} steps)...\n"
                )
                floor_found = False
                for _ds in range(1, state.max_steps + 1):
                    if _interrupted.is_set():
                        revert_to_last_good(state)
                        raise KeyboardInterrupt("User pressed Ctrl+C")
                    _floor_f_khz = anchor_stock_f_khz - mhz_to_khz(state.step_mhz * _ds)
                    if _floor_f_khz <= 0:
                        break
                    _cand_pts = _build_vlock_curve(
                        stock_points, anchor_idx, anchor_v_uv, _floor_f_khz, 0
                    )
                    write_curve_csv(cand_csv, _cand_pts)
                    _floor_label = (
                        f"vlock_floor_step{_ds:03d}_{anchor_v_mv}mv"
                        f"_{_floor_f_khz // 1000}mhz"
                        f"_minus{state.step_mhz * _ds}mhz"
                    )
                    print(
                        f"\n== Floor search — {_floor_label} ==\n"
                        f"  Testing: {anchor_v_mv} mV  @  {_floor_f_khz // 1000} MHz  "
                        f"(−{state.step_mhz * _ds} MHz vs stock)\n"
                    )
                    _fres = evaluate_candidate(state, cand_csv, _floor_label,
                                               max_freq_mhz=_floor_f_khz // 1000)
                    _log(_fres, _floor_label, {
                        "phase": "floor_search",
                        "anchor_v_mv": anchor_v_mv,
                        "freq_khz": _floor_f_khz,
                        "delta_mhz": -(state.step_mhz * _ds),
                        "uv_offset_mv": 0,
                    })
                    print(f"Result: {'PASS' if _fres.ok else 'FAIL'} | {_fres.reason}")
                    if _fres.ok:
                        oc_base_freq_khz = _floor_f_khz
                        floor_found = True
                        shutil.copyfile(cand_csv, last_good_csv)
                        state.vlock_anchor_freq_khz = oc_base_freq_khz
                        state.vlock_oc_base_freq_khz = oc_base_freq_khz
                        state.current_step = 0
                        save_session(state)
                        print(
                            f"\n  Stable floor found: {anchor_v_mv} mV @ "
                            f"{oc_base_freq_khz // 1000} MHz  "
                            f"(−{state.step_mhz * _ds} MHz vs stock)\n"
                            f"  Continuing Phase 1 OC sweep upward from this baseline...\n"
                        )
                        break

                if not floor_found:
                    revert_to_last_good(state)
                    _lowest_mhz = max(
                        (anchor_stock_f_khz - mhz_to_khz(state.step_mhz * state.max_steps)) // 1000,
                        0,
                    )
                    print(
                        f"\nFAIL: No stable frequency found at {anchor_v_mv} mV "
                        f"(tested down to {_lowest_mhz} MHz).\n"
                        f"  → Try a higher --target-voltage-mv "
                        f"(e.g. {anchor_v_mv + 13} mV).\n"
                    )
                    return

            else:
                # Baseline passed: record stock freq as current best anchor.
                state.vlock_anchor_freq_khz = anchor_stock_f_khz
                state.vlock_oc_base_freq_khz = 0  # 0 signals "use stock" on resume
                shutil.copyfile(cand_csv, last_good_csv)
                state.current_step = 0   # step 0 done; loop below starts at 1
                save_session(state)

        # Steps 1+: step frequency up until failure.
        oc_step = max(state.current_step, 1)
        while oc_step <= state.max_steps:
            if _interrupted.is_set():
                eprint("\nInterrupted — reverting to last known-good curve...")
                revert_to_last_good(state)
                raise KeyboardInterrupt("User pressed Ctrl+C")

            freq_khz = oc_base_freq_khz + mhz_to_khz(state.step_mhz * oc_step)
            cand_pts = _build_vlock_curve(
                stock_points, anchor_idx, anchor_v_uv, freq_khz, 0
            )
            cand_csv = out_dir / "candidate.csv"
            write_curve_csv(cand_csv, cand_pts)
            _base_label = (
                "stock" if oc_base_freq_khz == anchor_stock_f_khz
                else f"floor+{state.step_mhz * oc_step}"
            )
            label = (
                f"vlock_oc_step{oc_step:03d}_{anchor_v_mv}mv"
                f"_{freq_khz // 1000}mhz"
                f"_plus{state.step_mhz * oc_step}mhz"
            )
            print(
                f"\n== Phase 1 — {label} ==\n"
                f"  Anchor: {anchor_v_mv} mV  @  {freq_khz // 1000} MHz  "
                f"(+{state.step_mhz * oc_step} MHz vs {_base_label})\n"
                f"  Stress: {_stress_label}\n"
            )

            result = evaluate_candidate(state, cand_csv, label,
                                        max_freq_mhz=freq_khz // 1000)
            _log(result, label, {"phase": "oc", "anchor_v_mv": anchor_v_mv,
                                 "freq_khz": freq_khz,
                                 "delta_mhz": state.step_mhz * oc_step,
                                 "uv_offset_mv": 0})
            print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")

            if result.ok:
                state.vlock_anchor_freq_khz = freq_khz
                shutil.copyfile(cand_csv, last_good_csv)
                state.current_step = oc_step
                save_session(state)
                oc_step += 1
            else:
                revert_to_last_good(state)
                print(
                    f"\n== Phase 1 complete ==\n"
                    f"  Anchor locked: {anchor_v_mv} mV  @  "
                    f"{state.vlock_anchor_freq_khz // 1000} MHz  "
                    f"(failed at {freq_khz // 1000} MHz)\n"
                    f"  Switching to Phase 2: UV sweep below anchor...\n"
                )
                state.vlock_phase = "uv"
                state.current_step = 0
                save_session(state)
                break
        else:
            print(
                f"Phase 1 max steps reached — anchor at "
                f"{state.vlock_anchor_freq_khz // 1000} MHz. Moving to Phase 2."
            )
            state.vlock_phase = "uv"
            state.current_step = 0
            save_session(state)

    # ── Phase 2: per-bin UV sweep below anchor ────────────────────────────────
    if state.vlock_phase == "uv":
        if state.vlock_anchor_freq_khz == 0:
            state.vlock_anchor_freq_khz = anchor_stock_f_khz   # safety fallback

        # OC gain achieved in Phase 1: the extra MHz confirmed at the anchor voltage.
        oc_gain_khz = state.vlock_anchor_freq_khz - anchor_stock_f_khz

        if anchor_idx == 0:
            print("Anchor is the lowest voltage bin — no sub-anchor bins to shift. Done.")
            state.vlock_phase = "done"
            save_session(state)
        elif oc_gain_khz <= 0:
            print(
                "Anchor frequency equals stock — no OC gain to apply to sub-anchor bins.\n"
                "Phase 2 skipped."
            )
            state.vlock_phase = "done"
            save_session(state)
        else:
            # Initialise outer bin-loop on first entry.
            if state.vlock_uv_bin_idx == -1:
                state.vlock_uv_bin_idx = anchor_idx - 1
                state.current_step = 0
                save_session(state)

            print(
                f"\nPhase 2 (per-bin OC shift): applying +{oc_gain_khz // 1000} MHz "
                f"to {anchor_idx} sub-anchor bin(s), tested individually.\n"
                f"  Anchor gain : {anchor_stock_f_khz // 1000} -> "
                f"{state.vlock_anchor_freq_khz // 1000} MHz  "
                f"(+{oc_gain_khz // 1000} MHz at {anchor_v_mv} mV)\n"
                f"  Sub-anchor  : stock voltages kept; full +{oc_gain_khz // 1000} MHz "
                f"tried first, steps down {state.step_mhz} MHz on failure.\n"
                f"  Test method : GPU free to boost up to anchor during stress;\n"
                f"                ray/matrix exercise sub-anchor bins at ~50% util,\n"
                f"                frequency-max exercises anchor for a clean score.\n"
                f"  Stability   : doloMing output metrics (\"Successfully maintain\" /\n"
                f"                \"Unstable\") used as primary pass/fail signal.\n"
                f"  Above anchor: clamped at {state.vlock_anchor_freq_khz // 1000} MHz.\n"
            )

            # ── Outer loop: one bin at a time, highest sub-anchor → lowest ────
            # For each bin we first try the full oc_gain_khz.  On failure we
            # step the gain down by step_mhz repeatedly until a stable frequency
            # is found or we exhaust all steps (leaving the bin at stock).
            step_khz = state.step_mhz * 1000

            while state.vlock_uv_bin_idx >= 0:
                if _interrupted.is_set():
                    eprint("\nInterrupted — reverting to last known-good curve...")
                    revert_to_last_good(state)
                    raise KeyboardInterrupt("User pressed Ctrl+C")

                bin_idx     = state.vlock_uv_bin_idx
                bin_stock_f = stock_points[bin_idx].freq_khz
                bin_v_mv    = stock_points[bin_idx].voltage_uv // 1000

                # Initialise gain for this bin on first attempt (or resume).
                if state.vlock_p2_current_gain_khz <= 0:
                    state.vlock_p2_current_gain_khz = oc_gain_khz

                current_gain  = state.vlock_p2_current_gain_khz
                bin_target_f  = min(bin_stock_f + current_gain,
                                    state.vlock_anchor_freq_khz)
                bin_actual_gain = bin_target_f - bin_stock_f

                print(
                    f"\n{'─' * 56}\n"
                    f"  Phase 2 — bin {bin_idx:2d} :  {bin_v_mv} mV  |  "
                    f"{bin_stock_f // 1000} -> {bin_target_f // 1000} MHz  "
                    f"(+{bin_actual_gain // 1000} MHz)\n"
                    f"{'─' * 56}"
                )

                last_good_pts = load_curve_csv(last_good_csv)
                test_pts, save_pts = _build_vlock_phase2_curves(
                    stock_points, last_good_pts,
                    bin_idx, anchor_idx,
                    anchor_v_uv, state.vlock_anchor_freq_khz,
                    current_gain,
                )
                cand_csv = out_dir / "candidate.csv"
                write_curve_csv(cand_csv, test_pts)
                label = (
                    f"vlock_p2_bin{bin_idx:03d}"
                    f"_{bin_v_mv}mv"
                    f"_{bin_target_f // 1000}mhz"
                )
                print(
                    f"\n== Phase 2, bin {bin_idx} ==\n"
                    f"  Bin    : {bin_v_mv} mV  @  {bin_stock_f // 1000} MHz  "
                    f"->  {bin_target_f // 1000} MHz  (+{bin_actual_gain // 1000} MHz)\n"
                    f"  (GPU free to boost up to anchor {state.vlock_anchor_freq_khz // 1000} MHz)\n"
                    f"  Stress : {_stress_label}\n"
                )

                result = evaluate_candidate(state, cand_csv, label,
                                            max_freq_mhz=state.vlock_anchor_freq_khz // 1000)
                _log(result, label, {
                    "phase": "p2_oc_shift",
                    "bin_idx": bin_idx,
                    "bin_stock_freq_khz": bin_stock_f,
                    "bin_target_freq_khz": bin_target_f,
                    "bin_v_mv": bin_v_mv,
                    "oc_gain_khz": current_gain,
                    "anchor_v_mv": anchor_v_mv,
                    "anchor_freq_khz": state.vlock_anchor_freq_khz,
                })
                print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")

                if result.ok:
                    # ── Bin confirmed at current gain ─────────────────────────
                    state.vlock_uv_offset_mv += 1   # repurpose as confirmed-bin count
                    write_curve_csv(last_good_csv, save_pts)
                    backups_dir = out_dir / "curve_backups"
                    ensure_dir(backups_dir)
                    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                    shutil.copyfile(last_good_csv,
                                    backups_dir / f"good_{label}_{ts}.csv")
                    print(f"  Bin {bin_idx}: CONFIRMED  "
                          f"{bin_stock_f // 1000} -> {bin_target_f // 1000} MHz  "
                          f"(+{bin_actual_gain // 1000} MHz at {bin_v_mv} mV)")
                    # Move to the next lower bin, reset gain tracker.
                    state.vlock_uv_bin_idx -= 1
                    state.vlock_p2_current_gain_khz = 0
                    state.current_step = 0
                else:
                    # ── Bin failed — step gain down and retry ─────────────────
                    revert_to_last_good(state)
                    next_gain = current_gain - step_khz
                    if next_gain > 0:
                        print(
                            f"  Bin {bin_idx}: FAILED at +{bin_actual_gain // 1000} MHz "
                            f"— stepping down to +{next_gain // 1000} MHz"
                        )
                        state.vlock_p2_current_gain_khz = next_gain
                        # Do NOT advance bin_idx — retry this bin at lower gain.
                    else:
                        # No more steps possible: leave this bin at stock.
                        print(
                            f"  Bin {bin_idx}: FAILED all gains — left at stock "
                            f"({bin_stock_f // 1000} MHz at {bin_v_mv} mV)"
                        )
                        state.vlock_uv_bin_idx -= 1
                        state.vlock_p2_current_gain_khz = 0
                        state.current_step = 0

                save_session(state)

            # All bins processed.
            state.vlock_phase = "done"
            save_session(state)

    # ── Final summary ─────────────────────────────────────────────────────────
    if state.vlock_phase == "done":
        oc_gain_khz = state.vlock_anchor_freq_khz - anchor_stock_f_khz
        # Derive per-bin frequency changes by comparing last_good_curve vs stock.
        confirmed_lines: List[str] = []
        skipped_lines:   List[str] = []
        if last_good_csv.exists() and anchor_idx > 0:
            final_pts = load_curve_csv(last_good_csv)
            for i in range(anchor_idx):
                stock_f = stock_points[i].freq_khz // 1000
                final_f = final_pts[i].freq_khz    // 1000
                v_mv    = stock_points[i].voltage_uv // 1000
                if final_f != stock_f:
                    confirmed_lines.append(
                        f"    bin {i:2d}: {v_mv:4d} mV  "
                        f"{stock_f} -> {final_f} MHz  (+{final_f - stock_f} MHz)"
                    )
                else:
                    skipped_lines.append(
                        f"    bin {i:2d}: {v_mv:4d} mV  "
                        f"{stock_f} MHz  (left at stock)"
                    )
        n_confirmed = len(confirmed_lines)
        n_total     = anchor_idx
        all_lines   = confirmed_lines + skipped_lines
        freq_block  = "\n".join(all_lines) if all_lines else "    (no bins processed)"

        print(
            f"\n{'=' * 56}\n"
            f"  vlock optimisation complete\n"
            f"{'=' * 56}\n"
            f"  Anchor  :  {anchor_v_mv} mV  @  "
            f"{state.vlock_anchor_freq_khz // 1000} MHz\n"
            f"  OC gain :  +{oc_gain_khz // 1000} MHz  "
            f"({n_confirmed}/{n_total} sub-anchor bin(s) confirmed)\n"
            f"\n  Sub-anchor bins:\n"
            f"{freq_block}\n"
            f"\n  Curve   :  {last_good_csv}\n"
            f"\n  To apply on next boot:\n"
            f"    python voltvandal.py restore --out {state.out_dir}\n"
        )


# ----------------------------
# Main tuning loop
# ----------------------------
def run_session(state: SessionState) -> None:
    out_dir = Path(state.out_dir)
    ensure_dir(out_dir)

    stock_csv = Path(state.stock_curve_csv)
    last_good_csv = Path(state.last_good_curve_csv)

    # Use stock as the reference baseline so offsets are not double-applied.
    stock_points = load_curve_csv(stock_csv)

    if state.current_step >= state.max_steps:
        print("Reached max steps. Nothing to do.")
        return

    # Sanity-check: warn if the GPU is already throttling at idle before we begin.
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            _h = pynvml.nvmlDeviceGetHandleByIndex(state.gpu)
            _throttle = int(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(_h))
            pynvml.nvmlShutdown()
            # Bit 0 (GPU idle / clocks optimised) is harmless; anything else is active throttle.
            _active = _throttle & ~0x1
            if _active != 0:
                eprint(
                    f"WARNING: GPU {state.gpu} is already throttling at baseline "
                    f"(reasons=0x{_throttle:08x}). Tuning results may be unreliable."
                )
        except Exception as _ex:
            eprint(f"WARNING: Could not read baseline throttle state: {_ex}")

    print(
        f"Starting from step {state.current_step}/{state.max_steps} mode={state.mode}"
    )

    while state.current_step < state.max_steps:
        if _interrupted.is_set():
            eprint("\nInterrupted — reverting to last known-good curve...")
            try:
                revert_to_last_good(state)
                eprint("Reverted successfully.")
            except Exception as ex:
                eprint(f"WARNING: revert failed: {ex}")
            raise KeyboardInterrupt("User pressed Ctrl+C")

        step = state.current_step + 1

        if state.mode == "uv":
            offset_mv, offset_mhz = -(state.step_mv * step), 0
        elif state.mode == "oc":
            offset_mv, offset_mhz = 0, state.step_mhz * step
        elif state.mode == "hybrid":
            # Two-phase approach:
            #   Phase 1 (hybrid_phase="uv"): lower voltage, clocks at stock.
            #     On first failure → lock UV at last_good + safety margin,
            #     transition to phase 2.
            #   Phase 2 (hybrid_phase="oc"): raise clocks on top of locked UV.
            #     On first failure → done.
            phase = getattr(state, "hybrid_phase", "uv")
            if phase == "uv":
                offset_mv = -(state.step_mv * step)
                offset_mhz = 0
            else:
                # OC phase: UV is locked at hybrid_locked_mv
                offset_mv = getattr(state, "hybrid_locked_mv", 0)
                oc_step = step - getattr(state, "hybrid_oc_start_step", 0)
                offset_mhz = state.step_mhz * oc_step
        else:
            raise ValueError(f"Unknown mode: {state.mode}")

        label = f"step{step:03d}_mv{offset_mv}_mhz{offset_mhz}"
        candidate_csv = out_dir / "candidate.csv"
        points_candidate = apply_offsets_to_bin(
            stock_points,
            bin_min_mv=state.bin_min_mv,
            bin_max_mv=state.bin_max_mv,
            offset_mv=offset_mv,
            offset_mhz=offset_mhz,
        )
        write_curve_csv(candidate_csv, points_candidate)

        print(
            f"\n== Candidate {label} ==\n"
            f"  Bin range: {state.bin_min_mv}-{state.bin_max_mv} mV\n"
            f"  Offset: {offset_mv} mV, {offset_mhz} MHz\n"
            f"  Stress: {state.stress_seconds}s (doloming={bool(state.doloming)}, gpuburn={bool(state.gpuburn)})\n"
            f"  Limits: edge<{state.temp_limit_c}C"
            f" hotspot<{state.hotspot_limit_c}C"
            f"({'NvAPI' if _nvapi_native else f'+{state.hotspot_offset_c}C est'})"
            f" power<{state.power_limit_w}W abort_on_throttle={state.abort_on_throttle}\n"
        )

        result = evaluate_candidate(state, candidate_csv, candidate_label=label)
        print(f"Result: {'PASS' if result.ok else 'FAIL'} | {result.reason}")

        steps_log = out_dir / "steps.jsonl"
        with steps_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        **asdict(result),
                        "utc": now_utc_iso(),
                        "label": label,
                        "offset_mv": offset_mv,
                        "offset_mhz": offset_mhz,
                    }
                )
                + "\n"
            )

        if result.ok:
            # Keep a timestamped backup before overwriting last_good_curve.csv.
            backups_dir = out_dir / "curve_backups"
            ensure_dir(backups_dir)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            shutil.copyfile(candidate_csv, backups_dir / f"good_{label}_{ts}.csv")
            shutil.copyfile(candidate_csv, last_good_csv)
            state.current_step = step  # advance so resume starts on the NEXT step
            state.current_offset_mv = offset_mv
            state.current_offset_mhz = offset_mhz
            save_session(state)
            print(f"Committed as last_good_curve.csv (step {step}). Continuing...")
        else:
            print("Reverting to last_good curve...")
            try:
                revert_to_last_good(state)
            except Exception as ex:
                eprint(f"CRITICAL: revert to last_good failed: {ex}")
                eprint("GPU may still have the failed candidate curve applied. Restore manually:")
                eprint(f"  python voltvandal.py restore --out {state.out_dir}")
                state.current_step = step - 1
                save_session(state)
                sys.exit(1)

            # Hybrid mode: UV failure triggers transition to OC phase
            if state.mode == "hybrid" and state.hybrid_phase == "uv":
                # Lock UV at last successful offset + one step of safety margin
                safety_mv = state.current_offset_mv + state.step_mv  # back off one step
                state.hybrid_phase = "oc"
                state.hybrid_locked_mv = safety_mv
                state.hybrid_oc_start_step = step
                # Don't increment step — reuse this step number for first OC attempt
                state.current_step = step - 1
                save_session(state)
                print(
                    f"\n== Hybrid phase 1 (UV) complete ==\n"
                    f"  UV floor found at {state.current_offset_mv} mV (failed at {offset_mv} mV)\n"
                    f"  Locked UV at {safety_mv} mV (with {state.step_mv} mV safety margin)\n"
                    f"  Switching to phase 2: OC sweep at locked UV...\n"
                )
                continue

            state.current_step = step - 1
            save_session(state)
            _steps_passed = step - 1
            if _steps_passed == 0:
                # Failed on the very first step — the starting configuration is already at its limit.
                if state.mode == "uv":
                    print(
                        f"\n== Tuning stopped — step 1 failed ==\n"
                        f"  The first UV reduction (−{state.step_mv} mV) was already unstable.\n"
                        f"  Your GPU is at its voltage floor with the current settings.\n"
                        f"  Options:\n"
                        f"    • Try smaller --step-mv (e.g. --step-mv 2) for finer granularity.\n"
                        f"    • Narrow the bin range (--bin-min-mv / --bin-max-mv).\n"
                        f"    • Run a longer stress (--stress-seconds) to confirm stock is stable.\n"
                    )
                elif state.mode == "oc":
                    print(
                        f"\n== Tuning stopped — step 1 failed ==\n"
                        f"  The first OC step (+{state.step_mhz} MHz) was already unstable.\n"
                        f"  Your GPU is at its clock ceiling with the current voltage.\n"
                        f"  Options:\n"
                        f"    • Try smaller --step-mhz (e.g. --step-mhz 5) for finer search.\n"
                        f"    • Raise voltage range (--bin-min-mv / --bin-max-mv) to allow more headroom.\n"
                    )
                else:
                    print(
                        f"\n== Tuning stopped — step 1 failed ==\n"
                        f"  Could not apply even the first adjustment in {state.mode} mode.\n"
                        f"  Consider smaller step sizes or a different bin range, then resume.\n"
                    )
            else:
                _best_mv = state.current_offset_mv
                _best_mhz = state.current_offset_mhz
                print(
                    f"\n== Tuning stopped — {_steps_passed} step(s) passed, step {step} failed ==\n"
                    f"  Best result : offset {_best_mv:+d} mV, {_best_mhz:+d} MHz\n"
                    f"  Last-good curve saved. Resume from here:\n"
                    f"    python voltvandal.py resume --out {state.out_dir}\n"
                    f"  (Resume retries step {step} — adjust limits/step sizes if needed.)\n"
                )
            break


# ----------------------------
# Commands
# ----------------------------
def cmd_dump(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    stock_csv, last_good_csv, checkpoint = session_paths(out_dir)

    print(f"Dumping curve (gpu={args.gpu}) to {stock_csv} ...")
    nvapi_dump_curve(args.nvapi_cmd, args.gpu, stock_csv)

    if not last_good_csv.exists():
        shutil.copyfile(stock_csv, last_good_csv)
        print(f"Saved last_good_curve.csv (initial) -> {last_good_csv}")

    print("Done.")
    print(f"Stock curve:     {stock_csv}")
    print(f"Last good curve: {last_good_csv}")
    print(f"Session file:    {checkpoint}")


def build_session_from_args(args: argparse.Namespace) -> SessionState:
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    stock_csv, last_good_csv, checkpoint = session_paths(out_dir)

    if not stock_csv.exists():
        print("No stock_curve.csv found; dumping now...")
        nvapi_dump_curve(args.nvapi_cmd, args.gpu, stock_csv)
    if not last_good_csv.exists():
        shutil.copyfile(stock_csv, last_good_csv)

    return SessionState(
        gpu=args.gpu,
        nvapi_cmd=args.nvapi_cmd,
        out_dir=str(out_dir),
        stock_curve_csv=str(stock_csv),
        last_good_curve_csv=str(last_good_csv),
        checkpoint_json=str(checkpoint),
        mode=args.mode,
        bin_min_mv=args.bin_min_mv,
        bin_max_mv=args.bin_max_mv,
        step_mv=args.step_mv,
        step_mhz=args.step_mhz,
        max_steps=args.max_steps,
        stress_seconds=args.stress_seconds,
        doloming=args.doloming,
        doloming_mode=args.doloming_mode,
        doloming_modes=getattr(args, "doloming_modes", ""),
        multi_stress_seconds=getattr(args, "multi_stress_seconds", 30),
        gpuburn=args.gpuburn,
        stress_timeout=getattr(args, "stress_timeout", None),
        poll_seconds=args.poll_seconds,
        temp_limit_c=args.temp_limit_c,
        hotspot_limit_c=getattr(args, "hotspot_limit_c", 90),
        hotspot_offset_c=getattr(args, "hotspot_offset_c", 15),
        power_limit_w=args.power_limit_w,
        abort_on_throttle=args.abort_on_throttle,
        current_step=0,
        current_offset_mv=0,
        current_offset_mhz=0,
        vlock_target_mv=getattr(args, "target_voltage_mv", 0),
        vlock_start_freq_mhz=getattr(args, "start_freq_mhz", 0),
        vlock_anchor_freq_khz=0,
        vlock_uv_offset_mv=0,
        vlock_phase="oc",
        power_limit_pct=getattr(args, "power_limit_pct", 100),
        live_display=not getattr(args, "no_live_display", False),
        no_plot=getattr(args, "no_plot", False),
        started_utc=now_utc_iso(),
        updated_utc=now_utc_iso(),
    )


def _dispatch_session(state: SessionState) -> None:
    """Route to the correct tuning loop based on session mode, applying and
    restoring a GPU power limit if --power-limit-pct was specified."""
    original_power_w: Optional[float] = None

    if state.power_limit_pct != 100:
        default_w = nvml_read_default_power_w(state.gpu)
        if default_w is not None:
            target_w = default_w * state.power_limit_pct / 100.0
            max_w    = default_w * 1.15  # hard cap at 115 %
            target_w = min(target_w, max_w)
            # Read the *current* limit (may already differ from default) so we
            # restore exactly what was set before this session ran.
            try:
                if pynvml is not None:
                    pynvml.nvmlInit()
                    h = pynvml.nvmlDeviceGetHandleByIndex(state.gpu)
                    original_power_w = float(
                        pynvml.nvmlDeviceGetPowerManagementLimit(h)
                    ) / 1000.0
            except Exception:
                original_power_w = default_w
            if nvml_apply_power_limit(state.gpu, target_w):
                print(
                    f"[power-limit] Applied {target_w:.0f}W "
                    f"({state.power_limit_pct}% of {default_w:.0f}W default)"
                )
        else:
            print("[power-limit] Could not read default TDP — power limit unchanged.")

    try:
        if state.mode == "vlock":
            run_vlock_session(state)
        else:
            run_session(state)
    finally:
        if original_power_w is not None:
            nvml_restore_power_limit(state.gpu, original_power_w)
            print(f"[power-limit] Restored to {original_power_w:.0f}W")

        # Post-run VF curve plot
        stock_csv    = Path(state.stock_curve_csv)
        last_good    = Path(state.last_good_curve_csv)
        plot_out     = Path(state.out_dir) / "vf_curve_plot.png"
        if not getattr(state, "no_plot", False):
            result = plot_vf_curve(stock_csv, last_good, plot_out)
            if result:
                print(f"[plot] VF curve saved → {result}")
            elif not _matplotlib_available:
                print("[plot] matplotlib not installed — skipping curve plot.  pip install matplotlib")


def cmd_run(args: argparse.Namespace) -> None:
    state = build_session_from_args(args)
    save_session(state)
    print(f"Wrote session checkpoint: {state.checkpoint_json}")
    _dispatch_session(state)


def cmd_resume(args: argparse.Namespace) -> None:
    state = load_session(Path(args.out))
    print(f"Loaded session: {state.checkpoint_json}")
    _dispatch_session(state)


def cmd_restore(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    _, last_good_csv, _ = session_paths(out_dir)
    curve = Path(args.curve) if args.curve else last_good_csv
    if not curve.exists():
        raise FileNotFoundError(f"Curve file not found: {curve}")
    print(f"Applying curve: {curve}")
    nvapi_apply_curve(args.nvapi_cmd, args.gpu, curve)
    print("Done.")


# ----------------------------
# Startup persistence (Windows Task Scheduler)
# ----------------------------
_STARTUP_TASK_PREFIX = "VoltVandal-RestoreCurve"


def _startup_task_name(gpu: int) -> str:
    return f"{_STARTUP_TASK_PREFIX}-GPU{gpu}"


def _run_powershell(script: str) -> "subprocess.CompletedProcess[str]":
    """Run a PowerShell script via -EncodedCommand to avoid shell-quoting issues."""
    import base64
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
    )


def cmd_register_startup(args: argparse.Namespace) -> None:
    """Register a Windows Task Scheduler logon task to restore the last-good curve."""
    if not is_windows():
        eprint("ERROR: register-startup is Windows-only (uses Task Scheduler).")
        raise SystemExit(1)

    script_path = Path(__file__).resolve()
    python_exe = Path(sys.executable).resolve()
    out_dir = Path(args.out).resolve()
    delay = args.startup_delay
    task = _startup_task_name(args.gpu)

    if not out_dir.exists():
        eprint(f"WARNING: --out directory does not exist yet: {out_dir}")
        eprint("         The task will fail at logon until a tuning session writes last_good_curve.csv.")

    def _q(s: object) -> str:
        """Escape a value for use inside a PowerShell single-quoted string."""
        return str(s).replace("'", "''")

    ps = f"""$action = New-ScheduledTaskAction `
    -Execute '{_q(python_exe)}' `
    -Argument '"{_q(script_path)}" restore --gpu {args.gpu} --out "{_q(out_dir)}"' `
    -WorkingDirectory '{_q(script_path.parent)}'

$trigger = New-ScheduledTaskTrigger -AtLogon -User $env:USERNAME
$trigger.Delay = 'PT{delay}S'

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew `
    -Hidden $true

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName '{_q(task)}' `
    -Action   $action `
    -Trigger  $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description 'VoltVandal: restore last-good VF curve {delay}s after logon (GPU {args.gpu})' `
    -Force | Out-Null
Write-Output 'REGISTERED'"""

    cp = _run_powershell(ps)
    if cp.returncode != 0 or "REGISTERED" not in cp.stdout:
        eprint("ERROR: Failed to register startup task.")
        eprint(cp.stderr.strip() or cp.stdout.strip())
        raise SystemExit(1)

    curve_path = out_dir / "last_good_curve.csv"
    print(f"Startup task registered: {task}")
    print(f"  Trigger  : logon  +{delay}s delay  (elevated / RunLevel Highest)")
    print(f"  Curve    : {curve_path}")
    print(f"  GPU      : {args.gpu}")
    print(f'\nVerify : schtasks /query /tn "{task}" /fo list')
    print(f"Remove : python voltvandal.py unregister-startup --gpu {args.gpu}")


def cmd_unregister_startup(args: argparse.Namespace) -> None:
    """Remove the VoltVandal logon startup task from Windows Task Scheduler."""
    if not is_windows():
        eprint("ERROR: unregister-startup is Windows-only.")
        raise SystemExit(1)

    task = _startup_task_name(args.gpu)

    def _q(s: object) -> str:
        return str(s).replace("'", "''")

    ps = f"""Unregister-ScheduledTask -TaskName '{_q(task)}' -Confirm:$false -ErrorAction Stop
Write-Output 'REMOVED'"""

    cp = _run_powershell(ps)
    if cp.returncode != 0 or "REMOVED" not in cp.stdout:
        eprint(f"ERROR: Failed to remove task '{task}'.")
        eprint(cp.stderr.strip() or cp.stdout.strip())
        raise SystemExit(1)

    print(f"Startup task removed: {task}")


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    # Pre-scan argv for --gpu-profile so we can apply it as defaults before
    # the full parse (profile values are overridden by any explicit CLI flags).
    _profile_defaults: dict = {}
    _pre_profile: Optional[str] = None
    for _i, _a in enumerate(sys.argv[1:]):
        if _a == "--gpu-profile" and _i + 2 < len(sys.argv):
            _pre_profile = sys.argv[_i + 2]
            break
        if _a.startswith("--gpu-profile="):
            _pre_profile = _a.split("=", 1)[1]
            break
    if _pre_profile is not None:
        if _gpu_profiles is None:
            eprint("ERROR: --gpu-profile requires gpu_profiles.py alongside voltvandal.py.")
            sys.exit(2)
        try:
            _prof = _gpu_profiles.get_profile(_pre_profile)
        except KeyError as _ke:
            eprint(f"ERROR: {_ke}")
            sys.exit(2)
        _profile_defaults = {
            k: _prof[k]
            for k in (
                "target_voltage_mv", "step_mhz", "step_mv", "max_steps",
                "stress_seconds", "multi_stress_seconds",
                "temp_limit_c", "hotspot_limit_c", "power_limit_w",
                "bin_min_mv", "bin_max_mv",
            )
            if k in _prof
        }
        eprint(f"Applying GPU profile '{_pre_profile}' ({_prof['name']}) as defaults.")

    p = argparse.ArgumentParser(
        description="VoltVandal v1.1 (single-file) - VF curve tweak + stress + monitor harness."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    common.add_argument(
        "--nvapi-cmd", dest="nvapi_cmd", default=None,
        help="Path to nvapi-cmd.exe (optional if nvapi_curve.py native backend is available)"
    )
    common.add_argument(
        "--out", required=True, help="Artifacts output folder (e.g. artifacts)"
    )

    sp = sub.add_parser(
        "dump",
        parents=[common],
        help="Dump current VF curve and create stock/last_good artifacts.",
    )
    sp.set_defaults(func=cmd_dump)

    sp = sub.add_parser(
        "run", parents=[common], help="Start a new session and run tuning steps."
    )
    sp.add_argument("--mode", choices=["uv", "oc", "hybrid", "vlock"], default="uv")
    sp.add_argument(
        "--bin-min-mv", type=int, default=850, help="Min mV for affected voltage bin"
    )
    sp.add_argument(
        "--bin-max-mv", type=int, default=1000, help="Max mV for affected voltage bin. Default 1000"
    )
    sp.add_argument(
        "--step-mv",
        type=int,
        default=5,
        help="mV step size per step (uv/hybrid). Default 5",
    )
    sp.add_argument(
        "--step-mhz",
        type=int,
        default=5,
        help="MHz step size per step (oc/hybrid). Default 5",
    )
    sp.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max tuning steps to attempt. Default 20",
    )
    sp.add_argument(
        "--stress-seconds",
        type=int,
        default=120,
        help="Seconds per stress tool step. Default 120",
    )
    sp.add_argument(
        "--doloming",
        type=str,
        default="auto",
        help="Path to doloMing stress script, or 'auto' to detect, or 'none' to disable. Default: auto",
    )
    sp.add_argument(
        "--doloming-mode",
        type=str,
        default="frequency-max",
        help="doloMing mode (e.g. frequency-max/ray/matrix). Default frequency-max",
    )
    sp.add_argument(
        "--doloming-modes",
        dest="doloming_modes",
        type=str,
        default="",
        help=(
            "Comma-separated doloMing modes to run in sequence per step "
            "(e.g. 'ray,matrix,frequency-max'). When set, overrides --doloming-mode "
            "and uses --multi-stress-seconds for per-mode duration."
        ),
    )
    sp.add_argument(
        "--multi-stress-seconds",
        dest="multi_stress_seconds",
        type=int,
        default=30,
        help="Per-mode duration (seconds) when --doloming-modes is used. Default 30",
    )
    sp.add_argument(
        "--gpuburn", type=str, default=None, help="Path to gpu-burn exe (optional)"
    )
    sp.add_argument(
        "--stress-timeout",
        dest="stress_timeout",
        type=int,
        default=None,
        metavar="SEC",
        help=(
            "Hard wall-clock timeout (seconds) per stress-tool run. "
            "Kills the stress process if it exceeds this limit. "
            "If omitted, VoltVandal auto-computes 5× the per-mode test "
            "duration (e.g. --multi-stress-seconds 60 -> 300s), which is "
            "enough to absorb long warm-up phases without risking infinite "
            "hangs. Pass an explicit value only if you need a tighter cap."
        ),
    )
    sp.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="NVML poll interval seconds. Default 1.0",
    )
    sp.add_argument(
        "--temp-limit-c", type=int, default=83, help="Abort edge temp threshold. Default 83C"
    )
    sp.add_argument(
        "--hotspot-limit-c", dest="hotspot_limit_c", type=int, default=90,
        help="Abort hotspot temp threshold. Default 90C"
    )
    sp.add_argument(
        "--hotspot-offset-c", dest="hotspot_offset_c", type=int, default=15,
        help="Hotspot-to-edge offset fallback when NvAPI sensors unavailable. Default 15C"
    )
    sp.add_argument(
        "--power-limit-w",
        type=float,
        default=370.0,
        help="Abort power threshold. Default 370W",
    )
    sp.add_argument(
        "--abort-on-throttle",
        action="store_true",
        help="Abort if any throttle reason is nonzero",
    )
    sp.add_argument(
        "--power-limit-pct",
        dest="power_limit_pct",
        type=int,
        default=100,
        metavar="PCT",
        help=(
            "Set GPU power limit as a percentage of the factory default TDP "
            "before the run starts, and restore it on exit. "
            "Accepted range: 1–115 (hardware enforces its own max; values above "
            "115%% are clamped). Default 100 (no change). "
            "Example: --power-limit-pct 110 to allow 10%% above default TDP."
        ),
    )
    sp.add_argument(
        "--no-live-display",
        dest="no_live_display",
        action="store_true",
        help=(
            "Disable the live GPU metrics line printed to stderr during each "
            "stress run (GPU temp, hotspot, VRAM junction, core/mem clocks, "
            "load%%, voltage, power). Useful when piping output to a file."
        ),
    )
    sp.add_argument(
        "--no-plot",
        dest="no_plot",
        action="store_true",
        help=(
            "Skip generating the VF curve PNG after the run completes. "
            "By default VoltVandal saves artifacts/vf_curve_plot.png comparing "
            "the stock curve against the last-good tuned curve (requires matplotlib)."
        ),
    )
    sp.add_argument(
        "--target-voltage-mv",
        dest="target_voltage_mv",
        type=int,
        default=0,
        metavar="mV",
        help=(
            "[vlock mode] Voltage (mV) to lock the curve anchor at. "
            "The nearest stock-curve bin is selected automatically. "
            "Required when --mode vlock. Example: --target-voltage-mv 987"
        ),
    )
    sp.add_argument(
        "--start-freq-mhz",
        dest="start_freq_mhz",
        type=int,
        default=0,
        metavar="MHz",
        help=(
            "[vlock mode] Skip the step-0 baseline test and begin the OC "
            "search directly at this frequency. Useful when stock stability "
            "at the anchor voltage is already confirmed and you want to start "
            "near your expected ceiling rather than stepping up from scratch. "
            "The value is rounded to the nearest --step-mhz boundary above "
            "the stock anchor frequency. "
            "Example: --start-freq-mhz 1980"
        ),
    )
    sp.add_argument(
        "--gpu-profile",
        dest="gpu_profile",
        default=None,
        metavar="PROFILE",
        help=(
            "Apply a GPU-series profile as default values "
            "(e.g. rtx20, rtx30, rtx40, rtx50). "
            "Any explicit flag overrides the profile. "
            "Run with --list-profiles to see all available profiles."
        ),
    )
    # Apply GPU profile values as subparser defaults (overridden by explicit flags).
    if _profile_defaults:
        sp.set_defaults(**_profile_defaults)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser(
        "resume", help="Resume an existing session from --out/session.json"
    )
    sp.add_argument(
        "--out", required=True, help="Artifacts output folder (e.g. artifacts)"
    )
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser(
        "restore", parents=[common], help="Apply a curve (default last_good) and exit."
    )
    sp.add_argument(
        "--curve",
        default=None,
        help="Curve csv to apply (default: last_good_curve.csv in --out)",
    )
    sp.set_defaults(func=cmd_restore)

    sp = sub.add_parser(
        "register-startup",
        help="Register a logon task that auto-applies the last-good curve on boot (Windows only).",
    )
    sp.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    sp.add_argument("--out", required=True, help="Artifacts folder containing last_good_curve.csv")
    sp.add_argument(
        "--startup-delay",
        dest="startup_delay",
        type=int,
        default=15,
        metavar="SECONDS",
        help="Seconds to wait after logon before applying the curve (default: 15).",
    )
    sp.set_defaults(func=cmd_register_startup)

    sp = sub.add_parser(
        "unregister-startup",
        help="Remove the VoltVandal logon startup task (Windows only).",
    )
    sp.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    sp.set_defaults(func=cmd_unregister_startup)

    return p.parse_args()


def main() -> int:
    _install_signal_handlers()

    # --list-profiles: works without a subcommand; handle before full parse.
    if "--list-profiles" in sys.argv:
        if _gpu_profiles is not None:
            _gpu_profiles.list_profiles()
        else:
            eprint("gpu_profiles.py not found. Place it alongside voltvandal.py.")
        return 0

    args = parse_args()
    warn_if_not_admin()

    if args.cmd in ("dump", "run", "restore"):
        if _NVAPI_BACKEND == "native":
            eprint(f"Using native nvapi_curve.py backend.")
        elif args.nvapi_cmd and Path(args.nvapi_cmd).exists():
            eprint(f"Using nvapi-cmd.exe subprocess backend: {args.nvapi_cmd}")
        elif args.nvapi_cmd:
            eprint(f"ERROR: nvapi-cmd not found: {args.nvapi_cmd}")
            return 2
        else:
            eprint("ERROR: nvapi_curve.py not available and --nvapi-cmd not specified.")
            return 2
    if args.cmd == "run":
        # Auto-detect doloMing stress script
        if getattr(args, "doloming", None) == "auto":
            script_dir = Path(__file__).resolve().parent
            auto_path = script_dir / "doloMing" / "stress.py"
            if auto_path.exists():
                args.doloming = str(auto_path)
                eprint(f"Auto-detected doloMing: {args.doloming}")
            else:
                args.doloming = None
                eprint("WARNING: doloMing not found at ./doloMing/stress.py — running without GPU stress.")
                eprint("         Results may not reflect real stability. Install via: .\\setup.ps1")
        elif getattr(args, "doloming", None) in (None, "none", "None", ""):
            args.doloming = None
        if args.doloming and not Path(args.doloming).exists():
            eprint(f"ERROR: doloMing path not found: {args.doloming}")
            return 2
        if args.gpuburn and not Path(args.gpuburn).exists():
            eprint(f"ERROR: gpu-burn path not found: {args.gpuburn}")
            return 2
        if args.mode == "vlock":
            if not getattr(args, "target_voltage_mv", 0):
                eprint("ERROR: --mode vlock requires --target-voltage-mv <mV>")
                return 2
        if args.mode != "vlock" and args.bin_min_mv > args.bin_max_mv:
            eprint("ERROR: bin-min-mv must be <= bin-max-mv")
            return 2
        if args.max_steps <= 0:
            eprint("ERROR: max-steps must be > 0")
            return 2

    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        eprint("\nInterrupted by user.")
        return 130
    except Exception as ex:
        eprint(f"\nERROR: {ex}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
