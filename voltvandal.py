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

    # progress
    current_step: int = 0
    stress_timeout: Optional[int] = None  # hard wall-clock cap on each stress run (seconds)
    current_offset_mv: int = 0
    current_offset_mhz: int = 0

    # hybrid mode state (survives resume)
    hybrid_phase: str = "uv"           # "uv" or "oc"
    hybrid_locked_mv: int = 0          # UV offset locked after phase 1
    hybrid_oc_start_step: int = 0      # step number where OC phase began

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
) -> Tuple[int, str]:
    exe_is_py = doloming_path.lower().endswith(".py")
    base_cmd = [sys.executable, doloming_path] if exe_is_py else [doloming_path]

    candidates = [
        base_cmd + ["--mode", mode, "--seconds", str(seconds)],
        base_cmd + [mode, str(seconds)],
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
                    return (999, "ABORTED_BY_MONITOR")
                if deadline and time.time() > deadline:
                    terminate_process_tree(p)
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
    hotspot_c: float  # real NvAPI hotspot, or estimated (edge + offset)
    vram_junction_c: Optional[float]  # real NvAPI VRAM junction, or None
    power_w: float
    clock_mhz: int
    util_gpu: int
    throttle_reasons: int


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
    ):
        self.gpu_index = gpu_index
        self.poll_seconds = poll_seconds
        self.temp_limit_c = temp_limit_c
        self.hotspot_limit_c = hotspot_limit_c
        self.hotspot_offset_c = hotspot_offset_c
        self.power_limit_w = power_limit_w
        self.abort_on_throttle = abort_on_throttle
        self.log_csv = log_csv

        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.max_temp: Optional[int] = None
        self.max_hotspot: Optional[int] = None
        self.max_power: Optional[float] = None
        self.any_throttle: bool = False
        self.last_snapshot: Optional[MonitorSnapshot] = None
        self._consecutive_errors: int = 0

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
                        "util_gpu",
                        "throttle_reasons",
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
                util = int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)
                throttle = int(
                    pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle)
                )

                # Read real hotspot/VRAM temps via NvAPI if available
                hotspot: float = temp + self.hotspot_offset_c  # fallback estimate
                vram_junc: Optional[float] = None
                if _nvapi_native is not None:
                    try:
                        thr = _nvapi_native.get_thermal_sensors(self.gpu_index)
                        if thr["hotspot_c"] is not None:
                            hotspot = thr["hotspot_c"]
                        vram_junc = thr.get("vram_junction_c")
                    except Exception:
                        pass  # fall back to estimate

                snap = MonitorSnapshot(temp, hotspot, vram_junc, power, clock, util, throttle)
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

                vram_str = f"{vram_junc:.1f}" if vram_junc is not None else ""
                with self.log_csv.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [now_utc_iso(), temp, f"{hotspot:.1f}", vram_str,
                         f"{power:.1f}", clock, util, throttle]
                    )

                if temp >= self.temp_limit_c:
                    self.abort_event.set()
                if self.hotspot_limit_c is not None and hotspot >= self.hotspot_limit_c:
                    self.abort_event.set()
                if power >= self.power_limit_w:
                    self.abort_event.set()
                if self.abort_on_throttle and throttle != 0:
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
# Candidate evaluation
# ----------------------------
def evaluate_candidate(
    state: SessionState, candidate_csv: Path, candidate_label: str
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
        )
        monitor.start()
        abort_event = monitor.abort_event

    stress_exit_codes = {}

    if state.doloming:
        dololog = logs_dir / f"{candidate_label}_doloming_{state.doloming_mode}.log"
        rc, out_text = run_doloming(
            state.doloming,
            state.doloming_mode,
            state.stress_seconds,
            None,
            dololog,
            abort_event,
            stress_timeout=state.stress_timeout,
        )
        stress_exit_codes["doloming"] = rc
        if rc != 0:
            if monitor:
                monitor.stop()
            return CandidateResult(
                False,
                f"DOLOMING_RC_{rc}",
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
                "DOLOMING_OUTPUT_ERROR_KEYWORD",
                monitor.max_temp if monitor else None,
                monitor.max_power if monitor else None,
                monitor.any_throttle if monitor else None,
                stress_exit_codes,
            )

    if state.gpuburn:
        burnlog = logs_dir / f"{candidate_label}_gpuburn.log"
        rc, out_text, parsed_ok = run_gpuburn(
            state.gpuburn, state.stress_seconds, None, burnlog, abort_event,
            stress_timeout=state.stress_timeout,
        )
        stress_exit_codes["gpuburn"] = rc
        if rc != 0:
            if monitor:
                monitor.stop()
            return CandidateResult(
                False,
                f"GPUBURN_RC_{rc}",
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
                "GPUBURN_UNCONFIRMED_ZERO_ERRORS",
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
            return CandidateResult(
                False,
                "MONITOR_ABORT_THRESHOLD",
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
            print(
                "Stopped after failure. You can adjust step sizes/limits and resume (it will retry this step)."
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
        started_utc=now_utc_iso(),
        updated_utc=now_utc_iso(),
    )


def cmd_run(args: argparse.Namespace) -> None:
    state = build_session_from_args(args)
    save_session(state)
    print(f"Wrote session checkpoint: {state.checkpoint_json}")
    run_session(state)


def cmd_resume(args: argparse.Namespace) -> None:
    state = load_session(Path(args.out))
    print(f"Loaded session: {state.checkpoint_json}")
    run_session(state)


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
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
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
    sp.add_argument("--mode", choices=["uv", "oc", "hybrid"], default="uv")
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
        "--gpuburn", type=str, default=None, help="Path to gpu-burn exe (optional)"
    )
    sp.add_argument(
        "--stress-timeout",
        dest="stress_timeout",
        type=int,
        default=180,
        help="Hard wall-clock timeout (seconds) for each stress run. Kills hung stress tools. Default: 180",
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

    return p.parse_args()


def main() -> int:
    _install_signal_handlers()
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
        if args.bin_min_mv > args.bin_max_mv:
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
