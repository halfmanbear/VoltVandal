import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple
from ..core.utils import is_windows

_ACTIVE_PROCS: Set[subprocess.Popen] = set()
_ACTIVE_PROCS_LOCK = threading.Lock()


def _repo_root() -> Path:
    # .../src/voltvandal/stress/runner.py -> repo root
    return Path(__file__).resolve().parents[3]


def _official_stress_script() -> Path:
    return _repo_root() / "gpu-cpu-stress-tests" / "nvidia_gpu_stress_test.py"


def _canary_script() -> Path:
    return Path(__file__).resolve().with_name("canary.py")


def _build_canary_cmd(gpu: int, seconds: int, util_pct: int = 0) -> List[str]:
    return [sys.executable, "-u", str(_canary_script()), "--gpu", str(gpu),
            "--seconds", str(seconds), "--target-percent", str(util_pct)]


def _build_official_stress_cmd(gpu: int, mode: str, seconds: int, util_pct: int = 0) -> List[str]:
    cmd = [
        sys.executable,
        "-u",  # Stream startup/progress output; CUDA initialization may be slow.
        str(_official_stress_script()),
        "-m",
        mode,
        "-d",
        str(seconds),
        "-g",
        str(gpu),
    ]
    # Safe defaults requested for official runner target utilization.
    if mode == "matrix":
        cmd += ["-t", str(util_pct or 75)]
    elif mode == "ray":
        cmd += ["-t", str(util_pct or 85)]
    return cmd

_FATAL_STRESS_OUTPUT = re.compile(
    r"cuda_?error\w*|illegal memory access|device-side assert|"
    r"unspecified launch failure|launch timeout|driver shutting down|"
    r"Error during (?:stress )?test:",
    re.I,
)


def _reader_thread(
    pipe, lines: List[str], done: threading.Event,
    fatal_event: Optional[threading.Event] = None,
    live_log_path: Optional[Path] = None,
) -> None:
    try:
        log = live_log_path.open("w", encoding="utf-8", errors="replace") if live_log_path else None
        try:
            last_flush = time.monotonic()
            for line in iter(pipe.readline, ""):
                lines.append(line)
                if log:
                    log.write(line)
                if fatal_event and _FATAL_STRESS_OUTPUT.search(line):
                    if log:
                        log.flush()
                    fatal_event.set()
                elif log and time.monotonic() - last_flush >= 1.0:
                    log.flush()
                    last_flush = time.monotonic()
        finally:
            if log:
                log.close()
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
        import subprocess as sp
        creationflags = sp.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.add(p)
    return p


def _untrack_process_if_done(p: subprocess.Popen) -> None:
    if p.poll() is None:
        return
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.discard(p)


def terminate_all_active_processes() -> int:
    """
    Best-effort termination of any tracked stress subprocesses.
    Returns the number of processes that were still active and targeted.
    """
    with _ACTIVE_PROCS_LOCK:
        procs = list(_ACTIVE_PROCS)
    if not procs:
        return 0

    targeted = 0
    for p in procs:
        if p.poll() is None:
            targeted += 1
            terminate_process_tree(p)
        _untrack_process_if_done(p)
    return targeted

def terminate_process_tree(p: subprocess.Popen, gentle_seconds: float = 1.0) -> None:
    if p.poll() is not None:
        _untrack_process_if_done(p)
        return
    if is_windows():
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
            try:
                p.wait(timeout=2.0)
            except Exception:
                pass
            _untrack_process_if_done(p)
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
    try:
        p.wait(timeout=2.0)
    except Exception:
        pass
    _untrack_process_if_done(p)


def _write_stress_log(log_path: Path, output_lines: List[str], trailer: Optional[str] = None) -> str:
    out_text = "".join(output_lines)
    if trailer:
        if out_text and not out_text.endswith("\n"):
            out_text += "\n"
        out_text += f"{trailer}\n"
    log_path.write_text(out_text, encoding="utf-8", errors="replace")
    return out_text

def run_doloming(
    doloming_path: Optional[str],
    gpu: int,
    mode: str,
    seconds: int,
    workdir: Optional[Path],
    log_path: Path,
    abort_event: threading.Event,
    manual_recovery_event: threading.Event,
    interrupted_event: threading.Event,
    stress_timeout: Optional[int] = None,
    max_freq_mhz: int = 0,
    util_pct: int = 0,
) -> Tuple[int, str]:
    if doloming_path and doloming_path not in ("auto", "integrated"):
        exe_is_py = doloming_path.lower().endswith(".py")
        cmd = (
            [sys.executable, "-u", doloming_path, "--mode", mode, "--seconds", str(seconds)]
            if exe_is_py
            else [doloming_path, "--mode", mode, "--seconds", str(seconds)]
        )
    elif mode == "canary":
        cmd = _build_canary_cmd(gpu=gpu, seconds=seconds, util_pct=util_pct)
    else:
        stress_script = _official_stress_script()
        if not stress_script.exists():
            out_text = _write_stress_log(
                log_path,
                [],
                "OFFICIAL_STRESS_SCRIPT_NOT_FOUND",
            )
            return (1, out_text)
        cmd = _build_official_stress_cmd(gpu=gpu, mode=mode, seconds=seconds, util_pct=util_pct)

    p = start_process(cmd, cwd=workdir)
    output_lines: List[str] = []
    reader_done = threading.Event()
    fatal_output = threading.Event()
    reader = threading.Thread(
        target=_reader_thread,
        args=(p.stdout, output_lines, reader_done, fatal_output, log_path),
        daemon=True,
    )
    reader.start()
    deadline: Optional[float] = time.time() + stress_timeout if stress_timeout else None
    # Matrix setup can spend tens of seconds allocating and compiling CUDA
    # kernels before its next complete output line. The hard stress timeout
    # remains the upper bound for a genuinely stuck process.
    no_output_timeout: float = float(max(90, min(180, seconds * 2)))
    last_output_line_count = 0
    last_output_ts = time.time()
    exit_code: Optional[int] = None
    exit_reason: Optional[str] = None
    interrupted_by_user = False
    try:
        while not reader_done.is_set():
            cur_count = len(output_lines)
            if cur_count != last_output_line_count:
                last_output_line_count = cur_count
                last_output_ts = time.time()
            if manual_recovery_event.is_set():
                terminate_process_tree(p)
                exit_code = 997
                exit_reason = "MANUAL_RECOVERY_REQUEST"
                break
            if interrupted_event.is_set():
                terminate_process_tree(p)
                interrupted_by_user = True
                exit_reason = "INTERRUPTED_BY_USER"
                break
            if fatal_output.is_set():
                terminate_process_tree(p)
                exit_code = 995
                exit_reason = "FATAL_STRESS_OUTPUT"
                break
            if abort_event.is_set():
                terminate_process_tree(p)
                exit_code = 999
                exit_reason = "ABORTED_BY_MONITOR"
                break
            if deadline and time.time() > deadline:
                terminate_process_tree(p)
                exit_code = 998
                exit_reason = "STRESS_TIMEOUT"
                break
            if time.time() - last_output_ts > no_output_timeout:
                terminate_process_tree(p)
                exit_code = 996
                exit_reason = "NO_OUTPUT_TIMEOUT"
                break
            reader_done.wait(timeout=0.25)
    finally:
        reader.join(timeout=3.0)
    try:
        p.wait(timeout=2.0)
    except Exception:
        pass
    _untrack_process_if_done(p)

    out_text = _write_stress_log(log_path, output_lines, exit_reason)
    if interrupted_by_user:
        raise KeyboardInterrupt("User pressed Ctrl+C")
    if exit_code is not None:
        return (exit_code, out_text)
    return (p.returncode or 0, out_text)

def run_gpuburn(
    gpuburn_path: str,
    seconds: int,
    workdir: Optional[Path],
    log_path: Path,
    abort_event: threading.Event,
    manual_recovery_event: threading.Event,
    interrupted_event: threading.Event,
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
    no_output_timeout: float = float(max(20, min(90, seconds * 2)))
    last_output_line_count = 0
    last_output_ts = time.time()
    exit_code: Optional[int] = None
    exit_reason: Optional[str] = None
    interrupted_by_user = False
    try:
        while not reader_done.is_set():
            cur_count = len(output_lines)
            if cur_count != last_output_line_count:
                last_output_line_count = cur_count
                last_output_ts = time.time()
            if manual_recovery_event.is_set():
                terminate_process_tree(p)
                exit_code = 997
                exit_reason = "MANUAL_RECOVERY_REQUEST"
                break
            if interrupted_event.is_set():
                terminate_process_tree(p)
                interrupted_by_user = True
                exit_reason = "INTERRUPTED_BY_USER"
                break
            if abort_event.is_set():
                terminate_process_tree(p)
                exit_code = 999
                exit_reason = "ABORTED_BY_MONITOR"
                break
            if deadline and time.time() > deadline:
                terminate_process_tree(p)
                exit_code = 998
                exit_reason = "STRESS_TIMEOUT"
                break
            if time.time() - last_output_ts > no_output_timeout:
                terminate_process_tree(p)
                exit_code = 996
                exit_reason = "NO_OUTPUT_TIMEOUT"
                break
            reader_done.wait(timeout=0.25)
    finally:
        reader.join(timeout=3.0)
    try:
        p.wait(timeout=2.0)
    except Exception:
        pass
    _untrack_process_if_done(p)

    out_text = _write_stress_log(log_path, output_lines, exit_reason)
    if interrupted_by_user:
        raise KeyboardInterrupt("User pressed Ctrl+C")
    if exit_code is not None:
        return (exit_code, exit_reason or "", False)
    m = re.search(r"errors?\s*[:=]\s*(\d+)", out_text, re.I)
    ok = int(m.group(1)) == 0 if m else (False if re.search(r"\bfail(ed)?\b|\berror\b", out_text, re.I) else True)
    return (p.returncode or 0, out_text, ok)
