import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple
from ..core.utils import is_windows, eprint

def _reader_thread(pipe, lines: List[str], done: threading.Event) -> None:
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
        import subprocess as sp
        creationflags = sp.CREATE_NEW_PROCESS_GROUP
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
) -> Tuple[int, str]:
    # Search for vv_stress.py relative to this file or root
    stress_script = Path(__file__).resolve().parent / "workloads.py"
    if not stress_script.exists():
         # Fallback to current dir vv_stress.py if we haven't moved it yet
         stress_script = Path("vv_stress.py").resolve()

    if doloming_path and doloming_path not in ("auto", "integrated"):
        exe_is_py = doloming_path.lower().endswith(".py")
        cmd = (
            [sys.executable, doloming_path, "--mode", mode, "--seconds", str(seconds)]
            if exe_is_py
            else [doloming_path, "--mode", mode, "--seconds", str(seconds)]
        )
    else:
        cmd = [
            sys.executable,
            str(stress_script),
            "--mode",
            mode,
            "--seconds",
            str(seconds),
            "--gpu",
            str(gpu),
        ]

    if mode == "frequency-max" and max_freq_mhz > 0:
        cmd += ["--max-freq-mhz", str(max_freq_mhz)]

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
            if manual_recovery_event.is_set():
                terminate_process_tree(p)
                return (997, "MANUAL_RECOVERY_REQUEST")
            if interrupted_event.is_set():
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
    log_path.write_text(out_text, encoding="utf-8", errors="replace")
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
    try:
        while not reader_done.is_set():
            if manual_recovery_event.is_set():
                terminate_process_tree(p)
                return (997, "MANUAL_RECOVERY_REQUEST", False)
            if interrupted_event.is_set():
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
    m = re.search(r"errors?\s*[:=]\s*(\d+)", out_text, re.I)
    ok = int(m.group(1)) == 0 if m else (False if re.search(r"\bfail(ed)?\b|\berror\b", out_text, re.I) else True)
    return (p.returncode or 0, out_text, ok)
