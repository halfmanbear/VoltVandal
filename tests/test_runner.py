import threading
from pathlib import Path

from voltvandal.stress.runner import _build_official_stress_cmd, run_doloming


def test_official_stress_runs_unbuffered():
    cmd = _build_official_stress_cmd(0, "matrix", 20)
    assert cmd[1] == "-u"
    assert cmd[-2:] == ["-t", "75"]


def test_run_doloming_writes_log_on_monitor_abort(tmp_path: Path):
    stress_script = tmp_path / "dummy_stress.py"
    stress_script.write_text(
        "\n".join(
            [
                "import argparse",
                "import time",
                "ap = argparse.ArgumentParser()",
                "ap.add_argument('--mode')",
                "ap.add_argument('--seconds')",
                "args = ap.parse_args()",
                "print('dummy-start', flush=True)",
                "time.sleep(5)",
            ]
        ),
        encoding="utf-8",
    )

    log_path = tmp_path / "dummy.log"
    abort_event = threading.Event()
    abort_event.set()
    manual_recovery_event = threading.Event()
    interrupted_event = threading.Event()

    rc, out_text = run_doloming(
        doloming_path=str(stress_script),
        gpu=0,
        mode="simple",
        seconds=10,
        workdir=None,
        log_path=log_path,
        abort_event=abort_event,
        manual_recovery_event=manual_recovery_event,
        interrupted_event=interrupted_event,
    )

    assert rc == 999
    assert out_text.rstrip().endswith("ABORTED_BY_MONITOR")
    assert log_path.exists()
    assert "ABORTED_BY_MONITOR" in log_path.read_text(encoding="utf-8")
    assert out_text == log_path.read_text(encoding="utf-8")


def test_run_doloming_stops_on_live_cuda_error(tmp_path: Path):
    stress_script = tmp_path / "fatal_stress.py"
    stress_script.write_text(
        "import time\n"
        "print('CUDA_ERROR_ILLEGAL_ADDRESS: illegal memory access', flush=True)\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    log_path = tmp_path / "fatal.log"
    rc, out_text = run_doloming(
        doloming_path=str(stress_script),
        gpu=0,
        mode="simple",
        seconds=60,
        workdir=None,
        log_path=log_path,
        abort_event=threading.Event(),
        manual_recovery_event=threading.Event(),
        interrupted_event=threading.Event(),
    )
    assert rc == 995
    assert "CUDA_ERROR_ILLEGAL_ADDRESS" in out_text
    assert out_text.rstrip().endswith("FATAL_STRESS_OUTPUT")
    assert log_path.read_text(encoding="utf-8") == out_text
