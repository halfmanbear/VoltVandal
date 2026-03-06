import threading
from pathlib import Path

from voltvandal.stress.runner import run_doloming


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
    assert out_text == "ABORTED_BY_MONITOR"
    assert log_path.exists()
    assert "ABORTED_BY_MONITOR" in log_path.read_text(encoding="utf-8")
