import sys

import pytest

from voltvandal.cli import create_parser, parse_args


def test_top_level_help_includes_run_flag_summary_and_safe_example():
    text = create_parser().format_help()
    assert "Run Flags:" in text
    assert "--fan-speed-pct" in text
    assert "--gpu-throttle-temp-c" in text
    assert "--mvscan-objective" in text
    assert "mvscan" in text
    assert "Safe Example" in text
    assert "python voltvandal.py run --mode uv" in text


@pytest.mark.parametrize("arguments", [
    ["run", "--mode", "uv", "--auto-plan"],
    ["run", "--mode", "vlock", "--auto-plan", "--stress-timeout", "120"],
])
def test_auto_plan_rejects_unsupported_combinations(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["voltvandal"] + arguments)
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_power_limit_max_is_exclusive_with_percent():
    parser = create_parser()
    args = parser.parse_args(["run", "--mode", "vlock", "--power-limit-max"])
    assert args.power_limit_max
    with pytest.raises(SystemExit) as error:
        parser.parse_args([
            "run", "--mode", "vlock", "--power-limit-max", "--power-limit-pct", "110"
        ])
    assert error.value.code == 2


def test_auto_plan_accepts_point_lock(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["voltvandal", "run", "--mode", "vlock", "--auto-plan", "--point-lock"])
    args = parse_args()
    assert args.auto_plan and args.point_lock
