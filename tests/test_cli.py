from voltvandal.cli import create_parser


def test_top_level_help_includes_run_flag_summary_and_safe_example():
    text = create_parser().format_help()
    assert "Run Flags:" in text
    assert "--fan-speed-pct" in text
    assert "--gpu-throttle-temp-c" in text
    assert "--mvscan-objective" in text
    assert "mvscan" in text
    assert "Safe Example" in text
    assert "python voltvandal.py run --mode uv" in text
