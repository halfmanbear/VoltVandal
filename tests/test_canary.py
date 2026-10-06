import numpy as np

from voltvandal.core.tuning import _parse_doloming_stability
from voltvandal.stress import runner
from voltvandal.stress.canary import int_chain_reference, ramp_duty


def test_int_chain_reference_is_deterministic_and_wraps():
    vals = np.array([0, 1, 0xFFFFFFFF, 12345], dtype=np.uint32)
    a = int_chain_reference(vals, 32)
    assert a.dtype == np.uint32
    assert np.array_equal(a, int_chain_reference(vals, 32))
    assert not np.array_equal(a, int_chain_reference(vals, 33))


def test_ramp_duty_rises_to_target_and_clamps():
    assert ramp_duty(0, 1.0) == 0.25
    assert 0.25 < ramp_duty(5, 1.0) < 1.0
    assert ramp_duty(10, 1.0) == 1.0 and ramp_duty(99, 0.6) == 0.6
    assert ramp_duty(99, 5.0) == 1.0 and ramp_duty(99, 0.0) == 0.05


def test_canary_data_error_output_fails_the_point():
    out = ("[canary] t-  3s | util= 99.0%\n\nTest Summary:\n"
           "Error during test: CANARY_DATA_ERROR kernel=int_chain run_vs_run mismatches=3 loop=7\n"
           "Status              : FAILED (data errors)\n")
    ok, reason = _parse_doloming_stability(out, "canary")
    assert not ok and reason == "DOLOMING_CANARY_STRESS_ERROR"


def test_canary_clean_output_passes():
    out = ("Test Summary:\nStatus              : Successfully maintain\nAverage Utilization : 74.00%\n"
           "Data errors         : 0\n")
    assert _parse_doloming_stability(out, "canary") == (True, None)


def test_canary_command_targets_the_canary_script():
    cmd = runner._build_canary_cmd(gpu=1, seconds=30, util_pct=60)
    assert cmd[-6:] == ["--gpu", "1", "--seconds", "30", "--target-percent", "60"]
    assert cmd[2].endswith("canary.py")
