import pytest

from voltvandal.core.models import SessionState
from voltvandal.hardware.nvapi import curves


@pytest.fixture(autouse=True)
def no_real_curve_writes(monkeypatch):
    """Fail any test that would write a VF curve to the physical GPU."""
    attempts = []

    def blocked(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise RuntimeError(f"test attempted a real NVAPI {name}")
        return call

    monkeypatch.setattr(curves, "apply_curve", blocked("apply_curve"))
    monkeypatch.setattr(curves, "reset_curve", blocked("reset_curve"))
    yield
    assert not attempts, f"real NVAPI writes were attempted: {attempts}"


@pytest.fixture
def mock_state(tmp_path):
    stock_csv = tmp_path / "stock.csv"
    stock_csv.write_text("voltageUV,frequencyKHz\n900000,1800000\n950000,1900000\n")

    last_good_csv = tmp_path / "last_good.csv"
    last_good_csv.write_text("voltageUV,frequencyKHz\n900000,1800000\n950000,1900000\n")

    checkpoint = tmp_path / "session.json"

    return SessionState(
        gpu=0,
        out_dir=str(tmp_path),
        stock_curve_csv=str(stock_csv),
        last_good_curve_csv=str(last_good_csv),
        checkpoint_json=str(checkpoint),
        mode="uv",
        bin_min_mv=800,
        bin_max_mv=1000,
        step_mv=5,
        step_mhz=15,
        max_steps=2,
        stress_seconds=1,
        doloming="mock",
        doloming_mode="simple",
        gpuburn=None,
        poll_seconds=0.1,
        temp_limit_c=90,
        hotspot_limit_c=100,
        hotspot_offset_c=15,
        power_limit_w=400,
        abort_on_throttle=True,
    )
