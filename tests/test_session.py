import json

from voltvandal.core.session import load_session, session_paths


def test_load_session_migrates_gpu_target_temp_field(tmp_path):
    stock, last_good, checkpoint = session_paths(tmp_path)
    payload = {
        "gpu": 0,
        "out_dir": str(tmp_path),
        "stock_curve_csv": str(stock),
        "last_good_curve_csv": str(last_good),
        "checkpoint_json": str(checkpoint),
        "mode": "uv",
        "bin_min_mv": 850,
        "bin_max_mv": 1050,
        "step_mv": 5,
        "step_mhz": 15,
        "max_steps": 1,
        "stress_seconds": 10,
        "doloming": "integrated",
        "doloming_mode": "simple",
        "gpuburn": None,
        "poll_seconds": 1.0,
        "temp_limit_c": 83,
        "hotspot_limit_c": 95,
        "hotspot_offset_c": 15,
        "power_limit_w": 400.0,
        "abort_on_throttle": True,
        "gpu_target_temp_c": 82,
    }
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    state = load_session(tmp_path)
    assert state.gpu_throttle_temp_c == 82
