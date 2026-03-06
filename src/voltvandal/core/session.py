import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Tuple
from .models import SessionState
from .utils import now_utc_iso

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
    # Backward compatibility: field renamed in March 2026.
    if "gpu_throttle_temp_c" not in data and "gpu_target_temp_c" in data:
        data["gpu_throttle_temp_c"] = data["gpu_target_temp_c"]
    
    # Get all field names from the dataclass
    all_fields = {f.name for f in fields(SessionState)}
    
    # Filter out data that isn't in the dataclass (for forward compatibility)
    filtered = {k: v for k, v in data.items() if k in all_fields}
    
    # Create the instance. Dataclass will use defaults for missing fields
    # that have them, but will still fail if a required field (no default) is missing.
    try:
        return SessionState(**filtered)
    except TypeError as e:
        raise ValueError(f"Failed to load session: {e}. The checkpoint file may be corrupted or from an incompatible version.") from e
