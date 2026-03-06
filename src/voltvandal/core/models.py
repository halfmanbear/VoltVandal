from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

@dataclass
class CurvePoint:
    voltage_uv: int
    freq_khz: int

@dataclass
class CandidateResult:
    ok: bool
    reason: str
    telemetry_max_temp_c: Optional[int] = None
    telemetry_max_power_w: Optional[float] = None
    telemetry_any_throttle: Optional[bool] = None
    stress_exit_codes: Optional[Dict] = None
    metrics: Optional[Dict] = None

@dataclass
class SessionState:
    gpu: int
    out_dir: str

    stock_curve_csv: str
    last_good_curve_csv: str
    checkpoint_json: str

    mode: str  # "uv" | "oc" | "hybrid" | "vlock" | "mvscan"
    bin_min_mv: int
    bin_max_mv: int

    # steps
    step_mv: int
    step_mhz: int
    max_steps: int

    # stress
    stress_seconds: int
    doloming: Optional[str]
    doloming_mode: str
    gpuburn: Optional[str]

    # monitor thresholds
    poll_seconds: float
    temp_limit_c: int
    hotspot_limit_c: Optional[int]
    hotspot_offset_c: int
    power_limit_w: float
    abort_on_throttle: bool

    # multi-mode stress (runs each doloMing mode in sequence per step)
    doloming_modes: str = ""          # comma-separated modes, e.g. "ray,matrix,frequency-max"; empty = use doloming_mode
    multi_stress_seconds: int = 30    # per-mode duration when doloming_modes is set

    # progress
    current_step: int = 0
    stress_timeout: Optional[int] = None  # hard wall-clock cap on each stress run (seconds)
    current_offset_mv: int = 0
    current_offset_mhz: int = 0

    # hybrid mode state (survives resume)
    hybrid_phase: str = "uv"           # "uv" or "oc"
    hybrid_locked_mv: int = 0          # UV offset locked after phase 1
    hybrid_oc_start_step: int = 0      # step number where OC phase began

    # vlock mode state (survives resume)
    vlock_target_mv: int = 0           # user-supplied anchor voltage
    vlock_anchor_freq_khz: int = 0     # peak freq confirmed stable in phase 1
    vlock_uv_offset_mv: int = 0        # last confirmed UV offset (for display / legacy)
    vlock_uv_bin_idx: int = -1         # Phase 2 outer loop: which sub-anchor bin is being tuned
                                       # -1 = not started; counts down from anchor_idx-1 to 0
    vlock_p2_current_gain_khz: int = 0 # Phase 2: gain currently being tested for the active bin
                                       # 0 = use full oc_gain on next bin entry
    vlock_phase: str = "oc"            # "oc" | "uv" | "done"
    vlock_oc_base_freq_khz: int = 0    # freq floor found by floor-search (0 = use stock)
    vlock_start_freq_mhz: int = 0     # Phase 1 OC search start frequency (MHz), 0 = base/stock
    vlock_last_fail_step: int = -1    # Phase 1 coarse->fine boundary (failing coarse step, -1 = coarse mode)
    mvscan_objective: str = "balanced"  # balanced | max-clock | min-cap

    # power limit
    power_limit_pct: int = 100        # % of GPU default TDP to apply before run (100 = unchanged)
    gpu_throttle_temp_c: int = 0      # real GPU throttle target temp via nvidia-smi -gtt (0 = unchanged)
    gpu_throttle_temp_restore_c: int = 0  # captured pre-run target temp to restore on emergency reset
    fan_mode: str = "auto"            # requested fan mode: auto|manual
    fan_speed_pct: int = 0            # requested manual fan speed percent

    # display
    live_display: bool = True         # print live GPU metrics line during stress
    no_plot: bool = False             # skip post-run VF curve PNG generation

    # manual crash recovery hotkey
    recovery_hotkey: str = "ctrl+shift+f12"
    recovery_hotkey_enabled: bool = True

    # bookkeeping
    started_utc: str = ""
    updated_utc: str = ""

@dataclass
class MonitorSnapshot:
    temp_c: int
    hotspot_c: float
    vram_junction_c: Optional[float]
    power_w: float
    clock_mhz: int
    mem_clock_mhz: int
    util_gpu: int
    throttle_reasons: int
    voltage_mv: Optional[int] = None
    voltage_estimated: bool = False
    pstate: Optional[int] = None
    perf_decrease: Optional[int] = None
    topo_gpu_mw: Optional[int] = None
    topo_total_mw: Optional[int] = None
