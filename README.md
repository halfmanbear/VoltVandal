# VoltVandal

VoltVandal is a professional, safety-first CLI tool for NVIDIA GPU undervolting and overclocking.

## Features

- **Automated Tuning:** Multiple modes including `uv` (undervolting), `oc` (overclocking), and the powerful `vlock` (voltage-lock) mode.
- **Safety-First:** Auto-revert to last-known-good settings on crashes or instability.
- **Detailed Monitoring:** Real-time GPU telemetry via NVML and NvAPI.
- **Stress Integration:** Built-in stress workloads using Cupy or external tools like `gpu-burn`.
- **Modern Structure:** Modular, extensible Python package structure.

## Installation

```bash
# Recommendation: use a virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install with dependencies
pip install .
```

Alternatively, run without installation:
```bash
python voltvandal.py run --mode vlock --gpu-profile rtx40
```

## Usage

### 1. View Available Profiles
```bash
python voltvandal.py profiles
```

### 2. Run a vlock session
```bash
python voltvandal.py run --mode vlock --gpu-profile rtx30 --gpu 0
```

Add `--auto-plan` to select a contiguous lower-voltage sweep automatically.
It runs a short stock-curve probe in all four built-in stress modes, uses only
measured voltage samples collected under load, and extends the sweep if later
tests repeatedly reach lower voltage points. A lower point needs at least three
adjacent loaded measured samples spanning two seconds within 12.5 mV,
so a single transition sample
does not expand the sweep. It leaves lower, unselected points unchanged
and never interpolates between tested points. If measured coverage is
insufficient, it selects the full lower-bin sweep. Each Phase 2 candidate
must then show at least three loaded samples and 80% measured voltage-and-clock
coverage at its target point. An unexercised bin is reported as inconclusive,
reverted, and stops the sweep; it is never promoted as a pass. This check
also applies without `--auto-plan` or `--point-lock`. A longer test across
the available stress modes checks a fully covered assembled curve before
completion. A failed final test restores the stock curve. The chosen range
and probe progress are saved in `session.json` for `resume`. `--auto-plan`
cannot be combined with
experimental `--point-lock`.
Vlock saves the active candidate before each stress test. If the process or OS
stops unexpectedly, `resume` refuses to retest that candidate automatically;
an ordinary Ctrl+C still resets the board and permits resume.

```powershell
python voltvandal.py run --mode vlock --gpu-profile rtx30 --gpu 0 --auto-plan --out artifacts-auto
```

Use a new `--out` directory for each `run`. Reusing a directory with an
existing session or baseline is refused so stale curves and logs cannot be
mistaken for a fresh baseline. `resume` is only for continuing a valid session.

Add `--power-limit-max` to apply the selected GPU's maximum power limit as
reported by NVML. The choice is saved in the session checkpoint and reapplied
by `resume`. For example, GPU 0 can be tuned with:

```powershell
python voltvandal.py run --mode vlock --gpu-profile rtx30 --gpu 0 --auto-plan --power-limit-max --out artifacts-auto-max
```

`--power-limit-max` and `--power-limit-pct` are mutually exclusive. The
separate `--power-limit-w` option sets the telemetry abort threshold in watts;
it does not set the board's power limit.

### 3. Resume a session
```bash
python voltvandal.py resume --out artifacts
```

During tuning, `artifacts/logs/telemetry.csv` records `measured_voltage_mv`
from a read-only NVAPI current-voltage call when the driver supports it.
`voltage_source` is `nvapi_measured`, `curve_estimate`, or `unavailable`.
Only `measured_voltage_mv` should be used to determine which voltage points
were actually observed. Existing rows from older logs are marked
`legacy_unknown` when the CSV is upgraded.

### Experimental point testing

`python voltvandal.py point-lock-info --gpu 0` checks the point-lock getter and
current-voltage telemetry without changing settings. Add `--point-lock` to a
new `run --mode vlock` session to request and verify each tested voltage point.
See [point tests and recovery](doc/point-lock.md) for coverage requirements,
API evidence, hardware-validation limits and recovery instructions.

### Anchor mode and crash safety
`--mode anchor` tests a few voltage points, interpolates the rest and verifies the result. It is designed not to look for the point where your GPU hangs:

- Gains rise in single steps from a conservative ceiling (`safe_cap_mhz` in the profile, 150 MHz without one; `--max-gain-mhz` overrides it and `--aggressive` lifts it to `step-mhz x max-steps`). The first failure ends the search at that voltage and the last pass minus a 4-step margin is used. Anchor mode stresses with the `canary` workload, which checks its own results so a failing point shows up as a data error instead of a hang.
- Every point is journalled to `fault_map.json` before it is applied. If the machine crashes, the point is blocked at the next start and never tried again.
- A watchdog reads the Windows event log while a test runs and stops the load on the first NVIDIA/WHEA/TDR event.
- `--anchor-finish` (on `run` or `resume`) runs no new search and builds the curve from gains already proven.

No user-mode tool can guarantee against a bluescreen: a GPU that hangs cannot be reset through NVAPI, and Windows may bugcheck about a minute later. Save your work before tuning, and never apply a tuned curve automatically at boot until it has soaked.

## Structure

- `src/voltvandal/core`: Core logic, session management, and tuning algorithms.
- `src/voltvandal/hardware`: NVAPI and NVML interaction, GPU profiles.
- `src/voltvandal/stress`: Stress testing workloads and runner.
- `src/voltvandal/ui`: CLI and plotting logic.

## Disclaimer

Undervolting and overclocking can lead to system instability or hardware damage. Use at your own risk.
