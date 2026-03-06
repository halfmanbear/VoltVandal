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

### 3. Resume a session
```bash
python voltvandal.py resume --out artifacts
```

## Structure

- `src/voltvandal/core`: Core logic, session management, and tuning algorithms.
- `src/voltvandal/hardware`: NVAPI and NVML interaction, GPU profiles.
- `src/voltvandal/stress`: Stress testing workloads and runner.
- `src/voltvandal/ui`: CLI and plotting logic.

## Disclaimer

Undervolting and overclocking can lead to system instability or hardware damage. Use at your own risk.
