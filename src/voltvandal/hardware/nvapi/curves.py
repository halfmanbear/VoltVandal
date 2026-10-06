"""VF-curve dump/apply/reset, with a worker-thread timeout wrapper."""

from __future__ import annotations

import csv
import threading
from pathlib import Path

from .native import (
    _nvapi_init, _get_handle, _read_active_bins, _reset_curve, _get_clock_masks,
    _get_vfp_curve, _get_clock_table, _set_clock_table, _active_core_indices,
)


def _call_with_timeout(func, timeout_seconds: float, *args, **kwargs):
    """
    Run an NVAPI operation on a daemon worker thread and bound wait time.
    This prevents a driver-side deadlock from freezing the CLI indefinitely.
    """
    done = threading.Event()
    result: dict = {}
    error: dict = {}

    def _worker() -> None:
        try:
            result["value"] = func(*args, **kwargs)
        except Exception as ex:
            error["value"] = ex
        finally:
            done.set()

    t = threading.Thread(target=_worker, name=f"nvapi_{func.__name__}", daemon=True)
    t.start()
    if not done.wait(timeout=max(1.0, float(timeout_seconds))):
        raise TimeoutError(
            f"NVAPI call {func.__name__} timed out after {float(timeout_seconds):.1f}s"
        )
    if "value" in error:
        raise error["value"]
    return result.get("value")


def dump_curve(gpu_index: int, out_csv: "str | Path") -> None:
    """
    Read the current VF curve from *gpu_index* and write it to *out_csv*.

    CSV format:
        voltageUV,frequencyKHz
        850000,1830000
        ...

    Raises RuntimeError on any NVAPI failure.
    """
    _nvapi_init()
    handle = _get_handle(gpu_index)

    masks, vfp, idx = _read_active_bins(handle)

    out_csv = Path(out_csv)
    with out_csv.open("w", newline="") as fh:
        fh.write("voltageUV,frequencyKHz\n")
        for i in idx:
            fh.write(f"{vfp.clocks[i].voltageUV},{vfp.clocks[i].frequencyKHz}\n")


def apply_curve(gpu_index: int, in_csv: "str | Path") -> None:
    """
    Apply a VF curve from *in_csv* to *gpu_index*.

    The CSV must have *voltageUV* and *frequencyKHz* columns (same format
    produced by dump_curve).  Frequency values are treated as **absolute**
    targets.  Internally this:

      1. Resets all frequency deltas to zero (driver defaults).
      2. Reads the resulting default absolute frequencies.
      3. Computes per-bin deltas: Δ = target_freq − default_freq.
      4. Writes the delta table back via SetClockBoostTable.

    Only bins whose voltage appears in the CSV are modified; all others
    retain zero delta (driver default).

    Raises RuntimeError on any NVAPI failure or CSV format error.
    """
    in_csv = Path(in_csv)
    csv_by_voltage: dict[int, int] = {}  # voltageUV → target frequencyKHz
    with in_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        norm = {k.lower(): k for k in (reader.fieldnames or [])}
        col_v = norm.get("voltageuv")
        col_f = norm.get("frequencykhz")
        if not col_v or not col_f:
            raise ValueError(
                f"CSV must have voltageUV and frequencyKHz columns: {in_csv}"
            )
        for row in reader:
            csv_by_voltage[int(row[col_v])] = int(row[col_f])

    if not csv_by_voltage:
        raise ValueError(f"No curve points found in {in_csv}")

    _nvapi_init()
    handle = _get_handle(gpu_index)

    # Step 1: zero all deltas → driver restores base frequencies
    _reset_curve(handle)

    # Step 2: read the default absolute frequencies (after reset)
    masks_def, vfp_def, idx_def = _read_active_bins(handle)
    default_by_voltage: dict[int, int] = {
        vfp_def.clocks[i].voltageUV: vfp_def.clocks[i].frequencyKHz
        for i in idx_def
    }

    # Step 3: fetch the delta table (mask already zeroed from step 1)
    masks2 = _get_clock_masks(handle)
    vfp2   = _get_vfp_curve(handle, masks2)
    table  = _get_clock_table(handle, masks2)

    # Step 4: compute and write deltas for CSV-supplied voltages
    for i in _active_core_indices(masks2, vfp2):
        volt = vfp2.clocks[i].voltageUV
        if volt in csv_by_voltage and volt in default_by_voltage:
            table.clocks[i].frequencyDeltaKHz = (
                csv_by_voltage[volt] - default_by_voltage[volt]
            )

    _set_clock_table(handle, table)


def reset_curve(gpu_index: int) -> None:
    """Reset VF curve deltas to driver defaults for the selected GPU."""
    _nvapi_init()
    _reset_curve(_get_handle(gpu_index))


def apply_curve_safe(
    gpu_index: int,
    in_csv: "str | Path",
    timeout_seconds: float = 10.0,
) -> None:
    _call_with_timeout(apply_curve, timeout_seconds, gpu_index, in_csv)


def reset_curve_safe(gpu_index: int, timeout_seconds: float = 10.0) -> None:
    _call_with_timeout(reset_curve, timeout_seconds, gpu_index)
