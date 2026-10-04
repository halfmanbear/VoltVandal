# Experimental voltage-point tests

VoltVandal can now request a voltage bin during `vlock` candidate tests. This is
opt-in: existing sessions and commands retain their previous curve-only behaviour.
It is the operating-point validation foundation for a faster optimiser, not a new
multi-anchor search algorithm or proof of everyday stability.

## Usage

Read-only capability check (does not call a setter):

```powershell
python voltvandal.py point-lock-info --gpu 0
```

Add `--point-lock` to a new `run --mode vlock` command with your chosen profile,
voltage, workload and limits. It is saved in `session.json` and retained on resume.
The switch is rejected for other modes. Phase 1 targets the chosen anchor; phase 2
targets the particular lower bin being edited. The search order remains unchanged.

After a read-only capability/identity check, each candidate applies its curve,
reads and journals the existing lock, applies the point request, verifies the lock
getter, runs the workload, and restores the previous lock. The journal covers the
temporary lock; the existing tuning recovery handles the curve.

The monitor uses the separate, driver-reported current voltage and NVML clock;
neither the requested lock nor voltage estimated from a curve counts as observed
coverage. Each workload needs at least three matching loaded samples and 80%
coverage. Clock tolerance is 15 MHz. Voltage tolerance is 49% of the nearest
neighbour-bin spacing, capped at 10 mV (3 mV for a single-bin fixture).
These are initial screening thresholds, not established stability criteria.
Sampling is sequential and does not prove exact instantaneous bin residency or
physical ADC accuracy. Fast load transitions require additional validation.

Coverage is saved per workload in `logs/*_point.jsonl`. Timeouts, missing voltage
or inadequate coverage stop the point test as inconclusive without advancing its
checkpoint or promoting it. Three successive voltage-read errors abort the
workload. Actual workload failures still use the existing tuning failure logic.
Validate the final assembled curve under varied workloads with temporary locks
released before treating it as an everyday configuration.

## Recovery and isolation

Native lock and voltage calls run in short-lived subprocesses with deadlines.
Only known structures are used; no signature fuzzing is performed. Setter
resolution alone is not evidence that a GPU supports a write. A lock can still
be constrained by other driver policies.

NVAPI and NVML bus identities must agree. Duplicate NVAPI bus IDs and nonzero
NVML PCI domains are rejected by this first implementation. Recovery also checks
the NVML UUID. It restores the original core lock entry, preserving other current
entries. A pre-existing manual lock is restored rather than silently released.

The original state is flushed to `point_lock_recovery.json` before the setter.
Restoration runs after success, failure, readback mismatch and Ctrl+C. A failed
restoration or forcibly terminated parent leaves the journal for explicit recovery:

```powershell
python voltvandal.py recover-point-lock --out artifacts
```

New runs/resume using that output directory are blocked until recovery succeeds.
The journal does not recover the curve, fan or power controls. Do not run another
tuner concurrently; this is not a system-wide exclusive GPU lock. A reboot or
driver reset may invalidate a stored control snapshot; failed recovery remains
visible and the journal is retained.

## API evidence

| Capability | IDs | Evidence |
| --- | --- | --- |
| Point-lock getter/setter | `0xE440B867` / `0x39442CFB` | Local key table names these `PerfClientLimitsGetStatus/SetStatus`; nvapioc calls them `GetClockBoostLock/SetClockBoostLock`. |
| Core curve status/control/write | `0x21537AD4` / `0x23F1B133` / `0x0733E009` | Local `ClockClientClkVfPoints*` names alias VoltVandal's existing curve interfaces. |
| Current voltage | `0x465F9BCF` | NvAPIWrapper `GetCurrentVoltage`, `PrivateVoltageStatusV1`: 76 bytes, microvolts at byte 40. The new path uses this layout without the legacy speculative fallbacks. |
| Rail-policy controls | `0x9DF23CA1` / `0xB9306D9B` | Candidate family only; also named voltage-boost-percent in NvAPIWrapper. IDs alone do not establish mVolt's rail-field layout. Not implemented. |
| Thermal overrides / propagation | `ThermChannel*` / `ClockClkProp*` | Name/behaviour hypotheses only. Not implemented. |

Lock entries are 24 bytes; the structure is 780 bytes with 32 entries. Version 2
is tried first, then version 1 **only** after `NVAPI_INCOMPATIBLE_STRUCT_VERSION`
(-9). Other getter errors stop the operation. Exactly one index-6 entry is required.
Mode 3 requests a manual voltage point; mode 0 is unlocked. Unknown original modes
are rejected. The requested voltage must exist in the driver's exposed core curve.

Primary implementation references:

- [nvapioc main.cpp](https://github.com/Demion/nvapioc/blob/master/Source/main.cpp): version-1 layout and voltage lock routine.
- [NvAPIWrapper lock layout](https://github.com/falahati/NvAPIWrapper/blob/master/NvAPIWrapper/Native/GPU/Structures/PrivateClockBoostLockV2.cs): version-2 layout.
- [NvAPIWrapper voltage layout](https://github.com/falahati/NvAPIWrapper/blob/master/NvAPIWrapper/Native/GPU/Structures/PrivateVoltageStatusV1.cs).
- [NvAPIWrapper interface IDs](https://github.com/falahati/NvAPIWrapper/blob/master/NvAPIWrapper/Native/Helpers/FunctionId.cs).

The local NVIDIA public SDK does not declare these private lock/curve APIs. The
saved key table is dated 2022, and matching names/IDs do not prove mVolt's internal
implementation. Sources were inspected on 2026-10-04.

## Validation performed

Unit tests mock native calls and exercise ABI layout, version negotiation, invalid
entries, readback mismatch, restoration, recovery persistence and per-workload
coverage. Read-only inspection on local GPU 0 accepted lock version 2, reported
mode 0, matched NVAPI/NVML identity, and returned 875 mV at the observation time.
No setter or stress workload was executed as part of this implementation check.
Setter compatibility, sustained coverage and tuning speed remain unmeasured.
