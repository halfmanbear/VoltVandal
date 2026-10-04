---

# 🔎 Codebase Review Summary

You explored the full VoltVandal codebase and identified architectural strengths, gaps, and upgrade opportunities. The main areas reviewed:

* `monitor.py` – NVML monitoring, hotspot handling, live display
* `nvapi.py` – Native NVAPI telemetry (hotspot, voltage, power, P-state)
* `models.py` – `SessionState` config and persistence
* `main.py` – CLI wiring and session creation
* `tuning.py` – Core tuning loops (`evaluate_candidate`, `mvscan`, `vlock`)
* Stress runners (`doloming`, `gpuburn`)
* Tests structure

---

# ✅ Key Findings

## 🔥 Hotspot Handling

* Fallback hotspot uses `edge_temp + hotspot_offset_c` (default 15°C).
* **Actual NVAPI hotspot reading exists** and overrides fallback if available.
* You want to:

  * **Always use real NVAPI hotspot when available**
  * Remove the hardcoded 15°C assumption
  * Potentially auto-calibrate offset during warmup

---

## 📊 Plotting Gap

* `SessionState.no_plot` exists.
* `--no-plot` flag appears in docs.
* **No plotting implementation exists.**
* `matplotlib` is already a dependency.

You correctly identified this as the most obvious missing feature.

---

## ⚡ mvscan Weakness

### Current Behavior

* Stops after **3 consecutive failures**
* Any `result.ok == False` increments the failure counter
* Does not distinguish:

  * True instability
  * Transient stress timeouts
  * Monitor aborts
  * Tool hiccups

### Risk

A single transient timeout can prematurely terminate a monotonic scan.

---

## 🧪 Stress Execution Model

`evaluate_candidate()`:

* Applies curve
* Runs stress
* Monitors temps/power/throttle
* Returns structured `CandidateResult`

Stress exit codes:

* `996` → NO_OUTPUT_TIMEOUT
* `998` → STRESS_TIMEOUT
* `999` → ABORTED_BY_MONITOR
* `CUDA_RUNTIME_ERROR` parsed from output
* gpuburn parses error count

There is enough signal to distinguish **hard instability vs transient termination.**

---

# 🎯 Your Final Decisions

You explicitly requested:

### ❌ Do NOT change:

* `--gpu-throttle-temp-c` (keep nvidia-smi integration)
* vlock Phase 1 starting frequency

### ✅ DO implement:

1. VF curve plot generation
2. Adaptive stress duration (probe phase)
3. mvscan confirm-runs option
4. Real NVAPI hotspot reading (no hardcoded assumption)
5. mvscan transient vs real failure distinction
6. Completion summary report

---

# 🛠 Planned Implementation (High-Level)

## 1️⃣ Post-Run VF Plot (High Priority)

* New plotting module using matplotlib
* Overlay:

  * Stock curve
  * Last-good curve
  * Pass/fail markers per step (from `steps.jsonl`)
* Save as `vf_curve.png`
* Respect `--no-plot`

**Impact:** Massive usability improvement.

---

## 2️⃣ Adaptive Stress Duration

* Add short probe run (e.g. 30s)
* If fails hard → skip full run
* If passes → run full duration
* Integrated into `evaluate_candidate_confident`

**Impact:** Large session time reduction in mvscan.

---

## 3️⃣ mvscan Confirm-Runs

Add CLI option:

```
--confirm-runs N
```

* Re-test borderline candidates
* Especially those with throttle or non-zero errors
* Improves boundary detection reliability

---

## 4️⃣ Real NVAPI Hotspot Enforcement

* Ensure `get_thermal_sensors()` hotspot is always used if available
* Fallback only if NVAPI truly unavailable
* Remove assumption-based 15°C logic

---

## 5️⃣ Smarter mvscan Failure Classification

Introduce logic:

Count toward 3-fail streak only if:

* CUDA runtime errors
* Unstable status
* gpuburn error count > 0
* Driver reset

Do NOT count:

* STRESS_TIMEOUT
* NO_OUTPUT_TIMEOUT
* Monitor abort due to limits (unless repeated)

Prevents premature termination.

---

## 6️⃣ End-of-Session Summary

Print structured report:

* Voltage delta vs stock
* Clock gain
* Best P95
* Total runtime
* Final curve saved location

Optional: write `report.txt`.

---

# 🧠 Overall Assessment

VoltVandal architecture is already strong:

* Clean separation of tuning logic
* Well-structured monitoring
* Good stress abstraction
* Test coverage in place

The improvements you selected focus on:

* ✔ Usability
* ✔ Correctness at the stability boundary
* ✔ Runtime efficiency
* ✔ Telemetry accuracy

Not architectural overhauls — just smart refinement.

