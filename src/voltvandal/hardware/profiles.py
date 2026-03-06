"""
VoltVandal — GPU series reference profiles
==========================================
Generic conservative starting points for each NVIDIA RTX generation.

These are community-derived baselines drawn from publicly available
overclocking and undervolting data. Every GPU chip is different (silicon
lottery), so treat these as an informed first guess, not a guarantee.

Workflow guidance (vlock mode):
  1. Pick the profile for your RTX generation.
  2. Start with --mode vlock --target-voltage-mv <profile value>.
  3. Let VoltVandal's floor-search + Phase 1 OC find the optimal freq.
  4. Phase 2 UV then sweeps sub-anchor bins lower.
  5. Register the result with `register-startup` so it survives reboots.

Usage:
  python voltvandal.py run --mode vlock --gpu-profile rtx40 --out artifacts ...
  python voltvandal.py --list-profiles

Fields (all values are conservative starting points):
  target_voltage_mv   Anchor voltage for vlock mode (mV)
  step_mhz            OC step size per trial (MHz)
  step_mv             UV step size per trial (mV)
  max_steps           Max trials per phase
  stress_seconds      Duration per stress run (seconds; used in single-mode)
  multi_stress_seconds Per-mode duration when --doloming-mode is a comma-separated list
  temp_limit_c        Edge (GPU die) temperature abort threshold (°C)
  hotspot_limit_c     Hotspot (junction) temperature abort threshold (°C)
  power_limit_w       Power abort threshold (W) — set near card TDP
  bin_min_mv          Voltage bin sweep minimum (uv/oc/hybrid modes)
  bin_max_mv          Voltage bin sweep maximum (uv/oc/hybrid modes)
"""

from __future__ import annotations
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Profile data
# ---------------------------------------------------------------------------
GPU_PROFILES: Dict[str, Dict[str, Any]] = {

    # ── RTX 20xx — Turing (SM 7.5) ─────────────────────────────────────────
    # Cards: RTX 2060 / 2060 Super / 2070 / 2070 Super / 2080 / 2080 Super /
    #        2080 Ti
    # Stock boost range: ~1650–2100 MHz
    # Typical UV sweet spot: 850–950 mV
    # TDP range: 160–280 W (2080 Ti FE 250 W, some OC editions higher)
    "rtx20": {
        "name": "RTX 20xx (Turing)",
        "target_voltage_mv": 900,
        "step_mhz": 10,
        "step_mv": 5,
        "max_steps": 30,
        "stress_seconds": 60,
        "multi_stress_seconds": 40,
        "temp_limit_c": 90,
        "hotspot_limit_c": 95,
        "power_limit_w": 260,
        "bin_min_mv": 800,
        "bin_max_mv": 975,
        "notes": (
            "Turing chips respond well to undervolting. "
            "Try 900 mV first; many 2080/2080 Ti samples stabilise at 875–925 mV. "
            "Boost clock headroom above stock is typically 50–150 MHz in vlock OC phase. "
            "Hotspot sensor may not be real NvAPI on all Turing drivers; "
            "the +15 C edge-offset fallback is used automatically if unavailable."
        ),
    },

    # ── RTX 30xx — Ampere (SM 8.6) ─────────────────────────────────────────
    # Cards: RTX 3060 / 3060 Ti / 3070 / 3070 Ti / 3080 / 3080 Ti /
    #        3090 / 3090 Ti
    # Stock boost range: ~1665–2100 MHz
    # Typical UV sweet spot: 850–975 mV
    # TDP range: 170–450 W (3090 Ti 450 W)
    "rtx30": {
        "name": "RTX 30xx (Ampere)",
        "target_voltage_mv": 890,
        "step_mhz": 15,
        "step_mv": 5,
        "max_steps": 30,
        "stress_seconds": 60,
        "multi_stress_seconds": 40,
        "temp_limit_c": 90,
        "hotspot_limit_c": 95,
        "power_limit_w": 380,
        "bin_min_mv": 450,
        "bin_max_mv": 1000,
        "notes": (
            "Ampere GA102 chips (3080/3090 family) are power-hungry; "
            "set power_limit_w near your card's rated TDP. "
            "GA104 (3070/3060 Ti) is more conservative — 350 W limit is fine. "
            "UV sweet spot is often 875–950 mV; start at 912 mV and let floor-search "
            "find stability if needed. OC headroom is typically 75–150 MHz above stock."
        ),
    },

    # ── RTX 40xx — Ada Lovelace (SM 8.9) ───────────────────────────────────
    # Cards: RTX 4060 / 4060 Ti / 4070 / 4070 Super / 4070 Ti / 4070 Ti Super /
    #        4080 / 4080 Super / 4090
    # Stock boost range: ~2310–2850 MHz
    # Typical UV sweet spot: 900–1000 mV
    # TDP range: 115–450 W (4090 FE 450 W)
    "rtx40": {
        "name": "RTX 40xx (Ada Lovelace)",
        "target_voltage_mv": 950,
        "step_mhz": 15,
        "step_mv": 5,
        "max_steps": 30,
        "stress_seconds": 60,
        "multi_stress_seconds": 40,
        "temp_limit_c": 90,
        "hotspot_limit_c": 95,
        "power_limit_w": 400,
        "bin_min_mv": 850,
        "bin_max_mv": 1062,
        "notes": (
            "Ada delivers real NvAPI hotspot sensor data; the 95 °C hotspot limit "
            "is appropriate (Nvidia's own spec is 100 °C junction). "
            "4090 samples frequently stabilise at 950–987 mV with +100–200 MHz OC. "
            "4070 Ti / 4080 often settle around 937–962 mV. "
            "If baseline fails at 950 mV, let floor-search find the stable freq "
            "then use Phase 2 UV to squeeze further savings below the anchor."
        ),
    },

    # ── RTX 50xx — Blackwell (GB202 / GB203 / GB205 / GB206) ───────────────
    # Cards: RTX 5070 Ti / 5080 / 5090 (and others as they release)
    # Stock boost range: ~2400–2900+ MHz (early data)
    # Typical UV sweet spot: community data still forming — use Ada as guide
    # TDP range: 250–575 W (5090 575 W)
    "rtx50": {
        "name": "RTX 50xx (Blackwell)",
        "target_voltage_mv": 962,
        "step_mhz": 15,
        "step_mv": 5,
        "max_steps": 30,
        "stress_seconds": 60,
        "multi_stress_seconds": 40,
        "temp_limit_c": 90,
        "hotspot_limit_c": 95,
        "power_limit_w": 450,
        "bin_min_mv": 862,
        "bin_max_mv": 1075,
        "notes": (
            "Blackwell community UV data is still maturing. "
            "These values are extrapolated from Ada Lovelace patterns. "
            "5090 TDP is 575 W (FE); set power_limit_w appropriately for your card. "
            "Floor-search is especially useful here — if baseline fails at 962 mV, "
            "the tool will auto-step down to find where your chip is stable. "
            "Update these values as community data for your specific SKU becomes available."
        ),
    },
}

# Short alias → canonical key mapping (e.g. "30" → "rtx30")
_ALIASES: Dict[str, str] = {
    alias: key
    for key in GPU_PROFILES
    for alias in (key, key[3:], key.replace("rtx", "RTX"))
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile(name: str) -> Dict[str, Any]:
    """Return the profile dict for *name*.  Raises KeyError on unknown name."""
    canonical = _ALIASES.get(name) or _ALIASES.get(name.lower())
    if canonical is None:
        raise KeyError(
            f"Unknown GPU profile '{name}'. "
            f"Available: {', '.join(sorted(GPU_PROFILES))}"
        )
    return GPU_PROFILES[canonical]


def list_profiles() -> None:
    """Print a human-readable summary of all available profiles."""
    bar = "=" * 60
    print(f"\n{bar}")
    print("  VoltVandal — GPU Series Profiles")
    print(bar)
    for key, p in GPU_PROFILES.items():
        print(f"\n  Profile  : {key}")
        print(f"  Name     : {p['name']}")
        print(f"  Anchor   : {p['target_voltage_mv']} mV  (--target-voltage-mv)")
        print(f"  OC step  : {p['step_mhz']} MHz            (--step-mhz)")
        print(f"  UV step  : {p['step_mv']} mV             (--step-mv)")
        print(f"  Max steps: {p['max_steps']}              (--max-steps)")
        print(f"  Stress   : {p['stress_seconds']}s / {p['multi_stress_seconds']}s per mode (--stress-seconds / --multi-stress-seconds)")
        print(f"  Limits   : edge<{p['temp_limit_c']}°C  hotspot<{p['hotspot_limit_c']}°C  power<{p['power_limit_w']}W")
        print(f"  Bins     : {p['bin_min_mv']}–{p['bin_max_mv']} mV  (--bin-min-mv / --bin-max-mv)")
        print(f"  Notes    : {p['notes']}")
    print(f"\n{bar}")
    print(
        "\nUsage:\n"
        "  python voltvandal.py run --mode vlock --gpu-profile rtx40 \\\n"
        "    --gpu 0 --out artifacts \\\n"
        "    --doloming-mode ray,matrix,frequency-max\n"
        "\n  Profile values are defaults — any explicit flag overrides them.\n"
    )
    print(bar)
