from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _matplotlib_available = True
except ImportError:
    plt = None
    _matplotlib_available = False

def plot_vf_curve(
    stock_csv: Path,
    last_good_csv: Path,
    out_path: Path,
) -> Optional[Path]:
    if not _matplotlib_available:
        return None
    if not stock_csv.exists() or not last_good_csv.exists():
        return None

    def _load(p: Path):
        import csv as _csv
        rows = []
        with p.open(newline="") as f:
            for row in _csv.reader(f):
                if len(row) >= 2:
                    try:
                        rows.append((int(row[0]) / 1000, int(row[1]) / 1000))
                    except ValueError:
                        pass
        return rows

    stock = _load(stock_csv)
    last_good = _load(last_good_csv)
    if not stock or not last_good:
        return None

    s_v, s_f = zip(*stock)
    g_v, g_f = zip(*last_good)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s_v, s_f, color="silver", linestyle="--", linewidth=1.5, label="Stock curve")
    ax.plot(g_v, g_f, color="#2196F3", linewidth=2, label="Last-good (tuned)")

    common_v = sorted(set(s_v) | set(g_v))
    s_interp = np.interp(common_v, s_v, s_f)
    g_interp = np.interp(common_v, g_v, g_f)
    ax.fill_between(
        common_v, s_interp, g_interp,
        where=(g_interp >= s_interp), alpha=0.15, color="#4CAF50", label="Freq gain (OC)"
    )
    ax.fill_between(
        common_v, s_interp, g_interp,
        where=(g_interp < s_interp), alpha=0.15, color="#F44336", label="Freq reduction"
    )

    ax.set_xlabel("Voltage (mV)")
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title("VoltVandal — VF Curve Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path
