# ece_single_panels.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# X-axis (shots per class)
shots = np.array([4, 8, 16, 32])

# ===== Your data =====
left_data = {
    "Maple":        [6.53, 6.22, 4.28, 3.20],
    "Maple+Ours":   [5.88, 5.08, 3.45, 1.29],
    "Coop":         [7.46, 5.49, 4.93, 2.81],
    "Coop+Ours":    [5.41, 5.33, 4.80, 2.26],
    "KgCoop":       [18.21, 15.05, 12.12, 4.57],
    "KgCoop+Ours":  [16.07, 14.32, 11.21, 2.96],
}

right_data = {
    "Maple":        [6.75, 10.18, 11.67, 12.26],
    "Maple+Ours":   [5.89,  8.39, 10.07, 11.32],
    "Coop":         [9.45,  9.54, 12.13, 12.32],
    "Coop+Ours":    [2.70,  2.96,  3.51,  4.57],
    "KgCoop":       [4.21,  4.84,  5.03, 12.22],
    "KgCoop+Ours":  [3.12,  4.06,  4.78,  6.66],
}
# =====================

def plot_panel(ax, x, series_dict, title=None, show_legend=False):
    for name, y_vals in series_dict.items():
        y = np.asarray(y_vals, dtype=float)
        is_ours = name.endswith("+Ours")
        ls = "-" if is_ours else "--"
        ax.plot(x, y, linestyle=ls, marker="o", linewidth=2.0, markersize=5, label=name)

    ax.set_xlabel("", fontsize=18)  # keep blank like your version
    ax.set_ylabel("ECE (%)", fontsize=18)
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, linestyle=":", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=20, pad=8)
    if show_legend:
        ax.legend(loc="upper left", ncol=1, frameon=True, fontsize=14)

def save_single_panel(x, data, title, basename):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_panel(ax, x, data, title=title, show_legend=True)
    plt.tight_layout()
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)

# --- make two separate images ---
save_single_panel(shots, left_data,  "Base Class",  "ece_base_class")
save_single_panel(shots, right_data, "Novel Class", "ece_novel_class")

print("Saved: figures/ece_base_class.{png,pdf} and figures/ece_novel_class.{png,pdf}")
