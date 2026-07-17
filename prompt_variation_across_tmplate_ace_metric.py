# ACE variation across prompt templates (MAPLE)
# This script mirrors the structure of the ECE analysis but uses
# Average Calibration Error (ACE) instead of ECE.

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Raw per-template results across the 4 datasets -----

datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

raw = {
    "a photo of a": {
        "ACC": [81.77, 77.70, 56.83, 59.39],
        "ACE": [3.10, 2.84, 7.17, 7.31],
    },
    "a photo of a retina with": {
        "ACC": [81.84, 77.73, 55.19, 59.23],
        "ACE": [2.62, 0.81, 9.02, 9.00],
    },
    "a color fundus photograph showing diabetic retinopathy severity level as": {
        "ACC": [81.38, 77.69, 57.38, 60.11],
        "ACE": [3.12, 1.90, 6.79, 6.78],
    },
    "Classify diabetic retinopathy severity from this fundus photo of a": {
        "ACC": [80.51, 77.49, 49.38, 63.28],
        "ACE": [3.57, 2.42, 9.54, 8.44],
    },
    "You are classifying diabetic retinopathy on a fundus photo of a": {
        "ACC": [81.61, 77.62, 51.96, 58.56],
        "ACE": [2.52, 2.62, 8.54, 6.48],
    },
    "This pic illustrates diab retinopthy severty of a": {
        "ACC": [81.62, 77.51, 50.47, 62.75],
        "ACE": [2.21, 2.52, 8.71, 6.57],
    },
    "The DR level is": {
        "ACC": [81.24, 77.59, 51.05, 61.39],
        "ACE": [1.88, 2.12, 9.52, 6.62],
    },
}

# ----- Text size controls -----
x_label_size = 15          # x-axis label size
x_tick_size = 12           # x-axis values/numbers size

y_label_size = 15          # y-axis label size
y_tick_size = 12           # y-axis P1, P2, P3... size

title_size = 16            # title text size
annotation_size = 12       # bar annotation text size: std (μ=mean)

# ----- Tidy table -----
rows = []

for templ, vals in raw.items():
    for ds_i, ds in enumerate(datasets):
        rows.append({
            "Template": templ,
            "Dataset": ds,
            "ACC_percent": float(vals["ACC"][ds_i]),
            "ACE_percent": float(vals["ACE"][ds_i]),
        })

df = pd.DataFrame(rows)

# ----- Per-template ACE variation across datasets -----
per_template = (
    df.groupby("Template")["ACE_percent"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "ACE_mean", "std": "ACE_std"})
      .sort_values("ACE_std", ascending=True)
      .reset_index(drop=True)
)

# Assign compact labels P1..Pn
n = len(per_template)
per_template["Px"] = [f"P{i+1}" for i in range(n)]

# ===== Plot =====
FIG_W = 14.0
FIG_H = max(7.0, 0.75 * n + 3.0)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# Horizontal bars
y = np.arange(n)
bars = ax.barh(y, per_template["ACE_std"].values)

# Y-axis labels
ax.set_yticks(y)
ax.set_yticklabels(
    per_template["Px"].tolist(),
    fontsize=y_tick_size
)

ax.set_ylabel(
    "Prompt Template",
    fontsize=y_label_size
)

ax.invert_yaxis()

# X-axis label and x-axis values
ax.set_xlabel(
    "ACE standard deviation across datasets (percentage points)",
    fontsize=x_label_size
)

ax.tick_params(
    axis="x",
    labelsize=x_tick_size
)

ax.tick_params(
    axis="y",
    labelsize=y_tick_size
)

# Title
ax.set_title(
    "ACE Variation Across Prompt Templates (MAPLE)\n"
    "(lower = more stable across APTOS / EyePACS / Messidor / Messidor-2)",
    fontsize=title_size
)

ax.grid(axis="x", alpha=0.3)

# More headroom on the right so annotations do not clip
xmax = float(per_template["ACE_std"].max())

RIGHT_PAD_FRAC = 0.40
RIGHT_PAD_ABS = 0.80

pad = max(RIGHT_PAD_ABS, RIGHT_PAD_FRAC * xmax)
ax.set_xlim(0, xmax + pad)

# Slight y padding so top/bottom bars do not touch edges
ax.set_ylim(-0.5, n - 0.5)

# Annotate each bar with "std  (μ=mean ACE)"
text_offset = 0.02 * (xmax + pad)

for rect, std_val, mean_val in zip(
    bars,
    per_template["ACE_std"].values,
    per_template["ACE_mean"].values
):
    x_val = rect.get_width()

    ax.text(
        x_val + text_offset,
        rect.get_y() + rect.get_height() / 2,
        f"{std_val:.2f}  (μ={mean_val:.2f})",
        va="center",
        ha="left",
        fontsize=annotation_size
    )

# Mean reference line
ax.axvline(
    per_template["ACE_std"].mean(),
    linestyle="--",
    linewidth=1.0,
    alpha=0.55
)

# ---- Bold borders ----
ax.spines["top"].set_visible(True)
ax.spines["top"].set_linewidth(1.8)

for side in ["left", "right", "bottom"]:
    ax.spines[side].set_linewidth(1.2)

# Figure-level top border strip
fig.canvas.draw()

from matplotlib.patches import Rectangle as _Rect

bbox = fig.bbox
border_px = 2
height_fig = bbox.height
border_h = border_px / height_fig

top_border = _Rect(
    (0, 1 - border_h),
    1,
    border_h,
    transform=fig.transFigure,
    color="black",
    ec="none"
)

fig.add_artist(top_border)

# Wider right/left margins
plt.subplots_adjust(
    left=0.18,
    right=0.985,
    top=0.90,
    bottom=0.10
)

# Save
out_dir = Path("medical_bm_fig")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "ace_variation_across_templates_STD_hbar_Pnums_filled_wide.png"

plt.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved to: {out_path.resolve()}")