# MCE variation across prompt templates (MAPLE)
#
# This script adapts the ECE analysis to plot Maximum Calibration Error (MCE)
# variation across different hard prompt initializations.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Datasets in order -----
datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

# ----- Raw per-template results across the 4 datasets -----
raw = {
    "a photo of a": {
        "ACC": [81.77, 77.70, 56.83, 59.39],
        "MCE": [0.98, 1.03, 2.48, 3.00],
    },
    "a photo of a retina with": {
        "ACC": [81.84, 77.73, 55.19, 59.23],
        "MCE": [0.82, 0.32, 3.50, 3.23],
    },
    "a color fundus photograph showing diabetic retinopathy severity level as": {
        "ACC": [81.38, 77.69, 57.38, 60.11],
        "MCE": [1.02, 0.65, 1.65, 2.47],
    },
    "Classify diabetic retinopathy severity from this fundus photo of a": {
        "ACC": [80.51, 77.49, 49.38, 63.28],
        "MCE": [1.17, 0.83, 4.11, 2.34],
    },
    "You are classifying diabetic retinopathy on a fundus photo of a": {
        "ACC": [81.61, 77.62, 51.96, 58.56],
        "MCE": [0.82, 1.17, 3.76, 2.19],
    },
    "This pic illustrates diab retinopthy severty of a": {
        "ACC": [81.62, 77.51, 50.47, 62.75],
        "MCE": [2.21, 2.55, 8.71, 6.57],
    },
    "The DR level is": {
        "ACC": [81.24, 77.59, 51.05, 61.39],
        "MCE": [1.88, 2.12, 9.52, 6.62],
    },
}

# ----- Text size controls -----
x_label_size = 15          # x-axis label size
x_tick_size = 12           # x-axis values/numbers size

y_label_size = 15          # y-axis label size
y_tick_size = 12           # y-axis P1, P2, P3... size

title_size = 16            # title text size
annotation_size = 12       # bar annotation text size: std (μ=mean)
legend_text_size = 12      # top-left corner legend text size

# ----- Tidy table -----
rows = []

for templ, vals in raw.items():
    for ds_i, ds in enumerate(datasets):
        rows.append({
            "Template": templ,
            "Dataset": ds,
            "ACC_percent": float(vals["ACC"][ds_i]),
            "MCE_percent": float(vals["MCE"][ds_i]),
        })

df = pd.DataFrame(rows)

# ----- Per-template MCE variation across datasets -----
per_template = (
    df.groupby("Template")["MCE_percent"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "MCE_mean", "std": "MCE_std"})
      .sort_values("MCE_std", ascending=True)
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
bars = ax.barh(
    y,
    per_template["MCE_std"].values,
    label="MCE Std."
)

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
    "MCE standard deviation across datasets (percentage points)",
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
    "MCE Variation Across Prompt Templates (MAPLE)\n"
    "(lower = more stable across APTOS / EyePACS / Messidor / Messidor-2)",
    fontsize=title_size
)

ax.grid(axis="x", alpha=0.3)

# Calculate padding for the x-axis
xmax = float(per_template["MCE_std"].max())

RIGHT_PAD_FRAC = 0.40
RIGHT_PAD_ABS = 0.80

pad = max(RIGHT_PAD_ABS, RIGHT_PAD_FRAC * xmax)
ax.set_xlim(0, xmax + pad)

# Y-axis padding
ax.set_ylim(-0.5, n - 0.5)

# Annotate bars with "std  (μ=mean MCE)"
text_offset = 0.02 * (xmax + pad)

for rect, std_val, mean_val in zip(
    bars,
    per_template["MCE_std"].values,
    per_template["MCE_mean"].values
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
mean_line = ax.axvline(
    per_template["MCE_std"].mean(),
    linestyle="--",
    linewidth=1.0,
    alpha=0.55,
    label="Mean Std."
)

# Top-left corner legend text
"""ax.legend(
    loc="upper left",
    fontsize=legend_text_size,
    framealpha=0.9
)"""

# Bold border lines
ax.spines["top"].set_visible(True)
ax.spines["top"].set_linewidth(1.8)

for side in ["left", "right", "bottom"]:
    ax.spines[side].set_linewidth(1.2)

# Figure-level top border strip
fig.canvas.draw()

bbox = fig.bbox
border_px = 2
height_fig = bbox.height
border_h = border_px / height_fig

top_border = Rectangle(
    (0, 1 - border_h),
    1,
    border_h,
    transform=fig.transFigure,
    color="black",
    ec="none"
)

fig.add_artist(top_border)

# Adjust margins for readability
plt.subplots_adjust(
    left=0.18,
    right=0.985,
    top=0.90,
    bottom=0.10
)

# Save figure
out_dir = Path("medical_bm_fig")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "mce_variation_across_templates_STD_hbar_Pnums_filled_wide.png"

plt.savefig(
    out_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved to: {out_path.resolve()}")