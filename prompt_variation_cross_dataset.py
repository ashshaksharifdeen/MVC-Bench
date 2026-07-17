# Variance across prompt initialisations (MAPLE)
# Figure: grouped bars showing STD across templates per dataset (ECE vs ACC)
# - Matplotlib only (no seaborn)
# - Adds a bold top border on both the axes and the overall figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Your raw per-template results across the 4 datasets -----
datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

raw = {
    "a photo of a": {
        "ACC": [81.77, 77.70, 56.83, 59.39],
        "ECE": [ 3.03,  2.80,  5.23,  7.19],
    },
    "a photo of a retina with": {
        "ACC": [81.84, 77.73, 55.19, 59.23],
        "ECE": [ 2.64,  0.79,  8.59,  9.15],
    },
    "a color fundus photograph showing diabetic retinopathy severity level as": {
        "ACC": [81.38, 77.69, 57.38, 60.11],
        "ECE": [ 3.08,  1.90,  5.12,  6.48],
    },
    "Classify diabetic retinopathy severity from this fundus photo of a": {
        "ACC": [80.51, 77.49, 49.38, 63.28],
        "ECE": [ 3.35,  2.41,  6.82,  8.32],
    },
    "You are classifying diabetic retinopathy on a fundus photo of a": {
        "ACC": [81.61, 77.62, 51.96, 58.56],
        "ECE": [ 2.70,  2.58,  7.49,  5.98],
    },
    "This pic illustrates diab retinopthy severty of a": {
        "ACC": [81.62, 77.51, 50.47, 62.75],
        "ECE": [ 2.26,  2.54,  6.88,  6.58],
    },
    "The DR level is": {
        "ACC": [81.24, 77.59, 51.05, 61.39],
        "ECE": [ 2.08,  2.13,  7.76,  6.22],
    },
}

# ----- Build tidy DataFrame -----
rows = []
for templ, vals in raw.items():
    for ds_i, ds in enumerate(datasets):
        rows.append({
            "Template": templ,
            "Dataset": ds,
            "ACC_percent": float(vals["ACC"][ds_i]),
            "ECE_percent": float(vals["ECE"][ds_i]),
        })
df = pd.DataFrame(rows)

# ----- Per-dataset variability (std across templates) -----
per_dataset = (
    df.groupby("Dataset")[["ACC_percent", "ECE_percent"]]
      .agg(['mean','std'])
      .reset_index()
)
per_dataset.columns = ["Dataset", "ACC_mean", "ACC_std", "ECE_mean", "ECE_std"]

# ----- Plot: grouped bars (ECE std vs ACC std) -----
fig, ax = plt.subplots(figsize=(9.2, 5.2))
x = np.arange(len(per_dataset))
width = 0.38

ece_bars = ax.bar(x - width/2, per_dataset["ECE_std"].values, width=width,
                  label="ECE std (across templates)")
acc_bars = ax.bar(x + width/2, per_dataset["ACC_std"].values, width=width,
                  label="ACC std (across templates)")

ax.set_xticks(x)
ax.set_xticklabels(per_dataset["Dataset"].values)
ax.set_ylabel("Standard deviation across templates (percentage points)")
ax.set_title("Variance across Prompt Initialisations (MAPLE): per-dataset STD of ECE vs ACC")
ax.grid(axis="y", alpha=0.3)

# Annotate bars
for rect in list(ece_bars) + list(acc_bars):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h + max(0.05, 0.015*h),
            f"{h:.2f}", ha="center", va="bottom", fontsize=9)

# Reference lines (mean std across datasets)
ax.axhline(per_dataset["ECE_std"].mean(), linestyle="--", linewidth=1.0, alpha=0.5)
ax.axhline(per_dataset["ACC_std"].mean(), linestyle="--", linewidth=1.0, alpha=0.5)

ax.legend(loc="upper left", framealpha=0.9)

# ----- Make top borders bold -----
# 1) Axes top spine
ax.spines["top"].set_visible(True)
ax.spines["top"].set_linewidth(1.6)
# Also keep other spines neat
for side in ["left", "right", "bottom"]:
    ax.spines[side].set_linewidth(1.2)

# 2) Figure-level top border (draw a thin rectangle at the very top of the figure)
#    This creates a crisp line at the top edge of the whole figure canvas.
fig.canvas.draw()  # ensure figure bbox exists
bbox = fig.bbox
# Height of the border in figure pixels (adjust if you want thicker)
border_px = 2
# Convert to figure coordinates [0..1]
height_fig = bbox.height
border_h = border_px / height_fig
top_border = Rectangle((0, 1 - border_h), 1, border_h,
                       transform=fig.transFigure, color="black", ec="none")
fig.add_artist(top_border)

plt.tight_layout()
plt.savefig("./medical_bm_fig/maple_variance_across_templates_per_dataset_STD_with_top_border.png",
            dpi=200, bbox_inches="tight")
plt.show()
