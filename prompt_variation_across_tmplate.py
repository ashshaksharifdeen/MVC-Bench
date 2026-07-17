# ECE variation across prompt templates (MAPLE)
# Enlarged figure + generous right/left margins so annotations never clip

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- Raw per-template results across the 4 datasets -----
datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

raw = {
    "a photo of a": {"ACC": [81.77, 77.70, 56.83, 59.39], "ECE": [3.03, 2.80, 5.23, 7.19]},
    "a photo of a retina with": {"ACC": [81.84, 77.73, 55.19, 59.23], "ECE": [2.64, 0.79, 8.59, 9.15]},
    "a color fundus photograph showing diabetic retinopathy severity level as": {"ACC": [81.38, 77.69, 57.38, 60.11], "ECE": [3.08, 1.90, 5.12, 6.48]},
    "Classify diabetic retinopathy severity from this fundus photo of a": {"ACC": [80.51, 77.49, 49.38, 63.28], "ECE": [3.35, 2.41, 6.82, 8.32]},
    "You are classifying diabetic retinopathy on a fundus photo of a": {"ACC": [81.61, 77.62, 51.96, 58.56], "ECE": [2.70, 2.58, 7.49, 5.98]},
    "This pic illustrates diab retinopthy severty of a": {"ACC": [81.62, 77.51, 50.47, 62.75], "ECE": [2.26, 2.54, 6.88, 6.58]},
    "The DR level is": {"ACC": [81.24, 77.59, 51.05, 61.39], "ECE": [2.08, 2.13, 7.76, 6.22]},
}

# ----- Tidy table -----
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

# ----- Per-template ECE variation across datasets -----
per_template = (
    df.groupby("Template")["ECE_percent"]
      .agg(['mean','std'])
      .reset_index()
      .rename(columns={'mean':'ECE_mean','std':'ECE_std'})
      .sort_values("ECE_std", ascending=True)  # most stable first
      .reset_index(drop=True)
)

# Assign compact labels P1..Pn (sorted order)
n = len(per_template)
per_template["Px"] = [f"P{i+1}" for i in range(n)]

# ===== Plot =====
# Bigger canvas + more vertical breathing room
FIG_W = 14.0
FIG_H = max(7.0, 0.75 * n + 3.0)  # scale with number of templates
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

# Horizontal bars
y = np.arange(n)
bars = ax.barh(y, per_template["ECE_std"].values)

# Y labels
ax.set_yticks(y)
ax.set_yticklabels(per_template["Px"].tolist(), fontsize=12)
ax.invert_yaxis()

ax.set_xlabel("ECE standard deviation across datasets (percentage points)")
ax.set_title("ECE Variation Across Prompt Templates (MAPLE)\n(lower = more stable across APTOS / EyePACS / Messidor / Messidor-2)")
ax.grid(axis="x", alpha=0.3)

# More headroom on the right so annotations don't clip
xmax = float(per_template["ECE_std"].max())
# Add generous padding: combine fractional + absolute padding
RIGHT_PAD_FRAC = 0.40   # 40% of the max bar length
RIGHT_PAD_ABS  = 0.80   # +0.8 p.p. absolute
pad = max(RIGHT_PAD_ABS, RIGHT_PAD_FRAC * xmax)
ax.set_xlim(0, xmax + pad)

# Slight y padding so top/bottom bars don't touch edges
ax.set_ylim(-0.5, n - 0.5)

# Annotate each bar with "std  (μ=mean ECE)"
# Use a small offset that scales with the axis width
text_offset = 0.02 * (xmax + pad)  # ~2% of axis width
for rect, std_val, mean_val in zip(bars, per_template["ECE_std"].values, per_template["ECE_mean"].values):
    x = rect.get_width()
    ax.text(x + text_offset, rect.get_y() + rect.get_height()/2,
            f"{std_val:.2f}  (μ={mean_val:.2f})",
            va="center", ha="left", fontsize=11)

# Mean reference
ax.axvline(per_template["ECE_std"].mean(), linestyle="--", linewidth=1.0, alpha=0.55)

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
top_border = _Rect((0, 1 - border_h), 1, border_h, transform=fig.transFigure, color="black", ec="none")
fig.add_artist(top_border)

# Wider right/left margins so nothing feels cramped
plt.subplots_adjust(left=0.18, right=0.985, top=0.90, bottom=0.10)

# Save
out_dir = Path("medical_bm_fig"); out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "ece_variation_across_templates_STD_hbar_Pnums_filled_wide.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.show()

print(f"Saved to: {out_path.resolve()}")
