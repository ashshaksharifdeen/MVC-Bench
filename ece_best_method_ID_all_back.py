# ID ECE per calibration method across backbones (boxplots) — with lifted N_best labels
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- your data dict exactly as you used ----------------
data = {
    "CLIP-ViT-B/32": {
        "Base": 3.89, "MDCA": 4.4725, "LS": 7.065, "MBLS": 3.835,
        "ECCV_ZS": 28.255, "ECCV_Penalty": 27.8625, "Temperature": 5.7825, "Zero-shot": 25.8575
    },
    "CLIP-ViT-B/16": {
        "Base": 4.4425, "MDCA": 4.825, "LS": 7.4075, "MBLS": 4.445,
        "ECCV_ZS": 26.535, "ECCV_Penalty": 28.41, "Temperature": 6.57, "Zero-shot": 24.3225
    },
    "CLIP-ResNet-50": {
        "Base": 4.4025, "MDCA": 4.4275, "LS": 6.12, "MBLS": 4.1625,
        "ECCV_ZS": 28.255, "ECCV_Penalty": 27.8625, "Temperature": 5.7825, "Zero-shot": 20.1175
    },
    "CLIP-ResNet-101": {
        "Base": 5.82, "MDCA": 5.6375, "LS": 7.9325, "MBLS": 6.105,
        "ECCV_ZS": 25.59, "ECCV_Penalty": 25.695, "Temperature": 6.6575, "Zero-shot": 28.86
    },
    "PLIP-ViT-B/32": {
        "Base": 3.383333333, "MDCA": 3.588333333, "LS": 9.393333332, "MBLS": 4.009166667,
        "ECCV_ZS": 3.531666667, "ECCV_Penalty": 18.34333334, "Temperature": 5.524166667, "Zero-shot": 14.4575
    },
    "QuiltNet-ViT-B/32": {
        "Base": 3.45, "MDCA": 3.7825, "LS": 4.736666667, "MBLS": 3.46,
        "ECCV_ZS": 3.531666667, "ECCV_Penalty": 16.38083334, "Temperature": 6.543333334, "Zero-shot": 17.38916667
    },
    "Med-VLM-ViT-B/32": {
        "Base": 6.688333333, "MDCA": 7.0575, "LS": 9.5725, "MBLS": 7.091666667,
        "ECCV_ZS": 5.631666667, "ECCV_Penalty": 21.0275, "Temperature": 10.07083333, "Zero-shot": 11.8075
    },
    "Biomed-VLM-ViT-B/32": {
        "Base": 6.688333333, "MDCA": 7.0575, "LS": 9.5725, "MBLS": 7.091666667,
        "ECCV_ZS": 5.631666667, "ECCV_Penalty": 21.0275, "Temperature": 10.07083333, "Zero-shot": 11.8075
    },
    "Med-VLM-ViT-B/32-X-RaY": {
        "Base": 23.97, "MDCA": 23.98833333, "LS": 24.20166667, "MBLS": 24.34666667,
        "ECCV_ZS": 24.16666667, "ECCV_Penalty": 23.82, "Temperature": 23.93333333, "Zero-shot": 21.36
    },
    "Biomed-VLM-ViT-B/32-X-RaY": {
        "Base": 7.781666667, "MDCA": 8.076666667, "LS": 8.225, "MBLS": 8.016666667,
        "ECCV_ZS": 7.435, "ECCV_Penalty": 7.838333333, "Temperature": 7.823333333, "Zero-shot": 20.095
    },
    "PLIP-ViT-B/32--Histopathology": {
        "Base": 7.002222223, "MDCA": 6.134444444, "LS": 5.685833333, "MBLS": 6.186666666,
        "ECCV_ZS": 6.083333334, "ECCV_Penalty": 8.352222223, "Temperature": 6.377777778, "Zero-shot": 13.82
    },
    "QuiltNet-ViT-B/32-Histopathology": {
        "Base": 6.214444443, "MDCA": 5.48, "LS": 3.7425, "MBLS": 6.044444444,
        "ECCV_ZS": 5.416666668, "ECCV_Penalty": 11.15888889, "Temperature": 6.117777778, "Zero-shot": 16.08
    },
}

# ---------------- tidy DF ----------------
rows = []
for backbone, vals in data.items():
    for method, ece in vals.items():
        rows.append((backbone, method, float(ece)))
df = pd.DataFrame(rows, columns=["backbone","method","ECE_ID"])

# ---------------- N_best counts (with fractional tie credit) ----------------
pivot = df.pivot_table(index="backbone", columns="method", values="ECE_ID", aggfunc="first")
best_counts = {m: 0.0 for m in pivot.columns}
eps = 1e-12
for _, row in pivot.iterrows():
    vals = row.values.astype(float)
    min_val = np.nanmin(vals)
    is_best = np.isclose(vals, min_val, atol=eps, rtol=0.0)
    k = int(is_best.sum())
    if k > 0:
        share = 1.0 / k
        for m, flag in zip(pivot.columns, is_best):
            if flag:
                best_counts[m] += share

# ---------------- sort & box data ----------------
methods_order = sorted(df["method"].unique(), key=lambda m: df[df["method"]==m]["ECE_ID"].median())
data_box = [df[df["method"]==m]["ECE_ID"].values for m in methods_order]

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(14, 6))

flierprops = dict(marker='o', markerfacecolor='none', markeredgecolor='k', markersize=6, alpha=0.9)
meanprops  = dict(marker='^', markerfacecolor='C2', markeredgecolor='C2', markersize=8)

ax.boxplot(data_box, labels=methods_order, whis=[10,90], showmeans=True,
           flierprops=flierprops, meanprops=meanprops)

# jittered per-backbone dots
rng = np.random.default_rng(123)
for xi, vals in enumerate(data_box, start=1):
    xj = xi + (rng.random(len(vals)) - 0.5) * 0.18
    ax.scatter(xj, vals, s=22, alpha=0.85)   # coloured dots (per-backbone observations)

ax.set_ylabel("ECE (%, ID)")
ax.set_title("Consistency Across Backbones: ID ECE per Calibration Method (Boxplots)")

# visible borders (including top)
for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.2)

# add a touch of headroom so labels never collide with top
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax * 1.05)

# N_best labels ABOVE boxes, using axes-fraction y to avoid overlap with dots
for xi, m in enumerate(methods_order, start=1):
    n_best = best_counts.get(m, 0.0)
    label = f"N_best={int(round(n_best))}" if abs(n_best - round(n_best)) < 1e-6 else f"N_best={n_best:.1f}"
    ax.text(xi, 0.985, label, transform=ax.get_xaxis_transform(),  # y in [0,1] of axes
            ha="center", va="top", fontsize=10)

plt.xticks(rotation=20)
plt.tight_layout()

out_dir = "./medical_bm_fig"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "id_ece_boxplots_from_user_data_with_Nbest_lifted.png"),
            dpi=300, bbox_inches="tight")
# plt.show()

# optional: export tables
pd.Series(best_counts, name="N_best").reindex(methods_order).to_csv(
    os.path.join(out_dir, "id_ece_boxplots_Nbest_counts.csv"))
df.to_csv(os.path.join(out_dir, "id_ece_boxplots_from_user_data.csv"), index=False)
