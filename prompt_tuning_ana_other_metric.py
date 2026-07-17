# Grouped bars with distinct colors for ACE and MCE
# - One figure, Matplotlib only (no seaborn)
# - Two bars per method (ACE %, MCE %)
# - Twin y-axes for readability (ACE left, MCE right)
# - Error bars = std across APTOS/EyePACS/Messidor/Messidor_2
# - Saves PNG and summary CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Your per-dataset values ----
raw = {
    "MAPLE": {
        "ACE": [3.96, 3.05, 6.44, 9.38],
        "MCE": [1.46, 1.11, 2.16, 3.67],
    },
    "KGCOOP": {
        "ACE": [3.89, 3.55, 4.79, 7.09],
        "MCE": [1.99, 0.75, 1.71, 1.41],
    },
    "PROMPT SRC": {
        "ACE": [15.99, 10.14, 10.06, 10.66],
        "MCE": [5.82, 6.53, 4.71, 3.33],
    },
    "Pro-GRAD": {
        "ACE": [27.55, 14.34, 23.77, 27.15],
        "MCE": [9.75, 8.24, 9.41, 16.23],
    },
    "Coop": {
        "ACE": [3.99, 2.30, 5.69, 6.24],
        "MCE": [1.46, 0.68, 1.78, 1.54],
    },
    "HiCroPL": {
        "ACE": [5.14, 1.10, 4.62, 3.28],
        "MCE": [1.06, 0.58, 1.38, 1.24],
    },
}

datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

# ---- Text size controls ----
x_label_size = 15          # x-axis label size
x_tick_size = 12           # x-axis method names size

y_label_size = 15          # y-axis label size
y_tick_size = 12           # y-axis values size

legend_text_size = 12      # top-left legend text size
title_size = 15            # title text size
bar_value_size = 10        # bar value annotation size

# ---- Tidy DF + summary stats (mean/std across datasets) ----
rows = []

for method, vals in raw.items():
    for i, ds in enumerate(datasets):
        rows.append({
            "Method": method,
            "Dataset": ds,
            "ACE_percent": float(vals["ACE"][i]),
            "MCE_percent": float(vals["MCE"][i]),
        })

df = pd.DataFrame(rows)

summary = (
    df.groupby("Method")[["ACE_percent", "MCE_percent"]]
      .agg(["mean", "std"])
      .reset_index()
)

summary.columns = ["Method", "Avg_ACE", "Std_ACE", "Avg_MCE", "Std_MCE"]

# ---- Display order ----
method_order = ["Coop", "KGCOOP", "MAPLE", "PROMPT SRC", "Pro-GRAD", "HiCroPL"]
summary = summary.set_index("Method").loc[method_order].reset_index()

# ---- Colors ----
acc_color = "tab:blue"
ece_color = "tab:red"

# ---- Plot ----
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(summary))
width = 0.36

# ACE bars - left axis
acc_bars = ax1.bar(
    x - width / 2,
    summary["Avg_ACE"].values,
    yerr=summary["Std_ACE"].values,
    capsize=4,
    width=width,
    label="ACE (%)",
    color=acc_color,
    edgecolor="black",
    linewidth=0.6
)

ax1.set_xlabel("Prompt-Tuning Method", fontsize=x_label_size)
ax1.set_ylabel("ACE (%) ↓ (lower is better)", color=acc_color, fontsize=y_label_size)

ax1.tick_params(axis="y", colors=acc_color, labelsize=y_tick_size)
ax1.tick_params(axis="x", labelsize=x_tick_size)

ax1.set_xticks(x)
ax1.set_xticklabels(
    summary["Method"].values,
    rotation=15,
    fontsize=x_tick_size
)

ax1.grid(
    axis="y",
    alpha=0.3,
    linestyle="--",
    linewidth=0.8
)

# MCE bars - right axis
ax2 = ax1.twinx()

ece_bars = ax2.bar(
    x + width / 2,
    summary["Avg_MCE"].values,
    yerr=summary["Std_MCE"].values,
    capsize=4,
    width=width,
    label="MCE (%)",
    color=ece_color,
    edgecolor="black",
    linewidth=0.6
)

ax2.set_ylabel("MCE (%) ↓ (lower is better)", color=ece_color, fontsize=y_label_size)
ax2.tick_params(axis="y", colors=ece_color, labelsize=y_tick_size)

# ---- Value labels on bars ----
for rect in acc_bars:
    h = rect.get_height()
    ax1.text(
        rect.get_x() + rect.get_width() / 2,
        h + max(0.6, 0.02 * h),
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=bar_value_size,
        color=acc_color
    )

for rect in ece_bars:
    h = rect.get_height()
    ax2.text(
        rect.get_x() + rect.get_width() / 2,
        h + max(0.3, 0.03 * h),
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=bar_value_size,
        color=ece_color
    )

# ---- Neat frame ----
for a in (ax1, ax2):
    for side in ["top", "right", "bottom", "left"]:
        a.spines[side].set_visible(True)
        a.spines[side].set_linewidth(1.2)

# ---- Single legend - top-left corner text ----
handles = [acc_bars, ece_bars]
labels = ["ACE (%)", "MCE (%)"]

ax1.legend(
    handles,
    labels,
    loc="upper left",
    framealpha=0.9,
    fontsize=legend_text_size
)

# ---- Title ----
plt.title(
    "Prompt-Tuning Families: Grouped ACE (blue) & MCE (red)",
    fontsize=title_size
)

plt.tight_layout()

# ---- Save figure ----
plt.savefig(
    "./medical_bm_fig/prompt_tuning_acc_mce_grouped_twoaxis_colored.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ---- Save the summary used for plotting ----
summary.to_csv(
    "./medical_bm_fig/prompt_tuning_acc_mce_grouped_summary.csv",
    index=False
)