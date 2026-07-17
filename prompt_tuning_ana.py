# Grouped bars with distinct colors for Accuracy and ECE
# - One figure, Matplotlib only (no seaborn)
# - Two bars per method (Accuracy %, ECE %)
# - Twin y-axes for readability (Acc left, ECE right)
# - Error bars = std across APTOS/EyePACS/Messidor/Messidor_2
# - Saves PNG and summary CSV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Your per-dataset values ----
raw = {
    "MAPLE": {
        "ACC": [80.6, 77.58, 57.83, 61.1],
        "ECE": [ 3.97,  3.02,  5.86,  8.56],
    },
    "KGCOOP": {
        "ACC": [76.31, 73.99, 58.2, 63.97],
        "ECE": [ 3.85,  3.45,  4.82,  7.18],
    },
    "PROMPT SRC": {
        "ACC": [84.42, 77.85, 60.61, 65.19],
        "ECE": [15.99, 10.14,  9.63, 10.32],
    },
    "Pro-GRAD": {
        "ACC": [73.13, 73.81, 54.92, 63.24],
        "ECE": [27.67, 27.16, 13.72, 23.77],
    },
    "Coop": {
        "ACC": [77.9, 74.64, 59.5, 64.66],
        "ECE": [ 3.99,  2.25,  5.29,  6.24],
    },
}
datasets = ["APTOS", "EyePACS", "Messidor", "Messidor_2"]

# ---- Tidy DF + summary stats (mean/std across datasets) ----
rows = []
for method, vals in raw.items():
    for i, ds in enumerate(datasets):
        rows.append({
            "Method": method,
            "Dataset": ds,
            "ACC_percent": float(vals["ACC"][i]),
            "ECE_percent": float(vals["ECE"][i]),
        })
df = pd.DataFrame(rows)

summary = (
    df.groupby("Method")[["ACC_percent", "ECE_percent"]]
      .agg(["mean", "std"]).reset_index()
)
summary.columns = ["Method", "Avg_ACC", "Std_ACC", "Avg_ECE", "Std_ECE"]

# ---- Display order (edit if you prefer a different order) ----
method_order = ["Coop", "KGCOOP", "MAPLE", "PROMPT SRC", "Pro-GRAD"]
summary = summary.set_index("Method").loc[method_order].reset_index()

# ---- Colors (explicit, per your request) ----
acc_color = "tab:blue"
ece_color = "tab:red"

# ---- Plot ----
fig, ax1 = plt.subplots(figsize=(10, 6))
x = np.arange(len(summary))
width = 0.36  # width of each bar

# Accuracy bars (left axis)
acc_bars = ax1.bar(
    x - width/2, summary["Avg_ACC"].values,
    yerr=summary["Std_ACC"].values, capsize=4, width=width,
    label="Accuracy (%)", color=acc_color, edgecolor="black", linewidth=0.6
)
ax1.set_ylabel("Accuracy (%)", color=acc_color)
ax1.tick_params(axis="y", colors=acc_color)
ax1.set_xticks(x)
ax1.set_xticklabels(summary["Method"].values, rotation=15)
ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)

# ECE bars (right axis)
ax2 = ax1.twinx()
ece_bars = ax2.bar(
    x + width/2, summary["Avg_ECE"].values,
    yerr=summary["Std_ECE"].values, capsize=4, width=width,
    label="ECE (%)", color=ece_color, edgecolor="black", linewidth=0.6
)
ax2.set_ylabel("ECE (%) ↓ (lower is better)", color=ece_color)
ax2.tick_params(axis='y', colors=ece_color)

# Value labels on bars
for rect in acc_bars:
    h = rect.get_height()
    ax1.text(
        rect.get_x() + rect.get_width()/2, h + max(0.6, 0.02*h),
        f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=acc_color
    )
for rect in ece_bars:
    h = rect.get_height()
    ax2.text(
        rect.get_x() + rect.get_width()/2, h + max(0.3, 0.03*h),
        f"{h:.2f}", ha="center", va="bottom", fontsize=9, color=ece_color
    )

# Neat frame (visible spines)
for a in (ax1, ax2):
    for side in ["top", "right", "bottom", "left"]:
        a.spines[side].set_visible(True)
        a.spines[side].set_linewidth(1.2)

# Single legend
handles = [acc_bars, ece_bars]
labels = ["Accuracy (%)", "ECE (%)"]
ax1.legend(handles, labels, loc="upper left", framealpha=0.9)

plt.title("Prompt-Tuning Families: Grouped Accuracy (blue) & ECE (red)")
plt.tight_layout()
plt.savefig("./medical_bm_fig/prompt_tuning_acc_ece_grouped_twoaxis_colored.png", dpi=200, bbox_inches="tight")
#plt.show()

# Save the summary used for plotting
summary.to_csv("./medical_bm_fig/prompt_tuning_acc_ece_grouped_summary.csv", index=False)
