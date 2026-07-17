# ============================================================
# Clean EMNLP-ready Multi-Modality Brier Score Plot
# Fundus + Histopathology + X-ray
# - Keeps all original data values unchanged
# - Computes mean/std across datasets within each modality-method
# - Uses one shared y-axis because both bars are Brier scores
# - Avoids overlapping value labels, modality labels, and x-tick labels
# - Saves PNG, PDF, SVG, summary CSV, and raw tidy CSV
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Output folder
# ============================================================

OUT_DIR = "./medical_bm_fig"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 2. Raw input data
# ============================================================
# IMPORTANT:
# Do not change the order of values inside each list.
# Each list follows the dataset order defined for that modality.
# ============================================================

raw = {
    # --------------------------------------------------------
    # Modality 1: Fundus
    # --------------------------------------------------------
    "Fundus": {
        "datasets": ["APTOS", "EyePACS", "Messidor", "Messidor_2"],
        "methods": {
            "CLIP-DR-ResNet-50": {
                "Vanilla Method": [0.32272, 0.39492, 0.567198, 0.490857],
                "Ours": [0.315848, 0.40024, 0.550047, 0.489354],
            },
            "CLIP-DR-ResNet-101": {
                "Vanilla Method": [0.324625, 0.398325, 0.565503, 0.493944],
                "Ours": [0.314723, 0.402607, 0.562363, 0.48895],
            },
            "CLIP-DR-ViT-B/16": {
                "Vanilla Method": [0.305034, 0.385071, 0.539631, 0.486312],
                "Ours": [0.30673, 0.371616, 0.539434, 0.483851],
            },
            "CLIP-DR-ViT-B/32": {
                "Vanilla Method": [0.301702, 0.388351, 0.573425, 0.491492],
                "Ours": [0.30583, 0.370455, 0.563359, 0.481934],
            },
            "MedCLIP-DR-B/32": {
                "Vanilla Method": [0.563917, 0.419636, 0.697619333, 0.581028333],
                "Ours": [0.563441333, 0.424214333, 0.687935, 0.570168667],
            },
            "BioMedCLIP-DR-B/32": {
                "Vanilla Method": [0.320678333, 0.401085667, 0.582377, 0.503538667],
                "Ours": [0.317174, 0.404839, 0.581467, 0.500357667],
            },
            "PLIP-DR-B/32": {
                "Vanilla Method": [0.302615, 0.390601, 0.575128333, 0.486529333],
                "Ours": [0.302192, 0.384551333, 0.571161333, 0.489126],
            },
            "QuiltNet-DR-B/32": {
                "Vanilla Method": [0.323384, 0.393123, 0.60206, 0.503167],
                "Ours": [0.317586, 0.394186667, 0.601065333, 0.502882667],
            },
        },
    },

    # --------------------------------------------------------
    # Modality 2: Histopathology
    # --------------------------------------------------------
    "Histopathology": {
        "datasets": ["DigestPath", "Kather", "Pannuke"],
        "methods": {
            "PLIP-Histo": {
                "Vanilla Method": [0.137915, 0.197844, 0.30347],
                "Ours": [0.17703, 0.20232, 0.360685],
            },
            "QuiltNet-Histo": {
                "Vanilla Method": [0.153295, 0.122077, 0.309384],
                "Ours": [0.140513, 0.137778, 0.337144],
            },
        },
    },

    # --------------------------------------------------------
    # Modality 3: X-ray
    # --------------------------------------------------------
    "X-ray": {
        "datasets": ["Rsna18", "CovidX"],
        "methods": {
            "MedCLIP-X-ray": {
                "Vanilla Method": [0.665873, 0.498611],
                "Ours": [0.655861, 0.498606],
            },
            "BioMedCLIP-X-ray": {
                "Vanilla Method": [0.253404, 0.484472],
                "Ours": [0.242499, 0.496302],
            },
        },
    },
}


# ============================================================
# 3. Display order
# ============================================================

modality_order = ["Fundus", "Histopathology", "X-ray"]

method_order = {
    "Fundus": [
        "CLIP-DR-ResNet-50",
        "CLIP-DR-ResNet-101",
        "CLIP-DR-ViT-B/16",
        "CLIP-DR-ViT-B/32",
        "MedCLIP-DR-B/32",
        "BioMedCLIP-DR-B/32",
        "PLIP-DR-B/32",
        "QuiltNet-DR-B/32",
    ],
    "Histopathology": ["PLIP-Histo", "QuiltNet-Histo"],
    "X-ray": ["MedCLIP-X-ray", "BioMedCLIP-X-ray"],
}


# ============================================================
# 4. Publication-style controls
# ============================================================

FIG_WIDTH = 18.5
FIG_HEIGHT = 12.0
DPI = 600

TITLE_SIZE = 20
X_AXIS_LABEL_SIZE = 18
Y_AXIS_LABEL_SIZE = 18
X_TICK_LABEL_SIZE = 15
Y_TICK_LABEL_SIZE = 15
VALUE_LABEL_SIZE = 12
MODALITY_LABEL_SIZE = 14
LEGEND_SIZE = 14

BAR_WIDTH = 0.34
ERROR_BAR_CAPSIZE = 4
ERROR_BAR_LINEWIDTH = 1.25
BAR_EDGE_LINEWIDTH = 0.6
VALUE_LABEL_OFFSET = 0.018

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
})


# ============================================================
# 5. Validate input data
# ============================================================

def validate_raw_data(raw_data):
    for modality, modality_info in raw_data.items():
        datasets = modality_info["datasets"]
        n_datasets = len(datasets)

        for method, values in modality_info["methods"].items():
            vanilla_values = values.get("Vanilla Method", [])
            ours_values = values.get("Ours", [])

            if len(vanilla_values) != n_datasets:
                raise ValueError(
                    f"[DATA ERROR] {modality} -> {method} has "
                    f"{len(vanilla_values)} Vanilla Method values, but "
                    f"{n_datasets} datasets are defined: {datasets}"
                )

            if len(ours_values) != n_datasets:
                raise ValueError(
                    f"[DATA ERROR] {modality} -> {method} has "
                    f"{len(ours_values)} Ours values, but "
                    f"{n_datasets} datasets are defined: {datasets}"
                )


validate_raw_data(raw)


# ============================================================
# 6. Convert raw dictionary to tidy DataFrame
# ============================================================

rows = []

for modality in modality_order:
    modality_info = raw[modality]
    datasets = modality_info["datasets"]
    methods = modality_info["methods"]

    for method in method_order[modality]:
        values = methods[method]

        for i, dataset in enumerate(datasets):
            rows.append({
                "Modality": modality,
                "Dataset": dataset,
                "Method": method,
                "Vanilla_Method": float(values["Vanilla Method"][i]),
                "Ours": float(values["Ours"][i]),
            })

df = pd.DataFrame(rows)

raw_csv_path = os.path.join(OUT_DIR, "multimodality_brierscore_raw_tidy_results.csv")
df.to_csv(raw_csv_path, index=False)


# ============================================================
# 7. Calculate mean and std across datasets
# ============================================================

summary = (
    df.groupby(["Modality", "Method"])[["Vanilla_Method", "Ours"]]
      .agg(["mean", "std"])
      .reset_index()
)

summary.columns = [
    "Modality",
    "Method",
    "Avg_Vanilla_Method",
    "Std_Vanilla_Method",
    "Avg_Ours",
    "Std_Ours",
]

summary["Std_Vanilla_Method"] = summary["Std_Vanilla_Method"].fillna(0.0)
summary["Std_Ours"] = summary["Std_Ours"].fillna(0.0)

ordered_rows = []

for modality in modality_order:
    for method in method_order[modality]:
        selected = summary[
            (summary["Modality"] == modality) &
            (summary["Method"] == method)
        ]

        if selected.empty:
            raise ValueError(f"[ORDER ERROR] Missing {modality} -> {method} in summary.")

        ordered_rows.append(selected)

summary = pd.concat(ordered_rows, ignore_index=True)

summary_csv_path = os.path.join(OUT_DIR, "multimodality_brierscore_summary.csv")
summary.to_csv(summary_csv_path, index=False)


# ============================================================
# 8. Create compact x-axis labels
# ============================================================

def make_display_label(method_name):
    label_map = {
        "CLIP-DR-ResNet-50": "CLIP-DR\nResNet-50",
        "CLIP-DR-ResNet-101": "CLIP-DR\nResNet-101",
        "CLIP-DR-ViT-B/16": "CLIP-DR\nViT-B/16",
        "CLIP-DR-ViT-B/32": "CLIP-DR\nViT-B/32",
        "MedCLIP-DR-B/32": "MedCLIP\nDR-B/32",
        "BioMedCLIP-DR-B/32": "BioMedCLIP\nDR-B/32",
        "PLIP-DR-B/32": "PLIP\nDR-B/32",
        "QuiltNet-DR-B/32": "QuiltNet\nDR-B/32",
        "PLIP-Histo": "PLIP\nHisto",
        "QuiltNet-Histo": "QuiltNet\nHisto",
        "MedCLIP-X-ray": "MedCLIP\nX-ray",
        "BioMedCLIP-X-ray": "BioMedCLIP\nX-ray",
    }

    return label_map.get(method_name, method_name)


summary["Display_Label"] = summary["Method"].apply(make_display_label)


# ============================================================
# 9. Plot clean grouped Brier score chart
# ============================================================

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

x = np.arange(len(summary))

vanilla_color = "tab:blue"
ours_color = "tab:red"

vanilla_bars = ax.bar(
    x - BAR_WIDTH / 2,
    summary["Avg_Vanilla_Method"].values,
    yerr=summary["Std_Vanilla_Method"].values,
    capsize=ERROR_BAR_CAPSIZE,
    width=BAR_WIDTH,
    label="Vanilla Method / CE",
    color=vanilla_color,
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    error_kw={
        "elinewidth": ERROR_BAR_LINEWIDTH,
        "capthick": ERROR_BAR_LINEWIDTH,
        "ecolor": "black",
    },
    zorder=3,
)

ours_bars = ax.bar(
    x + BAR_WIDTH / 2,
    summary["Avg_Ours"].values,
    yerr=summary["Std_Ours"].values,
    capsize=ERROR_BAR_CAPSIZE,
    width=BAR_WIDTH,
    label="Ours",
    color=ours_color,
    edgecolor="black",
    linewidth=BAR_EDGE_LINEWIDTH,
    error_kw={
        "elinewidth": ERROR_BAR_LINEWIDTH,
        "capthick": ERROR_BAR_LINEWIDTH,
        "ecolor": "black",
    },
    zorder=3,
)


# ============================================================
# 10. Axes, ticks, grid, and title
# ============================================================

ax.set_ylabel(
    "Brier Score ↓ (lower is better)",
    fontsize=Y_AXIS_LABEL_SIZE,
    fontweight="bold",
    labelpad=10,
)

ax.set_xlabel(
    "Methods grouped by modality",
    fontsize=X_AXIS_LABEL_SIZE,
    fontweight="bold",
    labelpad=65,
)

ax.set_title(
    "Multi-Modality Prompt-Tuning Performance: Brier Score Comparison",
    fontsize=TITLE_SIZE,
    fontweight="bold",
    pad=18,
)

ax.set_xticks(x)
ax.set_xticklabels(
    summary["Display_Label"].values,
    rotation=0,
    ha="center",
    fontsize=X_TICK_LABEL_SIZE,
    linespacing=1.15,
)

ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_SIZE)
ax.tick_params(axis="x", length=0, pad=8)

ax.grid(
    axis="y",
    alpha=0.28,
    linestyle="--",
    linewidth=0.8,
    zorder=0,
)

vanilla_top = summary["Avg_Vanilla_Method"].values + summary["Std_Vanilla_Method"].values
ours_top = summary["Avg_Ours"].values + summary["Std_Ours"].values
max_top = max(np.max(vanilla_top), np.max(ours_top))
ymax = max_top + 0.09
ax.set_ylim(0, ymax)


# ============================================================
# 11. Add value labels above error bars
# ============================================================

def add_value_labels(axis, bars, errors, color):
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        label_y = height + float(err) + VALUE_LABEL_OFFSET

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_SIZE,
            color=color,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
                "pad": 1.0,
            },
            clip_on=False,
            zorder=5,
        )


add_value_labels(
    ax,
    vanilla_bars,
    summary["Std_Vanilla_Method"].values,
    vanilla_color,
)

add_value_labels(
    ax,
    ours_bars,
    summary["Std_Ours"].values,
    ours_color,
)


# ============================================================
# 12. Add modality separators and group labels
# ============================================================

current_position = 0

for modality in modality_order:
    n_methods = len(method_order[modality])

    start = current_position
    end = current_position + n_methods - 1
    center = (start + end) / 2

    ax.text(
        center,
        -0.105,
        modality,
        ha="center",
        va="top",
        fontsize=MODALITY_LABEL_SIZE,
        fontweight="bold",
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )

    if modality != modality_order[-1]:
        ax.axvline(
            end + 0.5,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )

    current_position += n_methods


# ============================================================
# 13. Frame, legend, and layout
# ============================================================

for side in ["top", "right", "bottom", "left"]:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_linewidth(1.1)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    frameon=True,
    framealpha=0.96,
    edgecolor="lightgray",
    fontsize=LEGEND_SIZE,
    ncol=2,
)

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    top=0.86,
    bottom=0.28,
)


# ============================================================
# 14. Save figure and tables
# ============================================================

png_path = os.path.join(OUT_DIR, "multimodality_brierscore_emnlp_clean.png")
pdf_path = os.path.join(OUT_DIR, "multimodality_brierscore_emnlp_clean.pdf")
svg_path = os.path.join(OUT_DIR, "multimodality_brierscore_emnlp_clean.svg")

fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

# Uncomment this during local debugging if you want to display the figure.
# plt.show()

plt.close(fig)


# ============================================================
# 15. Print useful outputs
# ============================================================

print("\nSaved files:")
print(f"1. PNG figure: {png_path}")
print(f"2. PDF figure: {pdf_path}")
print(f"3. SVG figure: {svg_path}")
print(f"4. Summary CSV: {summary_csv_path}")
print(f"5. Raw tidy CSV: {raw_csv_path}")

print("\nSummary:")
print(summary)