# ============================================================
# Clean EMNLP-ready Multi‑Modality Classwise‑ECE Plot
#
# This script produces publication‑quality figures and summary
# tables for comparing calibration performance across multiple
# imaging modalities.  The figure displays Classwise‑ECE
# (Expected Calibration Error computed on a per‑class basis) for
# vanilla cross‑entropy models and a calibrated variant ("Ours").
# Values are drawn from the provided spreadsheet and must follow
# the exact dataset order for each modality.  The script
# automatically computes mean and standard deviation across
# datasets, draws a grouped bar chart with error bars, and
# outputs PNG/PDF/SVG figures along with tidy CSV summaries.
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
raw = {
    # --------------------------------------------------------
    # Modality 1: Fundus
    # --------------------------------------------------------
    "Fundus": {
        "datasets": ["APTOS", "EyePACS", "Messidor", "Messidor_2"],
        "methods": {
            # The following values correspond to the Classwise‑ECE
            # reported in the spreadsheet for each backbone.  The
            # "Vanilla Method" lists map to the row labelled
            # "Classwise‑ECE" under the CE (cross‑entropy) section,
            # while the "Ours" lists map to the row labelled
            # "Classwise‑ECE" under the Ours section.  Do not
            # rearrange the numbers or dataset order.
            "CLIP-DR-ResNet-50": {
                "Vanilla Method": [22.22, 33.45, 34.45, 33.42],
                "Ours": [19.96, 31.33, 34.6, 31.03],
            },
            "CLIP-DR-ResNet-101": {
                "Vanilla Method": [23.13, 34.47, 31.42, 31.74],
                "Ours": [21.53, 32.49, 31.26, 30.65],
            },
            "CLIP-DR-ViT-B/16": {
                "Vanilla Method": [20.79, 32.42, 32.11, 33.1],
                "Ours": [19.14, 30.33, 30.69, 31.67],
            },
            "CLIP-DR-ViT-B/32": {
                "Vanilla Method": [17.43, 33.32, 28.24, 29.27],
                "Ours": [16.89, 30.87, 28.17, 27.41],
            },
            "MedCLIP-DR-B/32": {
                "Vanilla Method": [34.546667, 37.796667, 48.956667, 48.04],
                "Ours": [34.09, 35.583333, 48.623333, 46.733333],
            },
            "BioMedCLIP-DR-B/32": {
                "Vanilla Method": [15.99, 34.853333, 36.326667, 31.346667],
                "Ours": [14.15, 32.39, 36.21, 28.62],
            },
            "PLIP-DR-B/32": {
                "Vanilla Method": [15.283333, 33.08, 28.786667, 28.813333],
                "Ours": [13.82, 30.556667, 27.97, 26.02],
            },
            "QuiltNet-DR-B/32": {
                "Vanilla Method": [18.396667, 34.426667, 25.78, 29.236667],
                "Ours": [16.836667, 32.216667, 24.61, 26.343333],
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
                # Classwise‑ECE for PLIP backbone on histopathology
                # datasets.  The values correspond to the CE row
                # (Vanilla Method) and Ours row in the spreadsheet.
                "Vanilla Method": [4.536667, 8.133333, 10.623333],
                "Ours": [7.78, 8.456667, 15.19],
            },
            "QuiltNet-Histo": {
                "Vanilla Method": [5.033333, 4.966667, 10.426667],
                "Ours": [6.983333, 5.653333, 14.006667],
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
                "Vanilla Method": [40.306667, 33.133333],
                "Ours": [39.38, 33.106667],
            },
            "BioMedCLIP-X-ray": {
                "Vanilla Method": [11.493333, 17.136667],
                "Ours": [8.84, 17.54],
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
# 4. Publication‑style controls
# ============================================================

FIG_WIDTH = 18.5
FIG_HEIGHT = 12.0
DPI = 600

TITLE_SIZE = 20
X_AXIS_LABEL_SIZE = 18
Y_AXIS_LABEL_SIZE = 18
X_TICK_LABEL_SIZE = 15
Y_TICK_LABEL_SIZE = 15
VALUE_LABEL_SIZE = 11
MODALITY_LABEL_SIZE = 14
LEGEND_SIZE = 14

BAR_WIDTH = 0.36
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

def validate_raw_data(raw_data: dict) -> None:
    """Ensure that each method's lists match the number of datasets."""
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

raw_csv_path = os.path.join(OUT_DIR, "multimodality_classwise_ece_raw_tidy_results.csv")
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

summary_csv_path = os.path.join(OUT_DIR, "multimodality_classwise_ece_summary.csv")
summary.to_csv(summary_csv_path, index=False)


# ============================================================
# 8. Create compact x-axis labels
# ============================================================

def make_display_label(method_name: str) -> str:
    """Return a concise two‑line label for long backbone names."""
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
# 9. Plot clean grouped Classwise‑ECE chart
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
    "Classwise‑ECE ↓ (lower is better)",
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
    "Multi‑Modality Prompt‑Tuning Performance: Classwise‑ECE Comparison",
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

# Determine a y‑axis limit that leaves space for value labels
vanilla_top = summary["Avg_Vanilla_Method"].values + summary["Std_Vanilla_Method"].values
ours_top = summary["Avg_Ours"].values + summary["Std_Ours"].values
max_top = max(np.max(vanilla_top), np.max(ours_top))
ymax = max_top + max_top * 0.15 + 2  # leave ~15% headroom plus a small margin
ax.set_ylim(0, ymax)


# ============================================================
# 11. Add value labels above error bars
# ============================================================

def add_value_labels(axis, bars, errors, color):
    """Attach a text label above each bar displaying its height."""
    for bar, err in zip(bars, errors):
        height = bar.get_height()
        label_y = height + float(err) + VALUE_LABEL_OFFSET * max_top
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
    bottom=0.30,
)


# ============================================================
# 14. Save figure and tables
# ============================================================

png_path = os.path.join(OUT_DIR, "multimodality_classwise_ece_emnlp_clean.png")
pdf_path = os.path.join(OUT_DIR, "multimodality_classwise_ece_emnlp_clean.pdf")
svg_path = os.path.join(OUT_DIR, "multimodality_classwise_ece_emnlp_clean.svg")

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