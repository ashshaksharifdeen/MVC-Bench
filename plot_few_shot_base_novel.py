"""
Create a publication-ready, two-panel few-shot accuracy figure.

Panels:
    (a) Base-class accuracy
    (b) Novel-class accuracy

The values below are copied exactly from:
    few_shot_base_novel_accuracy_comparison(1).xlsx

Output:
    base_novel_few_shot_comparison.png
    base_novel_few_shot_comparison.pdf
"""

from pathlib import Path
import numpy as np
import matplotlib

# Use a non-interactive backend so the script works on servers/HPC systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 1. USER-ADJUSTABLE FIGURE SETTINGS
# ============================================================

# AAAI full-width/two-column figure starting point.
# Increase the first value for a wider figure and the second for a taller figure.
FIGURE_SIZE = (7.2, 3.35)  # width, height in inches

# Export quality.
PNG_DPI = 600

# Font family. DejaVu Sans is bundled with Matplotlib and runs reliably.
FONT_FAMILY = "DejaVu Sans"

# Text sizes.
PANEL_TITLE_FONT_SIZE = 11.5
AXIS_LABEL_FONT_SIZE = 10.5
TICK_LABEL_FONT_SIZE = 9.0
LEGEND_FONT_SIZE = 8.4
ANNOTATION_FONT_SIZE = 6.5

# Line and marker settings.
LINE_WIDTH = 1.8
MARKER_SIZE = 5.3
MARKER_EDGE_WIDTH = 0.7

# Grid and border settings.
GRID_LINE_WIDTH = 0.65
GRID_ALPHA = 0.45
AXIS_BORDER_WIDTH = 0.9

# Display point values next to markers.
# False is recommended for a clean AAAI figure.
ANNOTATE_POINT_VALUES = False

# Number of decimal places used only when optional annotations are enabled.
ANNOTATION_DECIMALS = 2

# X-axis configuration.
X_TICKS = [1, 2, 4, 8, 16]
X_AXIS_LIMITS = (0.25, 16.75)

# Y-axis configuration.
# These limits affect only the visible plotting range; they do not alter the data.
BASE_Y_LIMITS = (69.0, 88.5)
BASE_Y_TICKS = np.arange(70, 89, 3)

NOVEL_Y_LIMITS = (65.5, 79.0)
NOVEL_Y_TICKS = np.arange(66, 80, 2)

# Figure spacing. Adjust these only when changing the overall figure dimensions
# or using much larger fonts.
LEFT_MARGIN = 0.095
RIGHT_MARGIN = 0.992
BOTTOM_MARGIN = 0.205
TOP_MARGIN = 0.785
PANEL_HORIZONTAL_SPACE = 0.17

# Legend position.
LEGEND_COLUMNS = 5
LEGEND_X = 0.50
LEGEND_Y = 0.985

# Output filenames.
OUTPUT_DIRECTORY = Path(".hspl_figures/")
PNG_FILENAME = "base_novel_few_shot_comparison.png"
PDF_FILENAME = "base_novel_few_shot_comparison.pdf"


# ============================================================
# 2. EXACT EXPERIMENT VALUES FROM THE UPLOADED WORKBOOK
# ============================================================
#
# Values are intentionally stored as strings so the original decimal fields
# remain directly visible and easy to cross-check. They are converted to float
# only inside the plotting function.
#
# Shot order for every row: 1-shot, 2-shot, 4-shot, 8-shot, 16-shot
# ============================================================

SHOTS = ["1", "2", "4", "8", "16"]

BASE_ACCURACY = {
    "CoOp":    ["72.432",    "76.213",    "77.879",    "80.697667", "83.365333"],
    "CoCoOp":  ["72.751333", "74.917667", "76.707",    "78.622",    "80.599333"],
    "MaPLe":   ["70.818",    "74.510333", "77.248667", "80.225667", "82.364667"],
    "HiCroPL": ["77.139333", "80.051333", "82.263",    "83.652",    "86.047667"],
    "HSPL":    ["77.008",    "79.882333", "82.282",    "83.653",    "87.018"],
}

NOVEL_ACCURACY = {
    "CoOp":    ["67.635",    "68.366667", "67.213",    "68.729",    "67.456333"],
    "CoCoOp":  ["72.747667", "74.619333", "72.866",    "73.039333", "73.370667"],
    "MaPLe":   ["71.003",    "73.402333", "73.377667", "73.461333", "74.001"],
    "HiCroPL": ["75.979",    "75.827667", "75.781667", "76.152",    "76.041333"],
    "HSPL":    ["77.315333", "77.691333", "76.547333", "77.285333", "77.072667"],
}

# Method order used consistently in both panels and the shared legend.
METHOD_ORDER = ["CoOp", "CoCoOp", "MaPLe", "HiCroPL", "HSPL"]

# Different marker shapes make the curves distinguishable in grayscale printing.
# No fixed colors are assigned; Matplotlib's standard publication-safe color
# cycle is used automatically.
METHOD_MARKERS = {
    "CoOp": "o",
    "CoCoOp": "s",
    "MaPLe": "^",
    "HiCroPL": "D",
    "HSPL": "P",
}


# ============================================================
# 3. VALIDATION AND CROSS-CHECK FUNCTIONS
# ============================================================

def validate_accuracy_data(data, table_name):
    """Validate method names, shot count and numeric values."""
    if list(data.keys()) != METHOD_ORDER:
        raise ValueError(
            f"{table_name}: method order must be exactly {METHOD_ORDER}, "
            f"but received {list(data.keys())}."
        )

    for method, values in data.items():
        if len(values) != len(X_TICKS):
            raise ValueError(
                f"{table_name}, {method}: expected {len(X_TICKS)} values, "
                f"but received {len(values)}."
            )

        for shot, value in zip(X_TICKS, values):
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{table_name}, {method}, {shot}-shot has a non-numeric "
                    f"value: {value!r}"
                ) from exc


def print_accuracy_table(title, data):
    """Print the exact source strings for manual cross-checking."""
    method_width = max(len("Method"), max(len(method) for method in METHOD_ORDER))
    value_width = 12

    print("\n" + title)
    print("=" * (method_width + value_width * len(X_TICKS) + 3))
    header = f"{'Method':<{method_width}}" + "".join(
        f"{shot:>{value_width}}" for shot in ["1-shot", "2-shot", "4-shot", "8-shot", "16-shot"]
    )
    print(header)
    print("-" * len(header))

    for method in METHOD_ORDER:
        row = f"{method:<{method_width}}" + "".join(
            f"{value:>{value_width}}" for value in data[method]
        )
        print(row)


# ============================================================
# 4. PLOTTING FUNCTIONS
# ============================================================

def plot_panel(ax, data, title, y_limits, y_ticks, panel_label):
    """Plot one accuracy panel using the exact supplied values."""
    x = np.asarray(X_TICKS, dtype=float)

    for method in METHOD_ORDER:
        y = np.asarray([float(value) for value in data[method]], dtype=float)

        ax.plot(
            x,
            y,
            marker=METHOD_MARKERS[method],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=method,
            zorder=3,
        )

        if ANNOTATE_POINT_VALUES:
            for x_value, y_value in zip(x, y):
                ax.annotate(
                    f"{y_value:.{ANNOTATION_DECIMALS}f}",
                    xy=(x_value, y_value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=ANNOTATION_FONT_SIZE,
                    clip_on=False,
                )

    ax.set_title(
        f"{panel_label} {title}",
        fontsize=PANEL_TITLE_FONT_SIZE,
        fontweight="bold",
        pad=7,
    )

    ax.set_xlim(*X_AXIS_LIMITS)
    ax.set_xticks(X_TICKS)
    ax.set_ylim(*y_limits)
    ax.set_yticks(y_ticks)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_LABEL_FONT_SIZE,
        width=AXIS_BORDER_WIDTH,
        length=3.5,
    )

    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=GRID_LINE_WIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_BORDER_WIDTH)


def create_figure():
    """Create and save the complete two-panel figure."""
    validate_accuracy_data(BASE_ACCURACY, "Base accuracy")
    validate_accuracy_data(NOVEL_ACCURACY, "Novel accuracy")

    # Print exact values before plotting for easy verification.
    print_accuracy_table("BASE-CLASS ACCURACY: EXACT SOURCE VALUES", BASE_ACCURACY)
    print_accuracy_table("NOVEL-CLASS ACCURACY: EXACT SOURCE VALUES", NOVEL_ACCURACY)

    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.linewidth": AXIS_BORDER_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": PNG_DPI,
        }
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=FIGURE_SIZE,
        sharex=True,
    )

    plot_panel(
        ax=axes[0],
        data=BASE_ACCURACY,
        title="Base Classes",
        y_limits=BASE_Y_LIMITS,
        y_ticks=BASE_Y_TICKS,
        panel_label="(a)",
    )

    plot_panel(
        ax=axes[1],
        data=NOVEL_ACCURACY,
        title="Novel Classes",
        y_limits=NOVEL_Y_LIMITS,
        y_ticks=NOVEL_Y_TICKS,
        panel_label="(b)",
    )

    # Shared labels avoid repeated text and save horizontal space.
    fig.supxlabel(
        "Number of training shots per class",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="medium",
        y=0.055,
    )
    fig.supylabel(
        "Accuracy (%)",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="medium",
        x=0.012,
    )

    # One shared legend prevents duplication and leaves the data panels clear.
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(LEGEND_X, LEGEND_Y),
        ncol=LEGEND_COLUMNS,
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.25,
        handletextpad=0.45,
    )

    fig.subplots_adjust(
        left=LEFT_MARGIN,
        right=RIGHT_MARGIN,
        bottom=BOTTOM_MARGIN,
        top=TOP_MARGIN,
        wspace=PANEL_HORIZONTAL_SPACE,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / PNG_FILENAME
    pdf_path = OUTPUT_DIRECTORY / PDF_FILENAME

    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)

    print(f"\nSaved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")


if __name__ == "__main__":
    create_figure()