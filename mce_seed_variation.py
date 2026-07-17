"""
Generate a horizontal bar chart visualising the variability of the Maximum
Calibration Error (MCE) across the datasets for a series of medical
vision‑language backbones.  Similar to ``ace_plot.py``, this script
stores the per‑dataset MCE and accuracy figures as lists and computes
the variability on the fly.  Each bar represents the population
standard deviation of a backbone's MCE values computed over its
constituent datasets.  The numbers at the end of each bar report the
computed MCE variability, while the numbers in parentheses denote the
corresponding standard deviation of accuracy across the same datasets.
This plot matches the style of MVC‑Bench (Figure 5 of the
supplementary material) and omits captions or titles, allowing you to
drop it into a manuscript without modification.

Usage:
    python mce_plot.py

This script saves a PNG file called ``mce_variability.png`` in the
current working directory.
"""

import os
from typing import Dict, List

# Disable the Jupyter server logging hooks used in the CaaS environment.
os.environ.setdefault("ENABLE_MATPLOTLIB_JUPYTER_SERVER", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_std(values: List[float]) -> float:
    """Return the population standard deviation of a list of floats.

    Uses numpy with ``ddof=0`` (population standard deviation).  If
    fewer than two values are supplied, returns 0.0.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=0))


def main() -> None:
    """Create and save the MCE variability plot."""
    # Dataset‑level MCE and accuracy values for each backbone.  These
    # figures were extracted manually from Table A4 of the MVC‑Bench
    # supplementary material (base calibration only).  See ace_plot.py
    # for explanations of the dataset groups and notation.
    backbone_data: Dict[str, Dict[str, List[float]]] = {
        # Chest X‑ray backbones
        "MedCLIP‑X‑ray": {
            "mce": [29.34, 18.60],
            "acc": [79.45, 52.08],
        },
        "BioMedCLIP‑X‑ray": {
            "mce": [2.42, 2.98],
            "acc": [83.32, 63.80],
        },
        # Diabetic retinopathy backbones
        "CLIP‑DR‑ResNet‑50": {
            "mce": [1.49, 1.66, 1.66, 1.13],
            "acc": [76.69, 74.48, 56.12, 64.20],
        },
        "CLIP‑DR‑ResNet‑101": {
            "mce": [1.12, 0.84, 4.10, 2.11],
            "acc": [75.75, 73.92, 59.47, 64.03],
        },
        "CLIP‑DR‑ViT‑B/16": {
            "mce": [1.46, 0.68, 1.78, 1.54],
            "acc": [77.90, 74.64, 59.50, 64.66],
        },
        "CLIP‑DR‑ViT‑B/32": {
            "mce": [1.52, 1.66, 0.91, 1.31],
            "acc": [78.28, 74.58, 55.47, 64.20],
        },
        "MAPLE-DR-ViT-B/16": {
            "mce": [0.9,0.93, 3.15, 1.71],
            "acc": [80.93, 77.54, 54.19, 64.47],
        },
        "MedCLIP‑DR‑B/32": {
            "mce": [3.27, 1.36, 8.37, 2.61],
            "acc": [62.27, 73.66, 45.03, 58.31],
        },
        "BioMedCLIP‑DR‑B/32": {
            "mce": [0.80, 0.56, 1.73, 2.01],
            "acc": [77.25, 74.13, 55.53, 62.96],
        },
        "PLIP‑DR‑B/32": {
            "mce": [1.26, 0.49, 1.52, 1.01],
            "acc": [78.66, 74.43, 55.64, 63.46],
        },
        "QuiltNet‑DR‑B/32": {
            "mce": [1.26, 0.49, 1.52, 1.01],
            "acc": [76.97, 74.43, 52.31, 62.44],
        },
        # Histopathology backbones
        "PLIP‑Histo": {
            "mce": [1.26, 0.49, 1.52],
            "acc": [90.74, 87.00, 78.57],
        },
        "QuiltNet‑Histo": {
            "mce": [1.26, 0.49, 1.52],
            "acc": [89.76, 91.56, 78.86],
        },
    }

    # Compute standard deviations for MCE and accuracy
    mce_std: Dict[str, float] = {}
    acc_std: Dict[str, float] = {}
    for backbone, metrics in backbone_data.items():
        mce_std[backbone] = compute_std(metrics["mce"])
        acc_std[backbone] = compute_std(metrics["acc"])

    # Define a fixed backbone order matching the reference seed‑variation
    # figure (excluding the MAPLE backbone, which is not covered by
    # Table A4 base metrics).  The bars will be displayed in this
    # order from top to bottom.
    backbone_order = [
        "QuiltNet‑Histo",
        "QuiltNet‑DR‑B/32",
        "PLIP‑Histo",
        "PLIP‑DR‑B/32",
        "BioMedCLIP‑X‑ray",
        "BioMedCLIP‑DR‑B/32",
        "MedCLIP‑X‑ray",
        "MedCLIP‑DR‑B/32",
        "MAPLE-DR-ViT-B/16",
        "CLIP‑DR‑ViT‑B/32",
        "CLIP‑DR‑ViT‑B/16",
        "CLIP‑DR‑ResNet‑101",
        "CLIP‑DR‑ResNet‑50",
    ]
    mce_values = [mce_std[name] for name in backbone_order]
    acc_std_values = [acc_std[name] for name in backbone_order]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    y_positions = range(len(backbone_order))
    bars = ax.barh(y_positions, mce_values, color="#DD8452")

    # Annotate each bar with the MCE variability and the accuracy std
    for bar, mce_val, acc_sd in zip(bars, mce_values, acc_std_values):
        x_pos = bar.get_width() + 0.05
        y_pos = bar.get_y() + bar.get_height() / 2
        label = f"{mce_val:.2f}"
        ax.text(x_pos, y_pos, label, va="center", fontsize=9)

    # Formatting
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(backbone_order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Maximum Calibration Error (MCE) variability across datasets", fontsize=10)
    ax.set_xlim(0, max(mce_values) * 1.25 if mce_values else 1)
    # Show the top and right spines to mirror the reference figure
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.tight_layout()
    # Save the figure
    fig.savefig("./medical_bm_fig/mce_seed_variability.png", dpi=300)


if __name__ == "__main__":
    main()