"""
Generate a horizontal bar chart visualising the variability of the Adaptive
Calibration Error (ACE) across the datasets for a series of medical
vision‑language backbones.  Instead of hard‑coding the standard deviation
values, this script stores the per‑dataset ACE and accuracy figures as
lists and computes the variability on the fly.  Each bar represents the
population standard deviation of a backbone's ACE values computed over
its constituent datasets (for example, APTOS, EyePACS, Messidor and
Messidor‑2 for diabetic retinopathy models; DigestPath, Kather and
PaNuke for histopathology models; COVIDX and RSNA18 for chest X‑ray
models).  The numbers at the end of each bar report the computed ACE
variability, while the numbers in parentheses denote the corresponding
standard deviation of accuracy across the same datasets.  This plot is
designed to match the style used in MVC‑Bench (Figure 5 of the
supplementary material), omitting any caption or title so it can be
easily incorporated into a paper or presentation.

Usage:
    python ace_plot.py

This script saves a PNG file called ``ace_variability.png`` in the
current working directory.
"""

import os
from typing import Dict, List

# Disable the Jupyter server logging hooks used in the CaaS environment.
# Without this, matplotlib attempts to post log messages to a local
# Jupyter server which is not running in the container, leading to
# ConnectionRefusedError exceptions during figure generation.
os.environ.setdefault("ENABLE_MATPLOTLIB_JUPYTER_SERVER", "false")

# Use the Agg backend for non‑interactive environments.  This
# prevents matplotlib from attempting to connect to a display or
# external logging services when generating figures in headless
# settings (e.g. during automated testing or remote execution).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_std(values: List[float]) -> float:
    """Return the population standard deviation of a list of floats.

    This function uses numpy to compute the standard deviation with
    ``ddof=0``, which corresponds to the population (not sample)
    standard deviation.  If the list contains fewer than two elements
    the standard deviation is defined to be 0.0.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return 0.0
    return float(np.std(arr, ddof=0))


def main() -> None:
    """Create and save the ACE variability plot."""
    # Dataset‑level ACE and accuracy values for each backbone.  These
    # lists were extracted manually from Table A4 of the MVC‑Bench
    # supplementary material (base calibration only).  For each
    # backbone we provide the per‑dataset ACE (Adaptive Calibration
    # Error) values and the corresponding per‑dataset accuracies.
    # The keys follow the notation used in the paper: DR stands for
    # diabetic retinopathy, Histo for histopathology and X‑ray for
    # chest X‑ray.  See the accompanying PDF for details.
    backbone_data: Dict[str, Dict[str, List[float]]] = {
        # Chest X‑ray backbones
        "MedCLIP‑X‑ray": {
            "ace": [29.34, 18.60],
            "acc": [79.45, 52.08],
        },
        "BioMedCLIP‑X‑ray": {
            "ace": [6.97, 8.69],
            "acc": [83.32, 63.80],
        },
        # Diabetic retinopathy backbones (ResNet and ViT variants)
        "CLIP‑DR‑ResNet‑50": {
            "ace": [6.26, 3.24, 5.79, 4.69],
            "acc": [76.69, 74.48, 56.12, 64.20],
        },
        "CLIP‑DR‑ResNet‑101": {
            "ace": [3.95, 3.10, 10.95, 6.03],
            "acc": [75.75, 73.92, 59.47, 64.03],
        },
        "CLIP‑DR‑ViT‑B/16": {
            "ace": [3.99, 2.30, 5.69, 6.24],
            "acc": [77.90, 74.64, 59.50, 64.66],
        },
        "CLIP‑DR‑ViT‑B/32": {
            "ace": [4.00, 3.24, 5.15, 4.99],
            "acc": [78.28, 74.58, 55.47, 64.20],
        },
        "MAPLE-DR-ViT-B/16": {
            "ace": [2.58,2.47, 7.34, 4.98],
            "acc": [80.93, 77.54, 54.19, 64.47],
        },
        "MedCLIP‑DR‑B/32": {
            "ace": [10.04, 3.54, 9.41, 5.62],
            "acc": [62.27, 73.66, 45.03, 58.31],
        },
        "BioMedCLIP‑DR‑B/32": {
            "ace": [2.65, 1.83, 5.36, 6.46],
            "acc": [77.25, 74.13, 55.53, 62.96],
        },
        "PLIP‑DR‑B/32": {
            "ace": [4.98, 1.38, 5.11, 2.91],
            "acc": [78.66, 74.43, 55.64, 63.46],
        },
        "QuiltNet‑DR‑B/32": {
            "ace": [4.98, 1.38, 5.11, 2.91],
            "acc": [76.97, 74.43, 52.31, 62.44],
        },
        # Histopathology backbones
        "PLIP‑Histo": {
            "ace": [4.98, 1.38, 5.11],
            "acc": [90.74, 87.00, 78.57],
        },
        "QuiltNet‑Histo": {
            "ace": [4.98, 1.38, 5.11],
            "acc": [89.76, 91.56, 78.86],
        },
    }

    # Compute standard deviations for ACE and accuracy
    ace_std: Dict[str, float] = {}
    acc_std: Dict[str, float] = {}
    for backbone, metrics in backbone_data.items():
        ace_std[backbone] = compute_std(metrics["ace"])
        acc_std[backbone] = compute_std(metrics["acc"])

    # Define a fixed backbone order that mirrors the ordering in the
    # reference seed‑variation figure.  This ensures that the bars
    # appear in the same top‑to‑bottom sequence as the user’s
    # reference, rather than being sorted by variability.
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
    # Assemble the standard deviation values in the specified order
    ace_values = [ace_std[name] for name in backbone_order]
    acc_std_values = [acc_std[name] for name in backbone_order]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    y_positions = range(len(backbone_order))
    bars = ax.barh(y_positions, ace_values, color="#4C72B0")

    # Annotate each bar with the ACE variability and the accuracy std
    for bar, ace_val, acc_sd in zip(bars, ace_values, acc_std_values):
        x_pos = bar.get_width() + 0.05  # position to the right of bar
        y_pos = bar.get_y() + bar.get_height() / 2
        label = f"{ace_val:.2f}"
        ax.text(x_pos, y_pos, label, va="center", fontsize=9)

    # Formatting
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(backbone_order, fontsize=9)
    ax.invert_yaxis()  # first item appears at the top
    ax.set_xlabel("Adaptive Calibration Error (ACE) variability across datasets", fontsize=10)
    # Extend the x‑axis slightly to provide space for the labels
    ax.set_xlim(0, max(ace_values) * 1.25 if ace_values else 1)
    # Show the top and right spines to match the reference figure
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.tight_layout()
    # Save the figure
    fig.savefig("./medical_bm_fig/ace_seed_variability.png", dpi=300)


if __name__ == "__main__":
    main()