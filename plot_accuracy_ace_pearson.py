import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

"""
This script constructs a Pearson correlation heatmap between classification
accuracy and Adaptive Calibration Error (ACE) for the in‑distribution
results presented in Table A4 of the MVC‑Bench supplementary material.

Table A4 reports per‑dataset accuracy and calibration metrics for eight
calibration baselines—Base, MDCA, LS, MBLS, ECCV_ZS, ECCV_Penalty,
Temperature Scaling (TS) and a Zero‑Shot baseline—across 12
backbones.  The “Ours (MCM)” column is intentionally excluded from
correlation computation since it represents a different loss function.

For each backbone, the correlation for a given method is computed
across all datasets that backbone was evaluated on.  For example,
diabetic retinopathy backbones have four datasets (APTOS, EyePACS,
Messidor, Messidor‑2), histopathology backbones have three datasets
(DigestPath, Kather, PaNuke), and chest‑X‑ray backbones have two
datasets (COVIDX, RSNA18).  The script sets no special case for
backbones with only a single dataset—if such a case arises, NumPy returns
nan for the correlation and we replace it with 1.0 for plotting.

To extend or modify the data, edit the `backbones` dictionary below.  Each
dataset entry contains a list of eight accuracy values and a list of
eight ACE values corresponding to the calibration methods defined in
`methods`.  The order must match exactly.

Running this script produces a heatmap saved as `pearson_heatmap_ace.png`.
"""

# Order of calibration methods (excluding the “Ours (MCM)” column)
methods = [
    "Base",
    "MDCA",
    "LS",
    "MBLS",
    "ECCV_ZS",
    "ECCV_Penalty",
    "TS",
    "zeroshot",
]

# Per‑backbone, per‑dataset accuracy and ACE values transcribed from Table A4.
# Each list contains eight numbers corresponding to the methods above.
backbones = {
    "CLIP-DR-ResNet-50": {
        "APTOS": {
            "accuracy": [76.69, 76.19, 76.64, 76.82, 75.34, 66.72, 78.59, 8.62],
            "ace":      [6.26, 5.15, 9.81, 7.00, 39.31, 33.31, 33.31, 26.47],
        },
        "EyePACS": {
            "accuracy": [74.48, 74.06, 74.02, 74.12, 74.40, 73.34, 74.57, 7.39],
            "ace":      [3.24, 2.39, 3.84, 2.22, 13.08, 38.52, 38.52, 20.41],
        },
        "Messidor": {
            "accuracy": [56.12, 54.44, 57.19, 57.06, 75.34, 44.70, 55.69, 20.58],
            "ace":      [5.79, 5.79, 5.36, 4.97, 24.93, 14.70, 14.70, 18.83],
        },
        "Messidor-2": {
            "accuracy": [64.20, 63.45, 63.95, 64.07, 63.28, 58.31, 63.97, 13.46],
            "ace":      [4.69, 4.10, 6.24, 3.66, 20.00, 29.68, 29.68, 14.76],
        },
    },
    "CLIP-DR-ResNet-101": {
        "APTOS": {
            "accuracy": [75.75, 75.12, 75.02, 75.78, 74.81, 57.18, 75.75, 8.03],
            "ace":      [3.95, 4.11, 7.32, 3.77, 38.90, 25.64, 5.77, 33.46],
        },
        "EyePACS": {
            "accuracy": [73.92, 73.93, 73.88, 73.95, 74.14, 73.47, 73.93, 2.59],
            "ace":      [3.10, 3.14, 5.15, 3.06, 9.16, 7.86, 7.86, 32.94],
        },
        "Messidor": {
            "accuracy": [59.47, 57.67, 59.31, 59.50, 58.21, 45.50, 59.47, 20.67],
            "ace":      [10.95, 9.03, 10.85, 10.92, 29.36, 16.99, 7.58, 11.31],
        },
        "Messidor-2": {
            "accuracy": [64.03, 63.68, 64.45, 64.11, 63.59, 58.31, 64.01, 2.01],
            "ace":      [6.03, 6.68, 9.01, 7.25, 24.95, 26.53, 7.76, 37.70],
        },
    },
    "CLIP-DR-ViT-B/16": {
        "APTOS": {
            "accuracy": [77.90, 77.45, 77.84, 77.94, 76.14, 56.78, 77.90, 7.95],
            "ace":      [3.99, 4.84, 7.90, 4.08, 38.96, 26.23, 4.74, 29.78],
        },
        "EyePACS": {
            "accuracy": [74.64, 74.65, 74.53, 74.64, 74.85, 72.71, 74.64, 3.21],
            "ace":      [2.30, 2.29, 4.29, 2.33, 18.80, 41.84, 7.47, 24.38],
        },
        "Messidor": {
            "accuracy": [59.50, 57.17, 59.61, 59.53, 58.31, 43.00, 59.50, 21.00],
            "ace":      [5.69, 6.55, 9.18, 5.81, 26.87, 14.91, 7.08, 14.72],
        },
        "Messidor-2": {
            "accuracy": [64.66, 64.51, 64.96, 64.62, 64.95, 58.26, 64.66, 2.18],
            "ace":      [6.24, 6.33, 8.37, 6.26, 21.17, 31.94, 7.46, 28.41],
        },
    },
    "CLIP-DR-ViT-B/32": {
        "APTOS": {
            "accuracy": [78.28, 77.98, 78.53, 78.26, 75.34, 66.72, 78.59, 7.86],
            "ace":      [4.00, 4.89, 8.93, 4.10, 39.31, 33.31, 3.54, 31.08],
        },
        "EyePACS": {
            "accuracy": [74.58, 74.49, 74.52, 74.55, 74.40, 73.34, 74.57, 8.86],
            "ace":      [3.24, 2.97, 6.22, 3.04, 13.08, 38.52, 7.20, 24.91],
        },
        "Messidor": {
            "accuracy": [55.47, 53.22, 55.45, 55.39, 56.17, 44.70, 55.69, 20.58],
            "ace":      [5.15, 4.56, 6.21, 5.04, 24.93, 14.70, 7.11, 21.65],
        },
        "Messidor-2": {
            "accuracy": [64.20, 63.90, 63.51, 64.10, 63.28, 58.31, 63.97, 7.63],
            # Corrected zero‑shot ACE value for the Messidor‑2 dataset.
            # Table A4 reports 30.38 for the zeroshot column (the eighth entry).
            "ace":      [4.99, 6.09, 8.03, 4.90, 20.00, 29.68, 6.05, 30.38],
        },
    },
    "MedCLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [62.27, 62.14, 63.16, 63.96, 62.27, 49.43, 62.27, 11.25],
            "ace":      [10.04, 10.00, 14.13, 10.68, 9.98, 22.19, 6.77, 8.82],
        },
        "EyePACS": {
            "accuracy": [73.66, 73.66, 73.66, 73.66, 73.65, 73.66, 73.66, 3.54],
            "ace":      [3.54, 3.47, 7.64, 3.66, 3.67, 20.87, 7.84, 16.00],
        },
        "Messidor": {
            "accuracy": [45.03, 45.05, 46.14, 45.39, 45.17, 44.33, 45.03, 14.33],
            "ace":      [9.41, 10.01, 7.10, 5.28, 3.37, 19.09, 13.72, 10.72],
        },
        "Messidor-2": {
            "accuracy": [58.31, 58.31, 58.31, 58.31, 58.31, 57.78, 58.31, 8.94],
            "ace":      [5.62, 7.60, 11.31, 9.81, 6.98, 23.00, 12.41, 11.13],
        },
    },
    "MedCLIP-X-ray": {
        "COVIDX": {
            "accuracy": [79.45, 79.46, 79.56, 79.69, 79.53, 79.02, 79.40, 78.77],
            "ace":      [29.34, 29.34, 29.45, 29.58, 29.41, 28.91, 4.98, 28.67],
        },
        "RSNA18": {
            "accuracy": [52.08, 52.11, 52.43, 52.60, 52.40, 52.21, 52.02, 47.60],
            "ace":      [18.60, 18.63, 18.95, 19.11, 18.92, 18.76, 1.38, 16.94],
        },
    },
    "BioMedCLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [77.25, 77.02, 76.88, 76.92, 62.27, 49.43, 77.25, 54.59],
            "ace":      [2.65, 2.51, 5.60, 2.60, 9.98, 22.19, 2.45, 18.98],
        },
        "EyePACS": {
            "accuracy": [74.13, 74.13, 74.07, 74.13, 73.65, 73.66, 74.13, 70.51],
            "ace":      [1.83, 1.85, 3.62, 1.83, 3.67, 20.87, 7.73, 16.16],
        },
        "Messidor": {
            "accuracy": [55.53, 54.97, 55.31, 55.42, 45.17, 44.33, 55.53, 45.67],
            "ace":      [5.36, 5.12, 6.73, 5.38, 3.37, 19.09, 8.84, 9.22],
        },
        "Messidor-2": {
            "accuracy": [62.96, 63.13, 63.19, 63.21, 58.31, 57.78, 62.96, 57.57],
            "ace":      [6.46, 6.59, 9.09, 6.29, 6.98, 23.00, 6.77, 6.94],
        },
    },
    "BioMedCLIP-X-ray": {
        "COVIDX": {
            "accuracy": [83.32, 78.27, 82.83, 83.39, 82.28, 81.46, 83.36, 84.37],
            "ace":      [6.97, 8.93, 11.69, 8.64, 6.72, 9.03, 7.09, 10.07],
        },
        "RSNA18": {
            "accuracy": [63.80, 63.97, 63.36, 63.58, 63.61, 61.67, 63.81, 49.71],
            "ace":      [8.69, 7.93, 4.93, 7.17, 8.14, 7.01, 9.04, 29.49],
        },
    },
    "PLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [78.66, 78.76, 78.03, 78.67, 77.52, 74.79, 78.67, 78.77],
            "ace":      [4.98, 5.24, 12.19, 5.05, 3.52, 21.00, 4.98, 10.71],
        },
        "EyePACS": {
            "accuracy": [74.43, 74.44, 74.28, 74.48, 74.33, 74.02, 74.43, 57.41],
            "ace":      [1.38, 1.15, 7.35, 1.43, 1.32, 19.35, 1.38, 29.22],
        },
        "Messidor": {
            "accuracy": [55.64, 55.83, 55.47, 55.58, 56.03, 54.36, 55.64, 45.58],
            "ace":      [5.11, 5.41, 9.03, 5.45, 5.92, 14.86, 5.11, 4.39],
        },
        "Messidor-2": {
            "accuracy": [63.46, 63.42, 63.74, 63.64, 63.28, 62.27, 63.46, 47.25],
            "ace":      [2.91, 3.95, 9.03, 4.78, 3.66, 18.01, 2.91, 14.69],
        },
    },
    "PLIP-Histo.": {
        "DigestPath": {
            "accuracy": [90.74, 91.12, 89.30, 91.09, 91.47, 83.32, 92.33, 80.69],
            "ace":      [4.98, 5.24, 12.19, 5.05, 3.52, 21.00, 4.98, 6.23],
        },
        "Kather": {
            "accuracy": [87.00, 86.51, 86.71, 85.90, 87.33, 87.69, 87.21, 57.74],
            "ace":      [1.38, 1.15, 7.35, 1.43, 1.32, 19.35, 1.38, 16.38],
        },
        "PaNuke": {
            "accuracy": [78.57, 77.84, 78.21, 78.35, 80.00, 67.86, 79.43, 56.74],
            "ace":      [5.11, 5.41, 9.03, 5.45, 5.92, 14.86, 5.11, 19.64],
        },
    },
    "QuiltNet-DR-B/32": {
        "APTOS": {
            "accuracy": [76.97, 76.12, 77.00, 76.72, 77.52, 73.04, 76.97, 10.03],
            "ace":      [4.98, 5.24, 12.19, 5.05, 3.52, 21.00, 4.98, 4.98],
        },
        "EyePACS": {
            "accuracy": [74.43, 74.40, 74.38, 74.33, 74.33, 73.90, 74.43, 22.22],
            "ace":      [1.38, 1.15, 7.35, 1.43, 1.32, 19.35, 1.38, 1.38],
        },
        "Messidor": {
            "accuracy": [52.31, 53.00, 53.36, 51.59, 56.03, 51.20, 52.31, 12.58],
            "ace":      [5.11, 5.41, 9.03, 5.45, 5.92, 14.86, 5.11, 5.11],
        },
        "Messidor-2": {
            "accuracy": [62.44, 62.44, 62.90, 63.23, 63.28, 61.75, 62.44, 24.03],
            "ace":      [2.91, 3.95, 9.03, 4.78, 3.66, 18.01, 2.91, 2.91],
        },
    },
    "QuiltNet-Histo.": {
        "DigestPath": {
            "accuracy": [89.76, 90.07, 89.68, 89.80, 90.40, 83.71, 89.22, 53.43],
            "ace":      [4.98, 5.24, 12.19, 5.05, 3.52, 21.00, 4.98, 20.23],
        },
        "Kather": {
            "accuracy": [91.56, 90.75, 90.61, 90.49, 93.23, 91.40, 91.84, 60.32],
            "ace":      [1.38, 1.15, 7.35, 1.43, 1.32, 19.35, 1.38, 3.84],
        },
        "PaNuke": {
            "accuracy": [78.86, 78.46, 79.19, 78.98, 79.99, 72.62, 78.11, 55.43],
            "ace":      [5.11, 5.41, 9.03, 5.45, 5.92, 14.86, 5.11, 24.15],
        },
    },
}


def compute_correlations(metric_data: dict) -> list:
    """Compute Pearson correlation coefficients for a single backbone.

    For each calibration method, this function computes the Pearson
    correlation between the accuracy and ACE values across all datasets
    associated with the given backbone.  If only a single dataset is
    available, NumPy will return nan; we replace nan with 1.0 to
    indicate undefined correlation (consistent with the original paper's
    treatment of such cases).

    Parameters
    ----------
    metric_data : dict
        A mapping from dataset name to a dictionary with keys
        "accuracy" and "ace" (list of floats).

    Returns
    -------
    list
        A list of correlation coefficients, one per calibration method.
    """
    num_methods = len(methods)
    correlations = []
    for i in range(num_methods):
        acc_vals = []
        ace_vals = []
        for data in metric_data.values():
            # ensure lists have the correct length
            if i < len(data["accuracy"]):
                acc_vals.append(data["accuracy"][i])
                ace_vals.append(data["ace"][i])
        if len(acc_vals) > 1:
            corr = np.corrcoef(acc_vals, ace_vals)[0, 1]
        else:
            corr = np.nan
        if np.isnan(corr):
            corr = 1.0
        correlations.append(corr)
    return correlations


def build_correlation_matrix(backbones_dict: dict) -> pd.DataFrame:
    """Construct a DataFrame of correlations for all backbones and methods."""
    data = {}
    for backbone_name, datasets in backbones_dict.items():
        data[backbone_name] = compute_correlations(datasets)
    df = pd.DataFrame.from_dict(data, orient="index", columns=methods)
    return df


def plot_heatmap(df: pd.DataFrame, filename: str) -> None:
    """Plot and save a heatmap using the same figure size and colour style as plot(1).py."""
    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(df.values, vmin=-1, vmax=1, aspect='auto', cmap='viridis')

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, ha='center')
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Calibration method", labelpad=10)
    ax.set_ylabel("Backbone", labelpad=10)

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=-1, vmax=1)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iat[i, j]
            if pd.notna(val):
                rgba = cmap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = 'black' if luminance > 0.6 else 'white'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pearson correlation (r)')

    ax.set_title('Calibration vs. Accuracy - Pearson r(Accuracy, ACE)', pad=12)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    df = build_correlation_matrix(backbones)
    print(df)
    plot_heatmap(df, "./medical_bm_fig/pearson_heatmap_ace.png")


if __name__ == "__main__":
    main()