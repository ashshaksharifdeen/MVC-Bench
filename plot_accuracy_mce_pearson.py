import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

"""
This script constructs a Pearson correlation heatmap between
classification accuracy and Maximum Calibration Error (MCE) for the
in‑distribution results presented in Table A4 of the MVC‑Bench
supplementary material.

The data structure and methodology mirror the ACE script.  Each
backbone contains per‑dataset lists of accuracy and MCE values for
eight calibration baselines (Base, MDCA, LS, MBLS, ECCV_ZS,
ECCV_Penalty, TS and zeroshot).  The “Ours (MCM)” column is
omitted.  Correlations are computed across datasets per backbone for
each calibration method, with single‑dataset cases returning 1.0.

Running this script produces a heatmap saved as `pearson_heatmap_mce.png`.
"""

# Calibration methods
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

# Per‑backbone, per‑dataset accuracy and MCE values from Table A4.
# Each list contains eight numbers corresponding to the calibration methods.
backbones = {
    "CLIP-DR-ResNet-50": {
        "APTOS": {
            "accuracy": [76.69, 76.19, 76.64, 76.82, 75.34, 66.72, 78.59, 8.62],
            "mce":      [1.49, 1.37, 2.37, 2.01, 30.20, 25.01, 25.01, 18.84],
        },
        "EyePACS": {
            "accuracy": [74.48, 74.06, 74.02, 74.12, 74.40, 73.34, 74.57, 7.39],
            "mce":      [1.66, 0.79, 1.99, 0.67, 3.06, 26.81, 26.81, 13.70],
        },
        "Messidor": {
            "accuracy": [56.12, 54.44, 57.19, 57.06, 75.34, 44.70, 55.69, 20.58],
            "mce":      [1.66, 1.68, 1.97, 1.53, 13.62, 13.96, 13.96, 9.80],
        },
        "Messidor-2": {
            "accuracy": [64.20, 63.45, 63.95, 64.07, 63.28, 58.31, 63.97, 13.46],
            "mce":      [1.13, 1.17, 1.71, 0.86, 6.73, 25.42, 25.42, 9.77],
        },
    },
    "CLIP-DR-ResNet-101": {
        "APTOS": {
            "accuracy": [75.75, 75.12, 75.02, 75.78, 74.81, 57.18, 75.75, 8.03],
            "mce":      [1.12, 1.12, 1.64, 1.11, 18.21, 14.78, 3.30, 19.20],
        },
        "EyePACS": {
            "accuracy": [73.92, 73.93, 73.88, 73.95, 74.14, 73.47, 73.93, 2.59],
            "mce":      [0.84, 0.93, 2.83, 0.87, 1.98, 2.85, 2.85, 22.17],
        },
        "Messidor": {
            "accuracy": [59.47, 57.67, 59.31, 59.50, 58.21, 45.50, 59.47, 20.67],
            "mce":      [4.10, 3.87, 4.14, 4.04, 19.00, 16.49, 2.02, 11.30],
        },
        "Messidor-2": {
            "accuracy": [64.03, 63.68, 64.45, 64.11, 63.59, 58.31, 64.01, 2.01],
            "mce":      [2.11, 1.87, 2.91, 2.37, 11.04, 21.58, 1.69, 18.16],
        },
    },
    "CLIP-DR-ViT-B/16": {
        "APTOS": {
            "accuracy": [77.90, 77.45, 77.84, 77.94, 76.14, 56.78, 77.90, 7.95],
            "mce":      [1.46, 1.46, 1.81, 1.38, 12.54, 13.95, 1.86, 17.20],
        },
        "EyePACS": {
            "accuracy": [74.64, 74.65, 74.53, 74.64, 74.85, 72.71, 74.64, 3.21],
            "mce":      [0.68, 0.78, 2.23, 0.72, 5.45, 21.84, 3.24, 19.22],
        },
        "Messidor": {
            "accuracy": [59.50, 57.17, 59.61, 59.53, 58.31, 43.00, 59.50, 21.00],
            "mce":      [1.78, 2.17, 2.53, 1.82, 17.43, 12.69, 2.91, 14.55],
        },
        "Messidor-2": {
            "accuracy": [64.66, 64.51, 64.96, 64.62, 64.95, 58.26, 64.66, 2.18],
            "mce":      [1.54, 2.05, 2.31, 1.55, 9.51, 31.87, 2.51, 14.18],
        },
    },
    "CLIP-DR-ViT-B/32": {
        "APTOS": {
            "accuracy": [78.28, 77.98, 78.53, 78.26, 75.34, 66.72, 78.59, 7.86],
            "mce":      [1.52, 1.74, 1.78, 1.51, 13.62, 25.01, 0.78, 17.28],
        },
        "EyePACS": {
            "accuracy": [74.58, 74.49, 74.52, 74.55, 74.40, 73.34, 74.57, 8.86],
            "mce":      [1.66, 1.55, 3.26, 1.52, 3.06, 26.81, 2.99, 18.52],
        },
        "Messidor": {
            "accuracy": [55.47, 53.22, 55.45, 55.39, 56.17, 44.70, 55.69, 20.58],
            "mce":      [0.91, 1.61, 1.62, 0.93, 13.88, 13.96, 2.32, 18.74],
        },
        "Messidor-2": {
            "accuracy": [64.20, 63.90, 63.51, 64.10, 63.28, 58.31, 63.97, 7.63],
            "mce":      [1.31, 1.53, 1.81, 1.39, 6.73, 25.42, 2.13, 17.94],
        },
    },
    "MedCLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [62.27, 62.14, 63.16, 63.96, 62.27, 49.43, 62.27, 11.25],
            "mce":      [3.27, 3.16, 4.46, 3.22, 3.16, 16.74, 4.29, 8.82],
        },
        "EyePACS": {
            "accuracy": [73.66, 73.66, 73.66, 73.66, 73.65, 73.66, 73.66, 3.54],
            "mce":      [1.36, 1.12, 4.10, 1.36, 1.85, 14.62, 4.74, 16.56],
        },
        "Messidor": {
            "accuracy": [45.03, 45.05, 46.14, 45.39, 45.17, 44.33, 45.03, 14.33],
            "mce":      [8.37, 7.54, 4.82, 2.27, 1.47, 19.09, 11.27, 10.72],
        },
        "Messidor-2": {
            "accuracy": [58.31, 58.31, 58.31, 58.31, 58.31, 57.78, 58.31, 8.94],
            "mce":      [2.61, 3.82, 6.41, 3.73, 3.17, 12.28, 7.77, 11.13],
        },
    },
    "MedCLIP-X-ray": {
        "COVIDX": {
            "accuracy": [79.45, 79.46, 79.56, 79.69, 79.53, 79.02, 79.40, 78.77],
            "mce":      [29.34, 29.34, 29.45, 29.58, 29.41, 28.91, 1.26, 28.67],
        },
        "RSNA18": {
            "accuracy": [52.08, 52.11, 52.43, 52.60, 52.40, 52.21, 52.02, 47.60],
            "mce":      [18.60, 18.63, 18.95, 19.11, 18.92, 18.73, 0.49, 14.05],
        },
    },
    "BioMedCLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [77.25, 77.02, 76.88, 76.92, 62.27, 49.43, 77.25, 54.59],
            "mce":      [0.80, 0.67, 1.33, 0.77, 3.16, 16.74, 0.59, 9.01],
        },
        "EyePACS": {
            "accuracy": [74.13, 74.13, 74.07, 74.13, 73.65, 73.66, 74.13, 70.51],
            "mce":      [0.56, 0.55, 2.61, 0.56, 1.85, 14.62, 2.81, 9.31],
        },
        "Messidor": {
            "accuracy": [55.53, 54.97, 55.31, 55.42, 45.17, 44.33, 55.53, 45.67],
            "mce":      [1.73, 1.38, 1.83, 1.57, 1.47, 19.09, 3.45, 3.43],
        },
        "Messidor-2": {
            "accuracy": [62.96, 63.13, 63.19, 63.21, 58.31, 57.78, 62.96, 57.57],
            "mce":      [2.01, 1.68, 2.62, 1.59, 3.17, 12.28, 2.24, 4.15],
        },
    },
    "BioMedCLIP-X-ray": {
        "COVIDX": {
            "accuracy": [83.32, 78.27, 82.83, 83.39, 82.28, 81.46, 83.36, 84.37],
            "mce":      [2.42, 3.85, 4.56, 3.34, 2.51, 3.60, 0.90, 8.88],
        },
        "RSNA18": {
            "accuracy": [63.80, 63.97, 63.36, 63.58, 63.61, 61.67, 63.81, 49.71],
            "mce":      [2.98, 2.56, 1.44, 2.38, 2.39, 2.33, 3.30, 16.67],
        },
    },
    "PLIP-DR-B/32": {
        "APTOS": {
            "accuracy": [78.66, 78.76, 78.03, 78.67, 77.52, 74.79, 78.67, 78.77],
            "mce":      [1.26, 1.24, 2.42, 1.30, 0.76, 4.24, 1.26, 6.85],
        },
        "EyePACS": {
            "accuracy": [74.43, 74.44, 74.28, 74.48, 74.33, 74.02, 74.43, 57.41],
            "mce":      [0.49, 0.39, 3.38, 0.52, 0.33, 7.69, 0.49, 21.04],
        },
        "Messidor": {
            "accuracy": [55.64, 55.83, 55.47, 55.58, 56.03, 54.36, 55.64, 45.58],
            "mce":      [1.52, 1.74, 2.39, 1.64, 1.67, 7.74, 1.52, 2.62],
        },
        "Messidor-2": {
            "accuracy": [63.46, 63.42, 63.74, 63.64, 63.28, 62.27, 63.46, 47.25],
            "mce":      [1.01, 1.35, 2.14, 1.22, 0.93, 6.07, 1.01, 8.61],
        },
    },
    "PLIP-Histo.": {
        "DigestPath": {
            "accuracy": [90.74, 91.12, 89.30, 91.09, 91.47, 83.32, 92.33, 80.69],
            "mce":      [1.26, 1.24, 2.42, 1.30, 0.76, 4.24, 1.26, 1.83],
        },
        "Kather": {
            "accuracy": [87.00, 86.51, 86.71, 85.90, 87.33, 87.69, 87.21, 57.74],
            "mce":      [0.49, 0.39, 3.38, 0.52, 0.33, 7.69, 0.49, 5.34],
        },
        "PaNuke": {
            "accuracy": [78.57, 77.84, 78.21, 78.35, 80.00, 67.86, 79.43, 56.74],
            "mce":      [1.52, 1.74, 2.39, 1.64, 1.67, 7.74, 1.52, 6.66],
        },
    },
    "QuiltNet-DR-B/32": {
        "APTOS": {
            "accuracy": [76.97, 76.12, 77.00, 76.72, 77.52, 73.04, 76.97, 10.03],
            "mce":      [1.26, 1.24, 2.42, 1.30, 0.76, 4.24, 1.26, 1.26],
        },
        "EyePACS": {
            "accuracy": [74.43, 74.40, 74.38, 74.33, 74.33, 73.90, 74.43, 22.22],
            "mce":      [0.49, 0.39, 3.38, 0.52, 0.33, 7.69, 0.49, 0.49],
        },
        "Messidor": {
            "accuracy": [52.31, 53.00, 53.36, 51.59, 56.03, 51.20, 52.31, 12.58],
            "mce":      [1.52, 1.74, 2.39, 1.64, 1.67, 7.74, 1.52, 1.52],
        },
        "Messidor-2": {
            "accuracy": [62.44, 62.44, 62.90, 63.23, 63.28, 61.75, 62.44, 24.03],
            "mce":      [1.01, 1.35, 2.14, 1.22, 0.93, 6.07, 1.01, 1.01],
        },
    },
    "QuiltNet-Histo.": {
        "DigestPath": {
            "accuracy": [89.76, 90.07, 89.68, 89.80, 90.40, 83.71, 89.22, 53.43],
            "mce":      [1.26, 1.24, 2.42, 1.30, 0.76, 4.24, 1.26, 7.31],
        },
        "Kather": {
            "accuracy": [91.56, 90.75, 90.61, 90.49, 93.23, 91.40, 91.84, 60.32],
            "mce":      [0.49, 0.39, 3.38, 0.52, 0.33, 7.69, 0.49, 1.20],
        },
        "PaNuke": {
            "accuracy": [78.86, 78.46, 79.19, 78.98, 79.99, 72.62, 78.11, 55.43],
            "mce":      [1.52, 1.74, 2.39, 1.64, 1.67, 7.74, 1.52, 8.12],
        },
    },
}


def compute_correlations(metric_data: dict) -> list:
    """Compute Pearson correlation coefficients for a single backbone."""
    num_methods = len(methods)
    correlations = []
    for i in range(num_methods):
        acc_vals = []
        mce_vals = []
        for data in metric_data.values():
            if i < len(data["accuracy"]):
                acc_vals.append(data["accuracy"][i])
                mce_vals.append(data["mce"][i])
        if len(acc_vals) > 1:
            corr = np.corrcoef(acc_vals, mce_vals)[0, 1]
        else:
            corr = np.nan
        if np.isnan(corr):
            corr = 1.0
        correlations.append(corr)
    return correlations


def build_correlation_matrix(backbones_dict: dict) -> pd.DataFrame:
    data = {}
    for name, datasets in backbones_dict.items():
        data[name] = compute_correlations(datasets)
    return pd.DataFrame.from_dict(data, orient="index", columns=methods)


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

    ax.set_title('Calibration vs. Accuracy - Pearson r(Accuracy, MCE)', pad=12)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    df = build_correlation_matrix(backbones)
    print(df)
    plot_heatmap(df, "./medical_bm_fig/pearson_heatmap_mce.png")


if __name__ == "__main__":
    main()