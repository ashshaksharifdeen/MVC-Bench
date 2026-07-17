# ==============================================
# Kendall τ heatmap for ECE ranks (Messidor -> OOD)
# - Edit the DATA dict below with your ECE (%) values.
# - One figure is saved to 'ood_tau_ece_clean.png'.
# - No SciPy required.
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) INPUT: Paste your numbers
# -----------------------------
# Put the same set of methods in the same order for each backbone.
# For each backbone:
#   - "methods": list of method names (Base, MDCA, LS, MBLS, TS, VectorScale, MAPLE, ...)
#   - "ID": ECE list on Messidor (source domain), one per method
#   - "APTOS": ECE list on APTOS (OOD), one per method
#   - "EYEPACS": ECE list on EYEPACS (OOD), one per method
#   - "Messidor_2": ECE list on Messidor_2 (OOD), one per method
#
# IMPORTANT:
# - All lists under a backbone must have the same length as "methods".
# - Values should be numeric (floats or ints). ECE is typically in percent.
"""
DATA = {
    "CLIP-ViT-B/16": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            59.50, 57.17, 59.61, 59.53, 58.31, 43.00, 59.5
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            28.69, 34.50, 26.82, 28.82, 31.3, 47.17, 28.69
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            40.95, 45.32, 45.77, 41.02, 53.78, 68.11, 40.95
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            29.07, 34.81, 29.80, 29.07, 30.47, 57.72, 29.07
        ],
    },

    "CLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            55.47, 53.22, 55.45, 55.39, 75.34, 44.7, 55.69
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            40.88, 44.18, 28.73, 40.88, 26.7, 25.76, 40.88
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            51.5, 58.45, 49.68, 51.5, 37.91, 22.99, 51.5
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            38.36, 38.28, 26.76, 38.36, 22.48, 27.93, 38.36
        ],
    },

    "CLIP-ResNet-50": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            56.12, 54.44, 57.19, 57.06, 75.34, 44.7, 55.69
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            43.75, 48.01, 50.54, 43.8, 20.71, 43.83, 47.29
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            66.33, 66.48, 69.84, 66.34, 44.33, 55.79, 65.45
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            46.79, 50.29, 52.96, 46.69, 18.9, 48.95, 46.62
        ],
    },

    "CLIP-ResNet-101": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            59.47, 57.67, 59.31, 59.5, 58.21, 45.5, 59.47
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            35.38, 49.9, 36.88, 35.45, 51.13, 18.08, 35.48
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            50.16, 69.27, 64.66, 50.25, 69.71, 18.47, 50.13
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            40.2, 57.72, 46.46, 40.27, 58.05, 9.12, 40.02
        ],
    },

    "QuiltNet-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            52.30, 53.00, 53.36, 51.58, 56.02, 51.19, 52.30
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            32.36, 45.50, 42.62, 39.79, 47.17, 48.21, 47.17
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            55.53, 56.39, 61.05, 66.04, 62.39, 58.74, 62.39
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            32.35, 40.25, 45.95, 41.93, 41.93, 42.79, 41.95
        ],
    },

    "PLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            55.63, 55.83, 55.47, 55.58, 56.02, 54.36, 55.63
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            39.01, 41.09, 41.26, 28.77, 39.45, 37.14, 39.45
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            70.34, 70.73, 66.95, 69.95, 69.88, 71.12, 69.88
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            56.84, 53.47, 52.31, 53.72, 53.48, 46.88, 53.48
        ],
    },
    
    "MedCLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            45.02, 45.05, 46.14, 45.39, 45.16, 44.33, 45.02
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            47.79, 40.03, 48.56, 49.10, 48.91, 34.97, 47.79
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            58.9, 53.76, 72.60, 73.19, 73.14, 30.89, 58.94
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            45.02, 45.05, 46.14, 45.39, 45.16, 44.33, 45.02
        ],
    },
    "BioMedCLIP-ViT-B/32": { #update it
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            55.53, 54.97, 6.15*, 5.04*, 3.24*, 19.08*, 6.36*
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            50.2, 51.76, 50.73, 50.58, 50.65, 52.80, 50.20
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            68.81, 69.49, 68.33, 68.24, 69.08, 70.98, 68.81
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            59.34, 59.46, 58.64, 58.73, 57.60, 59.5, 59.34
        ],
    },
}"""
#histopathology
"""DATA = {
    "QuiltNet-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            78.85, 78.46, 79.18, 78.98, 79.98, 72.62, 78.85
        ],
        "Digetspath":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            47.39, 39.66, 42.96, 36.96, 43.63, 31.01, 47.55
        ],
        "Kather":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            45.52, 50.58, 37.08, 47.95, 46.23, 28.12, 37.2
        ],
    },

    "PLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            78.56, 77.84, 78.20, 78.35, 79.99, 67.86, 80.03
        ],
        "Digetspath":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            65.14, 46.46, 61.52, 50.25, 74.22, 39.36, 65.57
        ],
        "Kather":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            40.51, 26.23, 27.38, 26.88, 56.48, 21.15, 51.07
        ],

    },
    
}"""
#X-RaY
"""DATA = {
    "MedCLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            52.08, 52.11, 52.43, 52.6, 52.4, 52.21, 52.08
        ],
        "Covid":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            79.15, 79.15, 79.07, 79.06, 79.02, 78.49, 78.24
        ],
    },

    "BioMedCLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            63.80, 63.97, 63.36, 63.58, 63.60, 61.67, 63.80
        ],
        "Covid":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            74.55, 73.39, 75.55, 77.44, 73.63, 77.15, 74.55
        ],

    },
    
}"""

# -----------------------------
# 2) SETTINGS: edit if you want
# -----------------------------
OOD_ORDER = ["APTOS", "EYEPACS", "Messidor_2"]      # columns
BACKBONE_ORDER = list(DATA.keys())                  # rows in dict order
SAVE_PATH = "/medical_bm_fig/ood__ece_fundus_clean.png"                 # output file

# -----------------------------
# 3) Kendall's tau-b (no SciPy)
# -----------------------------
def kendall_tau_b(x, y):
    """
    Tie-corrected Kendall’s tau-b without SciPy.
    Works well for small N (e.g., ~7 methods).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return np.nan
    # rank with average ties
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    c = d = t_x = t_y = 0
    for i in range(n-1):
        dx = rx[i+1:] - rx[i]
        dy = ry[i+1:] - ry[i]
        s = dx * dy
        c += np.sum(s > 0)
        d += np.sum(s < 0)
        t_x += np.sum(dx == 0)
        t_y += np.sum(dy == 0)
    denom = np.sqrt((c + d + t_x) * (c + d + t_y))
    return np.nan if denom == 0 else (c - d) / denom

# -----------------------------
# 4) Build τ matrix (rows x cols)
# -----------------------------
H = np.full((len(BACKBONE_ORDER), len(OOD_ORDER)), np.nan)

for i, b in enumerate(BACKBONE_ORDER):
    methods = DATA[b]["methods"]
    id_ece = np.array(DATA[b]["ID"], dtype=float)
    if len(id_ece) != len(methods):
        raise ValueError(f"[{b}] 'ID' length ({len(id_ece)}) != methods length ({len(methods)}).")
    for j, d in enumerate(OOD_ORDER):
        if d not in DATA[b]:
            # skip if OOD missing
            continue
        ood_ece = np.array(DATA[b][d], dtype=float)
        if len(ood_ece) != len(methods):
            raise ValueError(f"[{b} | {d}] OOD list length ({len(ood_ece)}) != methods length ({len(methods)}).")
        # (Optional) If some methods are missing, you could mask them here.
        tau = kendall_tau_b(id_ece, ood_ece)
        H[i, j] = tau

# -----------------------------
# 5) Plot & save
# -----------------------------
plt.figure(figsize=(9, 7))
im = plt.imshow(H, vmin=-1, vmax=1, aspect="auto")  # default colormap
plt.xticks(range(len(OOD_ORDER)), OOD_ORDER, rotation=45, ha="right")
plt.yticks(range(len(BACKBONE_ORDER)), BACKBONE_ORDER)

# In-cell numeric annotation ONLY (no labels, no 'N')
for i in range(len(BACKBONE_ORDER)):
    for j in range(len(OOD_ORDER)):
        if not np.isnan(H[i, j]):
            plt.text(j, i, f"{H[i, j]:.2f}", ha="center", va="center", fontsize=11)

plt.colorbar(im)  # no label text (clean)
plt.title("Kendall’s τ for ECE Ranks (ID Messidor → OOD)")  # edit/remove if you prefer
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=220)
print(f"Saved: {SAVE_PATH}")
# plt.show()  # uncomment if you want to display interactively
