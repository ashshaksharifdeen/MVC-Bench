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

"""DATA = {
    "CLIP-ViT-B/16": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            5.29, 6.02, 9.15, 5.31, 26.87, 14.80, 6.95
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            41.70, 22.61, 30.89, 41.80, 8.17, 23.21, 29.15
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            32.71, 15.60, 14.94, 32.77, 25.69, 43.89, 22.74
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            26.65, 15.61, 19.79, 26.59, 13.53, 32.77, 19.18
        ],
    },

    "CLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            3.11, 3.82, 5.01, 3.07, 39.31, 14.65, 6.42
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            23.27, 22.18, 16.39, 23.27, 17.19, 17.13, 20.60
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            11.67, 15.79, 8.15, 11.67, 11.95, 20.01, 9.98
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            12.55, 7.75, 14.19, 12.55, 14.46, 21.29, 15.24
        ],
    },

    "CLIP-ResNet-50": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            4.88, 4.43, 5.06, 4.48, 39.31, 14.65, 6.42
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            24.13, 15.37, 19.67, 24.25, 12.22, 17.57, 20.38
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            10.94, 13.97, 10.83, 10.95, 25.52, 31.43, 10.07
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            10.15, 10.18, 8.41, 9.98, 18.57, 23.22, 9.52
        ],
    },

    "CLIP-ResNet-101": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            10.67, 9.09, 10.90, 10.67, 29.36, 16.91, 6.87
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            21.38, 27.42, 17.84, 21.24, 26.89, 15.56, 32.71
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            16.53, 23.59, 15.13, 16.60, 43.77, 11.73, 26.14
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            13.07, 17.10, 12.52, 13.32, 33.72, 16.72, 22.07
        ],
    },

    "QuiltNet-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            4.00, 5.76, 4.66, 4.36, 5.69, 13.31, 6.20
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            19.28, 10.25, 9.73, 12.67, 10.22, 11.88, 8.67
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            8.80, 11.51, 9.27, 11.77, 8.74, 18.87, 10.80
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            18.18, 7.85, 11.12, 8.99, 7.09, 19.38, 10.14
        ],
    },

    "PLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            4.18, 4.26, 8.99, 5.13, 5.69, 15.00, 6.36
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            17.74, 17.01, 13.11, 23.03, 14.10, 9.86, 16.15
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            10.96, 9.69, 16.85, 13.50, 8.88, 25.38, 13.09
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            9.65, 9.97, 12.94, 13.07, 9.66, 21.18, 11.10
        ],
    },
    
    "Med-VLM-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            9.41, 9.19, 6.15, 5.04, 3.24, 19.08, 12.54
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            18.92, 8.21, 15.99, 14.13, 14.17, 16.08, 18.06
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            30.47, 27.10, 39.25, 38.3, 40.88, 18.75, 41.64
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            22.44, 15.69, 24.59, 23.73, 23.63, 14.57, 26.46
        ],
    },
    "Biomed-VLM-ViT-B/32": { #update it
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            5.12, 4.24, 5.98, 4.61, 3.24, 19.08, 9.04
        ],
        "APTOS":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            23.70, 26.15, 20.65, 21.85, 30.02, 15.19, 25.28
        ],
        "EYEPACS":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            8.95, 9.34, 12.53, 9.72, 15.68, 13.41, 9.08
        ],
        "Messidor_2": [  # <<< PASTE YOUR NUMBERS HERE >>>
            12.52, 11.94, 14.44, 13.01, 17.37, 11.50, 12.73
        ],
    },
}"""
#histopathology
"""DATA = {
    "QuiltNet-ViT-B/32-Histopathology ": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            11.86, 10.44, 7.98, 11.04, 10.42, 8.94, 5.31
        ],
        "Digetspath":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            41.3, 46.37, 42.33, 51.88, 44.47, 46.04, 60.38
        ],
        "Kather":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            15.01, 9.54, 15.58, 10.37, 13.40, 7.46, 12.93
        ],
    },

    "PLIP-ViT-B/32--Histopathology": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            11.97, 9.75, 2.95, 10.18, 10.90, 4.82, 11.21
        ],
        "Digetspath":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            24.36, 39.39, 15.87, 35.58, 16.71, 33.58, 24.06
        ],
        "Kather":    [  # <<< PASTE YOUR NUMBERS HERE >>>
            12.79, 13.11, 5.08, 12.53, 9.77, 9.37, 7.23
        ],

    },
    
}"""
#X-RaY
DATA = {
    "Med-VLM-ViT-B/32-X-RaY": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            18.60, 18.63, 18.95, 19.11, 18.92, 18.73, 18.56
        ],
        "Covid":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            29.03, 29.03, 28.96, 28.94, 28.91, 28.38, 29.06
        ],
    },

    "Biomed-VLM-ViT-B/32-X-RaY": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [  # <<< PASTE YOUR NUMBERS HERE >>>
            8.59, 7.31, 4.78, 7.38, 8.15, 6.72, 8.65
        ],
        "Covid":      [  # <<< PASTE YOUR NUMBERS HERE >>>
            8.99, 9.79, 6.82, 7.31, 10.38, 3.83, 8.69
        ],

    },
    
}

# -----------------------------
# 2) SETTINGS: edit if you want
# -----------------------------
OOD_ORDER = ["Covid"] #["Digetspath", "Kather"]    #["APTOS", "EYEPACS", "Messidor_2"]  # ["Digetspath", "Kather"]  ["Covid"] # columns
BACKBONE_ORDER = list(DATA.keys())                  # rows in dict order
SAVE_PATH = "./medical_bm_fig/ood__ece_x-ray_clean.png"                 # output file

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
plt.savefig(SAVE_PATH, dpi=300)
print(f"Saved: {SAVE_PATH}")
# plt.show()  # uncomment if you want to display interactively
