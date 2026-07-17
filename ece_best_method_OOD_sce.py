import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== 1) PASTE OF YOUR RAW DATA (UNCHANGED, ODDITIES INCLUDED) ==============
raw_data = {
    "CLIP-ViT-B/16": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [5.29, 6.02, 9.15, 5.31, 26.87, 14.80, 6.95],
        "APTOS":      [41.70, 22.61, 30.89, 41.80, 8.17, 23.21, 29.15],
        "EYEPACS":    [32.71, 15.60, 14.94, 32.77, 25.69, 43.89, 22.74],
        "Messidor_2": [26.65, 15.61, 19.79, 26.59, 13.53, 32.77, 19.18],
    },

    "CLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [3.11, 3.82, 5.01, 3.07, 39.31, 14.65, 6.42],
        "APTOS":      [23.27, 22.18, 16.39, 23.27, 17.19, 17.13, 20.60],
        "EYEPACS":    [11.67, 15.79, 8.15, 11.67, 11.95, 20.01, 9.98],
        "Messidor_2": [12.55, 7.75, 14.19, 12.55, 14.46, 21.29, 15.24],
    },

    "CLIP-ResNet-50": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [4.88, 4.43, 5.06, 4.48, 39.31, 14.65, 6.42],
        "APTOS":      [24.13, 15.37, 19.67, 24.25, 12.22, 17.57, 20.38],
        "EYEPACS":    [10.94, 13.97, 10.83, 10.95, 25.52, 31.43, 10.07],
        "Messidor_2": [10.15, 10.18, 8.41, 9.98, 18.57, 23.22, 9.52],
    },

    "CLIP-ResNet-101": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [10.67, 9.09, 10.90, 10.67, 29.36, 16.91, 6.87],
        "APTOS":      [21.38, 27.42, 17.84, 21.24, 26.89, 15.56, 32.71],
        "EYEPACS":    [16.53, 23.59, 15.13, 16.60, 43.77, 11.73, 26.14],
        "Messidor_2": [13.07, 17.10, 12.52, 13.32, 33.72, 16.72, 22.07],
    },

    "QuiltNet-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [4.00, 5.76, 4.66, 4.36, 5.69, 13.31, 6.20],
        "APTOS":      [19.28, 10.25, 9.73, 12.67, 10.22, 11.88, 8.67],
        "EYEPACS":    [8.80, 11.51, 9.27, 11.77, 8.74, 18.87, 10.80],
        "Messidor_2": [18.18, 7.85, 11.12, 8.99, 7.09, 19.38, 10.14],
    },

    "PLIP-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [4.18, 4.26, 8.99, 5.13, 5.69, 15.00, 6.36],
        "APTOS":      [17.74, 17.01, 13.11, 23.03, 14.10, 9.86, 16.15],
        "EYEPACS":    [10.96, 9.69, 16.85, 13.50, 8.88, 25.38, 13.09],
        "Messidor_2": [9.65, 9.97, 12.94, 13.07, 9.66, 21.18, 11.10],
    },
    
    "Med-VLM-ViT-B/32": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [9.41, 9.19, 6.15, 5.04, 3.24, 19.08, 12.54],
        "APTOS":      [18.92, 8.21, 15.99, 14.13, 14.17, 16.08, 18.06],
        "EYEPACS":    [30.47, 27.10, 39.25, 38.3, 40.88, 18.75, 41.64],
        "Messidor_2": [22.44, 15.69, 24.59, 23.73, 23.63, 14.57, 26.46],
    },

    "Biomed-VLM-ViT-B/32": { #update it
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [5.12, 4.24, 5.98, 4.61, 3.24, 19.08, 9.04],
        "APTOS":      [23.70, 26.15, 20.65, 21.85, 30.02, 15.19, 25.28],
        "EYEPACS":    [8.95, 9.34, 12.53, 9.72, 15.68, 13.41, 9.08],
        "Messidor_2": [12.52, 11.94, 14.44, 13.01, 17.37, 11.50, 12.73],
    },

    "QuiltNet-ViT-B/32-Histopathology": {  # note trailing space in the key (kept)
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [11.86, 10.44, 7.98, 11.04, 10.42, 8.94, 11.74],
        "Digetspath": [41.3, 46.37, 42.33, 51.88, 44.47, 46.04, 60.38],
        "Kather":     [15.01, 9.54, 15.58, 10.37, 13.40, 7.46, 12.93],
    },

    "PLIP-ViT-B/32--Histopathology": {     # note double hyphen (kept)
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":         [11.97, 9.75, 2.95, 10.18, 10.90, 4.82, 11.21],
        "Digetspath": [24.36, 39.39, 15.87, 35.58, 16.71, 33.58, 24.06],
        # 8 values given; we will clean to 7 by truncating with a warning.
        "Kather":     [12.79, 13.11, 5.08, 12.53, 9.77, 9.37, 37.2, 7.23],
    },

    "Med-VLM-ViT-B/32-X-RaY": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":    [18.60, 18.63, 18.95, 19.11, 18.92, 18.73, 18.56],
        "Covid": [29.03, 29.03, 28.96, 28.94, 28.91, 28.38, 29.30],
    },

    "Biomed-VLM-ViT-B/32-X-RaY": {
        "methods": ["Base", "MDCA", "LS", "MBLS", "ECCV_ZS", "ECCV_Penalty", "Temperature"],
        "ID":    [8.59, 7.31, 4.78, 7.38, 8.15, 6.72, 8.56],
        "Covid": [8.99, 9.79, 6.82, 7.31, 10.38, 3.83, 8.69],
    },
}

# ============== 2) HELPERS: CLEANING, COERCION, TIDY BUILD ==============
def coerce_floats(values, expected_len, where=""):
    """
    - Strip '*' and other non-numeric artifacts.
    - Convert to float.
    - If too many values: truncate with warning.
    - If too few values: pad with NaN with warning.
    """
    cleaned = []
    for v in values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cleaned.append(float(v))
            continue
        # extract first numeric token (handles '6.36*', '11.5', etc.)
        s = str(v)
        m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                cleaned.append(float(m[0]))
            except Exception:
                cleaned.append(np.nan)
        else:
            cleaned.append(np.nan)

    if len(cleaned) > expected_len:
        print(f"[WARN] {where}: list longer than methods ({len(cleaned)}>{expected_len}); truncating.")
        cleaned = cleaned[:expected_len]
    elif len(cleaned) < expected_len:
        print(f"[WARN] {where}: list shorter than methods ({len(cleaned)}<{expected_len}); padding with NaN.")
        cleaned = cleaned + [np.nan] * (expected_len - len(cleaned))
    return cleaned

def tidy_from_raw(raw):
    """
    Returns a tidy DataFrame with columns:
      backbone, method, split ('ID' or 'OOD'), dataset, ECE
    """
    rows = []
    for backbone, block in raw.items():
        methods = block.get("methods", [])
        if not methods:
            print(f"[WARN] {backbone}: no 'methods' list found; skipping.")
            continue

        # ID vector
        if "ID" not in block:
            print(f"[WARN] {backbone}: no 'ID' list; skipping backbone.")
            continue
        id_vec = coerce_floats(block["ID"], len(methods), where=f"{backbone}::ID")

        # OOD datasets = all keys except 'methods' and 'ID'
        ood_keys = [k for k in block.keys() if k not in ("methods", "ID")]
        for idx, method in enumerate(methods):
            # add ID row
            rows.append(dict(backbone=backbone, method=method, split="ID", dataset="ID", ECE=id_vec[idx]))

        for ds in ood_keys:
            ood_vec = coerce_floats(block[ds], len(methods), where=f"{backbone}::{ds}")
            for idx, method in enumerate(methods):
                rows.append(dict(backbone=backbone, method=method, split="OOD", dataset=ds, ECE=ood_vec[idx]))

    df = pd.DataFrame(rows)
    # Drop rows with NaN ECE, if any (you can keep them if you prefer)
    return df

# ============== 3) BUILD TIDY DF AND COMPUTE ΔECE SUMMARIES ==============
df = tidy_from_raw(raw_data)

# Pair ID & OOD within each backbone+method per OOD dataset
id_tbl  = df[df["split"]=="ID"][["backbone","method","ECE"]].rename(columns={"ECE":"ECE_ID"})
ood_tbl = df[df["split"]=="OOD"][["backbone","method","dataset","ECE"]].rename(columns={"ECE":"ECE_OOD"})

pairs = pd.merge(ood_tbl, id_tbl, on=["backbone","method"], how="inner")
pairs["dECE"] = pairs["ECE_OOD"] - pairs["ECE_ID"]  # OOD − ID (percentage points)

# Aggregate across OOD datasets => one ΔECE per (backbone, method)
per_backbone = (
    pairs
    .groupby(["method","backbone"], as_index=False)
    .agg(dECE_agg=("dECE","median"))  # robust across multiple OOD datasets
)

# Per-method summary across backbones
def q10(x): return float(np.nanquantile(x, 0.10)) if len(x)>0 else np.nan
def q25(x): return float(np.nanquantile(x, 0.25)) if len(x)>0 else np.nan
def q75(x): return float(np.nanquantile(x, 0.75)) if len(x)>0 else np.nan
def q90(x): return float(np.nanquantile(x, 0.90)) if len(x)>0 else np.nan

summary = (
    per_backbone
    .groupby("method")
    .agg(
        med_dECE = ("dECE_agg", "median"),
        q10_dECE = ("dECE_agg", q10),
        q25_dECE = ("dECE_agg", q25),
        q75_dECE = ("dECE_agg", q75),
        q90_dECE = ("dECE_agg", q90),
        N_backs  = ("dECE_agg", "size"),
        improve_rate = ("dECE_agg", lambda x: float(np.mean(np.array(x) < 0.0))),
    )
    .reset_index()
)

summary["IQR"] = summary["q75_dECE"] - summary["q25_dECE"]
summary = summary.sort_values("med_dECE", ascending=True).reset_index(drop=True)

# Save summary for your appendix/table
summary.to_csv("calib_method_summary.csv", index=False)

# ============== 4) SINGLE FIGURE: METHOD FOREST PLOT (ΔECE SUMMARY) ==============
fig = plt.figure(figsize=(9, 5.5))
ax = fig.add_axes([0.12, 0.14, 0.72, 0.78])

y_positions = np.arange(len(summary))[::-1]  # top-to-bottom
ax.axvline(0.0, linestyle="--", linewidth=1)

# 10–90% whiskers (thin)
for yi, (p10, p90) in enumerate(zip(summary["q10_dECE"], summary["q90_dECE"])):
    y = y_positions[yi]
    if not (math.isnan(p10) or math.isnan(p90)):
        ax.hlines(y, p10, p90, linewidth=1)

# IQR bars (thick)
for yi, (q25, q75) in enumerate(zip(summary["q25_dECE"], summary["q75_dECE"])):
    y = y_positions[yi]
    if not (math.isnan(q25) or math.isnan(q75)):
        ax.hlines(y, q25, q75, linewidth=6)

# Median points (dots)
ax.plot(summary["med_dECE"], y_positions, marker="o", linestyle="None", markersize=6)

# Labels
ax.set_yticks(y_positions)
ax.set_yticklabels(summary["method"])
ax.set_xlabel("ΔECE (OOD − ID)  [percentage points]")
ax.set_title("OOD Generalization by Calibration Method — ΔECE Summary (across backbones)")

# Right-side annotations: N and improvement rate
# To place text just inside the right edge, get current x-limits after plotting:
x_left, x_right = ax.get_xlim()
for yi, (n, rate) in enumerate(zip(summary["N_backs"], summary["improve_rate"])):
    y = y_positions[yi]
    ax.text(x_right, y, f"N={int(n)} | improve={int(rate*100)}%", va="center")

plt.tight_layout()
plt.savefig("./medical_bm_fig/method_forest_plot_from_given_data.png", dpi=300, bbox_inches="tight")
plt.show()

# ============== 5) OPTIONAL: PRINT A QUICK TEXT SUMMARY ==============
print("\n=== Per-method ΔECE summary (sorted) ===")
print(summary.to_string(index=False))
