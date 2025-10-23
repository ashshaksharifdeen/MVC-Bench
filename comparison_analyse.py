#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as osp
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import sys
from typing import Tuple, Dict, Union, Optional, Sequence
import glob
import csv
# Ensure project root is importable
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# --- Dassl / trainer plumbing ---
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from train import extend_cfg
from matplotlib.ticker import MultipleLocator
# Register MaPLe trainer (supports both package and flat file)
try:
    import trainers.maple as _  # noqa: F401
except Exception:
    try:
        import maple  # noqa: F401
    except Exception:
        pass


# --------------------
# Utilities
# --------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

@torch.no_grad()
def top1_margins_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """m_i = z_{y_i} - max_{j != y_i} z_j"""
    N, _ = logits.shape
    y = labels.long()
    true = logits[torch.arange(N), y]
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[torch.arange(N), y] = True
    tmp = torch.where(mask, torch.tensor(-1e30, device=logits.device, dtype=logits.dtype), logits)
    runner = tmp.max(dim=1).values
    m = true - runner
    return m.cpu().numpy()


def _ecdf(arr: np.ndarray):
    """Return sorted values x and empirical CDF F(x)=P(X<=x)."""
    x = np.sort(arr)
    n = len(x)
    F = np.arange(1, n + 1, dtype=float) / float(n)
    return x, F


def plot_margin_ecdfs(
    margin_dict: Dict[str, np.ndarray],
    save_path: str,
    thresholds: Sequence[float] = (0, 1, 2, 3, 4, 5, 6, 8, 10),
    method_order: Sequence[str] = ("CE", "Mbls", "Margin"),  # or ("CE","Mbls","Margin")
    title: str = "Margin ECDFs (lower curve ⇒ fewer samples ≤ x)",
    figsize=(12, 8),
    dpi: int = 300,
    xtick_size: int = 18,
    ytick_size: int = 18,
    show_nonneg_only: bool = True,
    x_max: Optional[float] = None,
    x_tick_step: Optional[float] = 1.0,
    ecdf_style: str = "line",  # "line" (smooth look) or "step" (classic ECDF)
):
    """
    Nicely-styled ECDFs for CE/Mbls/rmargin.
    - Lines + markers like your reference figure.
    - y-limit expanded so curves aren't hidden at the top edge.
    - ECDF math unchanged; we only change how it's drawn.
    """
    # Style per method (feel free to tweak)
    style_map = {
    "CE":      dict(color="#1f77b4", linestyle='-', linewidth=3.0),
    "Mbls":    dict(color="#ff7f0e", linestyle='-', linewidth=3.0),
    "Margin":  dict(color="#2ca02c", linestyle='-', linewidth=3.0),  # alias
}

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # Which methods to plot (only those present)
    methods = [m for m in method_order if m in margin_dict]

    # Compute right bound if not provided
    if x_max is None:
        all_vals = np.concatenate([v for v in margin_dict.values()
                                   if isinstance(v, np.ndarray) and v.size > 0])
        right = float(np.percentile(all_vals, 99.5)) if all_vals.size else 1.0
        # keep some headroom
        right = max(right, 1.0)
    else:
        right = float(x_max)

    # Draw vertical guide lines (behind curves)
    for t in thresholds:
        if (not show_nonneg_only or t >= 0) and t <= right:
            ax.axvline(t, linestyle="-", linewidth=1.1, color="#888", alpha=0.25, zorder=1)

    # Plot each ECDF
    for name in methods:
        arr = margin_dict[name]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue

        x = np.sort(arr)
        F = np.arange(1, len(x) + 1, dtype=float) / float(len(x))

        st = style_map.get(name, dict(color=None, marker="o"))
        color = st.get("color", None)
        marker = st.get("marker", "o")

        # choose ~12 evenly spaced markers so it doesn't look crowded
        if len(x) > 0:
            mark_idx = np.linspace(0, len(x) - 1, num=min(12, len(x)), dtype=int)
        else:
            mark_idx = []

        if ecdf_style == "step":
            #ax.step(x, F, where="post",
            #        color=color, linewidth=2.5, zorder=3, label=name)
            ax.plot(
    x, F,
    color=style_map.get(name, {}).get("color", None),
    linestyle='-',          # <— solid
    linewidth=3.0,          # <— thicker, crisp
    alpha=1.0,
    zorder=3,
    label=name
)
            ax.plot(
    x, F,
    color=style_map.get(name, {}).get("color", None),
    linestyle='-',          # <— solid
    linewidth=3.0,          # <— thicker, crisp
    alpha=1.0,
    zorder=3,
    label=name
)
            #ax.plot(x[mark_idx], F[mark_idx], linestyle="-",
            #        marker=marker, ms=6, mfc="white", mec=color, mew=1.5, zorder=4)
        else:
            # smooth-looking line (monotone; still the exact ECDF points)
            ax.plot(
    x, F,
    color=style_map.get(name, {}).get("color", None),
    linestyle='-',          # <— solid
    linewidth=3.0,          # <— thicker, crisp
    alpha=1.0,
    zorder=3,
    label=name
)
            #ax.plot(x, F, color=color, linewidth=2.5, zorder=3, label=name)
            #ax.plot(x[mark_idx], F[mark_idx], linestyle="none",
            #        marker=marker, ms=6, mfc="white", mec=color, mew=1.5, zorder=4)

    # Axes formatting
    if show_nonneg_only:
        ax.set_xlim(0.0, right)
    else:
        # keep the left bound as is but enforce the computed right
        ax.set_xlim(ax.get_xlim()[0], right)

    # Lift the top so the lines don't hide under the frame
    ax.set_ylim(0.0, 1.02)   # <- extra headroom above 1.0
    ax.margins(y=0.01)

    ax.set_xlabel("Margin Range", fontsize=18)
    ax.set_ylabel("ECDF", fontsize=18)
    ax.set_title("Margin vs Empirical Cumulative Distribution Function(ECDF)", fontsize=20)

    if x_tick_step is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax.tick_params(axis="x", labelsize=xtick_size)
    ax.tick_params(axis="y", labelsize=ytick_size)

    ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"[✓] Saved ECDF plot -> {save_path}")

def save_ecdf_table(
    margin_dict: Dict[str, np.ndarray],
    thresholds: Sequence[float],
    csv_path: str,
    method_order: Sequence[str] = ("CE", "Mbls", "Margin"),
):
    """
    Writes a table with P(m ≤ t) for each method at each threshold t.
    """
    methods = [m for m in method_order if m in margin_dict]
    fieldnames = ["threshold"] + methods
    rows = []
    for t in thresholds:
        row = {"threshold": float(t)}
        for m in methods:
            arr = margin_dict[m]
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                row[m] = float("nan")
            else:
                row[m] = float((arr <= t).mean())
        rows.append(row)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[✓] Saved ECDF table -> {csv_path}")


def _format_bucket_labels(edges: np.ndarray):
    labels = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        if np.isneginf(a) and np.isposinf(b):
            labels.append("all")
        elif np.isneginf(a):
            labels.append(f"< {b:g}")
        elif np.isposinf(b):
            labels.append(f"≥ {a:g}")
        else:
            labels.append(f"{a:g}–{b:g}")
    return labels

def barplot_margin_buckets(
    margin_dict: Dict[str, np.ndarray],
    save_path: str,
    edges: np.ndarray = None,
    normalize: bool = False,
    title: str = "Counts per margin bucket",
    xtick_size: int = 18,
    ytick_size: int = 18,
    x_rotation: int = 30,
):
    """
    Grouped bar chart of counts (or percentages) per margin bucket for each method.

    edges: array of bin edges (monotonic). Default buckets are paper-friendly:
           (-inf,0], (0,1], (1,2], (2,3], (3,4], (4,5], (5,6], (6,8], (8,10], (10, inf)
    normalize: if True, bars show percentage instead of raw counts.
    """
    # Choose interpretable default bins for a paper
    if edges is None:
        edges = np.array([-np.inf, 0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf], dtype=float)

    # Compute counts (aligned bins for all methods)
    methods = list(margin_dict.keys())
    counts = []
    totals = []
    for m in methods:
        arr = margin_dict[m]
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            hist = np.zeros(len(edges) - 1, dtype=int)
            tot = 0
        else:
            hist, _ = np.histogram(arr, bins=edges)
            tot = int(arr.size)
        counts.append(hist)
        totals.append(tot)
    counts = np.array(counts)   # shape: [M, B]
    labels = _format_bucket_labels(edges)

    # Normalize to percentages if requested
    if normalize:
        perc = []
        for i, tot in enumerate(totals):
            if tot == 0:
                perc.append(np.zeros_like(counts[i], dtype=float))
            else:
                perc.append(counts[i] * 100.0 / float(tot))
        values = np.array(perc)
        ylab = "Percentage of samples (%)"
    else:
        values = counts
        ylab = "Number of samples"

    # --- Plot (grouped bars) ---
    M, B = values.shape
    x = np.arange(B, dtype=float)
    width = 0.8 / max(M, 1)  # keep total group width reasonable

    plt.figure(figsize=(10, 8), dpi=300)
    for i, name in enumerate(methods):
        plt.bar(x + (i - (M - 1) / 2.0) * width, values[i], width, label=name)

    plt.xlabel("Margin bucket", fontsize=18)
    plt.ylabel(ylab, fontsize=18)
    plt.title(title, fontsize=20)
    plt.xticks(x, labels, rotation=x_rotation, ha="right", fontsize=xtick_size)
    plt.yticks(fontsize=ytick_size)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Saved bucketed bar plot -> {save_path}")

    # --- Also dump a CSV table for the paper appendix ---
    csv_path = save_path.replace(".png", ".csv")
    fieldnames = ["bucket"] + methods + [f"{m}_total" for m in methods]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in range(B):
            row = {"bucket": labels[b]}
            for i, m in enumerate(methods):
                row[m] = float(values[i, b]) if normalize else int(values[i, b])
                row[f"{m}_total"] = totals[i]
            w.writerow(row)
    print(f"[✓] Saved bucketed table -> {csv_path}")

def overlay_histogram(margin_dict: Dict[str, np.ndarray], save_path: str, bins: int = 60, title: str = "Margin distributions (overlay)"):
    vals = [v for v in margin_dict.values() if isinstance(v, np.ndarray) and v.size > 0]
    if not vals:
        print("[!] No margins to plot for overlay histogram.")
        return
    all_m = np.concatenate(vals)
    edges = np.linspace(all_m.min(), all_m.max(), bins + 1)

    plt.figure(figsize=(10, 8), dpi=300)
    for name, m in margin_dict.items():
        if isinstance(m, np.ndarray) and m.size > 0:
            plt.hist(m, bins=edges, density=True, histtype="step", linewidth=2, label=name)
    plt.xlabel("Top-1 margin  (z_y − max z_others)",fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.title(title, fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)    
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Saved overlay histogram -> {save_path}")

def boxplot_margins(margin_dict: Dict[str, np.ndarray], save_path: str, title: str = "Margin distributions (boxplot)"):
    labels = list(margin_dict.keys())
    data = [margin_dict[k] if isinstance(margin_dict[k], np.ndarray) else np.array([]) for k in labels]
    plt.figure(figsize=(10, 8), dpi=300)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Top-1 margin",fontsize=18)
    plt.title(title,fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Saved boxplot -> {save_path}")

def write_summary_csv(margin_dict: Dict[str, np.ndarray], save_path: str):
    import csv
    rows = []
    for name, m in margin_dict.items():
        if not isinstance(m, np.ndarray) or m.size == 0:
            rows.append({"Method": name, "N": 0, "mean": np.nan, "std": np.nan,
                         "p25": np.nan, "p50": np.nan, "p75": np.nan, "min": np.nan, "max": np.nan})
        else:
            rows.append({"Method": name,
                         "N": int(m.size),
                         "mean": float(np.mean(m)),
                         "std": float(np.std(m)),
                         "p25": float(np.percentile(m, 25)),
                         "p50": float(np.percentile(m, 50)),
                         "p75": float(np.percentile(m, 75)),
                         "min": float(np.min(m)),
                         "max": float(np.max(m))})
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[✓] Saved CSV summary -> {save_path}")

def _get_loader_by_name(trainer, split: str):
    """
    Try to fetch a loader by split name.
    Known aliases by split:
      - test:  test_loader, _test_loader, loader_test, loaders['test']
      - val:   val_loader, loader_val, loaders['val']
      - train: train_loader, train_loader_x, loader_train, loaders['train']
    """
    aliases = {
        "test": ["test_loader", "_test_loader", "loader_test"],
        "val":  ["val_loader", "loader_val"],
        "train":["train_loader", "train_loader_x", "loader_train"],
    }
    # attributes
    for attr in aliases.get(split, []):
        ld = getattr(trainer, attr, None)
        if ld is not None:
            return ld
    # dict-like
    loaders = getattr(trainer, "loaders", None)
    if isinstance(loaders, dict) and split in loaders:
        return loaders[split]
    return None

def _pick_eval_loader(trainer, pref: str = "auto"):
    """
    Return (loader, chosen_split). If pref=='auto', try test -> val -> train.
    """
    order = ["test", "val", "train"] if pref == "auto" else [pref]
    for split in order:
        ld = _get_loader_by_name(trainer, split)
        if ld is None:
            continue
        # if has dataset length 0, skip
        try:
            if hasattr(ld, "dataset") and len(ld.dataset) == 0:
                print(f"[warn] {split} loader exists but has 0 samples; trying next...")
                continue
        except Exception:
            pass
        return ld, split
    raise RuntimeError("No non-empty eval loader found (tried: {}).".format(order))

@torch.no_grad()
def collect_logits_labels(trainer, eval_split: str = "auto") -> Tuple[torch.Tensor, torch.Tensor]:
    loader, chosen = _pick_eval_loader(trainer, eval_split)
    print(f"[info] using '{chosen}' split for evaluation")
    try:
        if hasattr(loader, "dataset"):
            print(f"[info] {chosen} dataset size: {len(loader.dataset)}")
    except Exception:
        pass

    device = getattr(trainer, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    all_logits, all_labels = [], []
    n_batches = 0

    for batch in loader:
        n_batches += 1
        if hasattr(trainer, "parse_batch_test") and chosen in ("test", "val"):
            images, labels = trainer.parse_batch_test(batch)
        elif hasattr(trainer, "parse_batch_train") and chosen == "train":
            images, labels = trainer.parse_batch_train(batch)
        else:
            images, labels = batch  # assume (img, label)
            images, labels = images.to(device), labels.to(device)

        if hasattr(trainer, "model_inference"):
            out = trainer.model_inference(images)
        else:
            out = trainer.model(images)

        if isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                probs = out.get("probs", None)
                if probs is None:
                    raise RuntimeError("Model returned dict without 'logits' or 'probs'.")
                logits = torch.log(probs + 1e-12)
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if n_batches == 0:
        print("[warn] loader yielded 0 batches")
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"[info] collected logits: {tuple(logits.shape)}, labels: {tuple(labels.shape)}")
    return logits, labels


# --------------------
# Checkpoint resolution & loading
# --------------------

def _find_checkpoint(base_model_dir: str, method: str, seed: int, load_epoch: Union[int, str]) -> Optional[str]:
    """
    Search for a checkpoint file in both layouts:
      A) .../<method>/seed{seed}/**/model.pth.tar-<epoch>
      B) .../seed{seed}/<method>/**/model.pth.tar-<epoch>
    Fallbacks: model_best.pth.tar, model.pth.tar
    """
    pat_epoch = f"model.pth.tar-{load_epoch}"
    patterns = [
        osp.join(base_model_dir, method, f"seed{seed}", "**", pat_epoch),
        osp.join(base_model_dir, f"seed{seed}", method, "**", pat_epoch),
        osp.join(base_model_dir, method, f"seed{seed}", "**", "model_best.pth.tar"),
        osp.join(base_model_dir, f"seed{seed}", method, "**", "model_best.pth.tar"),
        osp.join(base_model_dir, method, f"seed{seed}", "**", "model.pth.tar"),
        osp.join(base_model_dir, f"seed{seed}", method, "**", "model.pth.tar"),
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            hits = sorted(hits, key=lambda p: len(p))
            return hits[0]
    return None

def _load_checkpoint_into_model(trainer, ckpt_file: str):
    """
    Manual load: supports various checkpoint dict formats.
    """
    print(f"[info] loading checkpoint file: {ckpt_file}")
    device = getattr(trainer, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(ckpt_file, map_location=device)
    if isinstance(state, dict):
        if "state_dict" in state:
            sd = state["state_dict"]
        elif "model" in state:
            sd = state["model"]
        else:
            sd = state
    else:
        raise RuntimeError("Unsupported checkpoint format (not a dict).")

    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    missing, unexpected = trainer.model.load_state_dict(new_sd, strict=False)
    print(f"[info] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  - missing keys (first 10):", missing[:10])
    if unexpected:
        print("  - unexpected keys (first 10):", unexpected[:10])


def build_and_load_trainer(
    root: str,
    dataset_cfg: str,
    method_cfg: str,
    subsample: str,
    base_model_dir: str,
    method: str,
    seed: int,
    load_epoch: Union[int, str]
):
    """
    Build a trainer, then try to load weights. We attempt:
      1) standard Dassl load_model(seed_dir, epoch)   [both layouts]
      2) manual load from resolved checkpoint path if (1) fails
    """
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(dataset_cfg)
    cfg.merge_from_file(method_cfg)

    cfg.defrost()
    cfg.DATASET.ROOT = root
    cfg.DATASET.SUBSAMPLE_CLASSES = subsample  # base/new/all
    cfg.TRAINER.NAME = "MaPLe"
    cfg.SEED = seed
    cfg.freeze()

    trainer = build_trainer(cfg)
    seed_dir_A = osp.join(base_model_dir, method, f"seed{seed}")
    seed_dir_B = osp.join(base_model_dir, f"seed{seed}")

    loaded = False
    for sd in [seed_dir_A, seed_dir_B]:
        if osp.isdir(sd):
            try:
                trainer.load_model(sd, epoch=load_epoch)
                print(f"[info] loaded via trainer.load_model: {sd} (epoch={load_epoch})")
                loaded = True
                break
            except Exception as e:
                print(f"[warn] load_model failed for {sd}: {e}")

    if not loaded:
        ckpt = _find_checkpoint(base_model_dir, method, seed, load_epoch)
        if ckpt is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for method={method}, seed={seed}, epoch={load_epoch}\n"
                f"Base dir: {base_model_dir}"
            )
        _load_checkpoint_into_model(trainer, ckpt)

    trainer.model.eval()
    return trainer


# --------------------
# Main
# --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--config-file", type=str, required=True)         # MaPLe method cfg yaml
    ap.add_argument("--dataset-config-file", type=str, required=True) # dataset yaml
    ap.add_argument("--base-model-dir", type=str, required=True)      # parent dir for CE/Mbls/rmargin OR seed{seed}/method
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--subsample-classes", type=str, default="base", choices=["base", "new", "all"])
    ap.add_argument("--eval-split", type=str, default="auto", choices=["auto","test","val","train"],
                    help="Which loader to use for evaluation. 'auto' tries test→val→train.")
    ap.add_argument("--methods", type=str, nargs="+", default=["CE", "Mbls", "rmargin"])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--load-epoch", default=5)  # int or "best"
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    out_dir = ensure_dir(args.output_dir)
    margin_dict: Dict[str, np.ndarray] = {}

    for method in args.methods:
        # We don't pre-validate layout; loader will try both + glob
        print(f"[•] Evaluating method: {method}")
        trainer = build_and_load_trainer(
            root=args.root,
            dataset_cfg=args.dataset_config_file,
            method_cfg=args.config_file,
            subsample=args.subsample_classes,
            base_model_dir=args.base_model_dir,
            method=method,
            seed=args.seed,
            load_epoch=args.load_epoch
        )
        logits, labels = collect_logits_labels(trainer, eval_split=args.eval_split)
        if logits.numel() == 0:
            print(f"[warn] no logits collected for {method} "
                  f"(subsample={args.subsample_classes}, eval-split={args.eval_split})")
            margin_dict[method] = np.array([])
        else:
            margins = top1_margins_from_logits(logits, labels)
            margin_dict[method] = margins

        del trainer, logits, labels
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    overlay_path = osp.join(out_dir, "margins_overlay_hist.png")
    overlay_histogram(
        margin_dict,
        overlay_path,
        bins=args.bins,
        title="PDF of Margin Distribution"#.format(osp.basename(args.base_model_dir), args.subsample_classes)
    )

    box_path = osp.join(out_dir, "margins_boxplot.png")
    boxplot_margins(
        margin_dict,
        box_path,
        title="Margin Distributions"#.format(args.subsample_classes)
    )

    csv_path = osp.join(out_dir, "margin_summary.csv")
    write_summary_csv(margin_dict, csv_path)

    # Bar plot of counts per margin bucket (raw counts)
    barplot_margin_buckets(
    margin_dict,
    save_path=os.path.join(out_dir, "margin_bucket_counts.png"),
    edges=np.array([-np.inf, 0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf]),
    normalize=False,
    title="Count of samples per margin bucket"
    )

    # Optional: percentage version (nice for comparing across datasets of different size)
    barplot_margin_buckets(
    margin_dict,
    save_path=os.path.join(out_dir, "margin_bucket_percent.png"),
    edges=np.array([-np.inf, 0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf]),
    normalize=True,
    title="Percentage of samples per margin bucket"
    )

    # Choose thresholds you want to highlight (match your example if you like)
    ecdf_thresholds = (0, 1, 2, 3, 4, 5, 6, 8, 10)

    # ECDF figure
    ecdf_png = osp.join(out_dir, "margins_ecdf.png")
    plot_margin_ecdfs(
    margin_dict,
    save_path=os.path.join(out_dir, "margins_ecdf_nonneg.png"),
    thresholds=(0,1,2,3,4,5,6,8,10),
    method_order=("CE","Mbls","Margin"),  # or ("CE","Mbls","rmargin")
    show_nonneg_only=True,   # <— hides negatives
    x_max=10,                # pick a nice right bound for your dataset
    x_tick_step=1.0
)

    # Table with P(m ≤ t) per method (great for the paper appendix)
    ecdf_csv = osp.join(out_dir, "margins_ecdf_table.csv")
    save_ecdf_table(margin_dict, thresholds=ecdf_thresholds, csv_path=ecdf_csv)

    readme = osp.join(out_dir, "README.txt")
    with open(readme, "w") as f:
        f.write(
            "Interpretation:\n"
            "- CE: wide margin spread (small & huge margins coexist).\n"
            "- MBLS: trims right tail (huge margins) but leaves fat left tail (small margins) -> underconfidence persists.\n"
            "- rmargin: fewer tiny margins + suppressed extreme right tail -> higher mean, smaller variance.\n"
        )
    print(f"[✓] Saved README -> {readme}")
    print("[✓] Done.")

if __name__ == "__main__":
    main()
