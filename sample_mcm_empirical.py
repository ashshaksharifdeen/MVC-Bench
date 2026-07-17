#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Margin Risk–Return Map from trained checkpoints.

y-axis: mean per-sample OvR margin μ  (higher is better)
x-axis: dispersion σ of per-sample OvR margin (lower is better)
bubble size: P(m̄ ≥ right_thr)  (spiky overconfidence tail)
bubble outline width: P(m̄ ≤ left_thr) (weak-separation mass)

Works with your CoOp setup and paths like:
  .../seed2/<METHOD>/prompt_learner/model.pth.tar-50
"""

import os, os.path as osp, re, argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from dassl.engine import build_trainer
from train import setup_cfg  # your repo's config builder

# --------------------------
# Helpers: cfg, loading, eval
# --------------------------
def normalize_dir_and_epoch(path_or_dir, default_epoch=None):
    """Accept a dir or a full '.../prompt_learner/model.pth.tar-XX' path."""
    p = path_or_dir.rstrip("/")
    m = re.search(r"model\.pth\.tar-(\d+)$", p)
    if m:
        epoch = int(m.group(1))
        directory = osp.dirname(osp.dirname(p))  # .../<METHOD>
        return directory, epoch
    if osp.basename(p) == "prompt_learner":
        return osp.dirname(p), default_epoch
    return p, default_epoch

def build_trainer_for_eval(root, dataset_cfg, method_cfg, trainer_name="CoOp", seed=-1, opts=None):
    class _Args: pass
    args = _Args()
    args.root = root
    args.output_dir = ""
    args.resume = ""
    args.seed = seed
    args.source_domains = None
    args.target_domains = None
    args.transforms = None
    args.trainer = trainer_name
    args.backbone = ""
    args.head = ""
    args.dataset_config_file = dataset_cfg
    args.config_file = method_cfg
    args.opts = opts or []
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    return trainer, cfg

def find_eval_loader(trainer, split="auto"):
    cands = {
        "auto": ["test_loader","val_loader","test_loader_x","val_loader_x","test_loader_u","val_loader_u"],
        "test": ["test_loader","test_loader_x","test_loader_u","val_loader"],
        "val":  ["val_loader","val_loader_x","val_loader_u","test_loader"],
        "train":["train_loader","train_loader_x"],
    }[split]
    for name in cands:
        if hasattr(trainer, name) and getattr(trainer, name) is not None:
            return getattr(trainer, name)
    raise RuntimeError("No eval loader found on trainer.")

@torch.no_grad()
def collect_logits_labels(trainer, loader):
    dev = trainer.device
    model = trainer.model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    logits_all, labels_all = [], []
    for batch in loader:
        if isinstance(batch, dict) and "img" in batch and "label" in batch:
            x = batch["img"].to(dev); y = batch["label"].to(dev)
        else:
            x, y = batch[0].to(dev), batch[1].to(dev)
        out = model(x)  # logits
        logits_all.append(out.detach().cpu())
        labels_all.append(y.detach().cpu())
    return torch.cat(logits_all, 0), torch.cat(labels_all, 0)

def per_sample_mean_ovr_margin(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    m_{i,k} = z[i,y] - z[i,k],  k!=y
    m̄_i     = mean_k m_{i,k}
    """
    B, C = logits.shape
    idx = torch.arange(B)
    z_true = logits[idx, labels].unsqueeze(1)             # (B,1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[idx, labels] = False
    others = logits[mask].view(B, C-1)                    # (B, C-1)
    margins = z_true - others                             # (B, C-1)
    mbar = margins.mean(dim=1)                            # (B,)
    return mbar.cpu().numpy()

def summarize_mbar(mbar: np.ndarray, left_thr=1.0, right_thr=8.0):
    mu = float(np.mean(mbar))
    sigma = float(np.std(mbar))
    p_left  = float((mbar <= left_thr).mean())
    p_right = float((mbar >= right_thr).mean())
    return mu, sigma, p_left, p_right

# --------------------------
# Plot: Risk–Return Map
# --------------------------
def plot_risk_return(rows, out_png, left_thr=1.0, right_thr=8.0):
    """
    rows: dict name -> (mu, sigma, p_left, p_right) with keys in desired display order.
    """
    labels = list(rows.keys())
    x = np.array([rows[k][1] for k in labels])  # sigma
    y = np.array([rows[k][0] for k in labels])  # mu
    p_left  = np.array([rows[k][2] for k in labels])
    p_right = np.array([rows[k][3] for k in labels])

    # visual encodings
    sizes = 400 + 60000 * p_right   # bubble area
    lws   = 1.0 + 12.0 * p_left     # outline thickness

    plt.figure(figsize=(9,7), dpi=300)
    ax = plt.gca()

    for xi, yi, si, lwi, lab in zip(x, y, sizes, lws, labels):
        ax.scatter(xi, yi, s=si, linewidths=lwi, edgecolors="black", alpha=0.95, label=lab)

    # arrow path CE -> MBLS -> MCM (if all exist)
    for chain in [["CE","Mean","Var","MCM"]]:
        if all(c in labels for c in chain):
            idx = [labels.index(k) for k in chain]
            for a, b in zip(idx[:-1], idx[1:]):
                ax.annotate("", xy=(x[b], y[b]), xytext=(x[a], y[a]),
                            arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_xlabel("Dispersion σ (lower is better)")
    ax.set_ylabel("Mean OvR margin μ (higher is better)")
    ax.set_title("Margin Risk–Return Map  (size=P(m̄≥{:.0f}), outline=P(m̄≤{:.0f}))".format(right_thr, left_thr))
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Legend at TOP-RIGHT (upper right), opaque
    ax.legend(loc="upper right", framealpha=1.0)

    # encoding reminder
    ax.text(0.02, 0.98, "Bubble size ∝ P(m̄ ≥ {:.0f})\nOutline width ∝ P(m̄ ≤ {:.0f})".format(right_thr, left_thr),
            transform=ax.transAxes, va="top", ha="left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config-file", required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--root", required=True)

    ap.add_argument("--ce", required=True, help="CE dir OR full .../prompt_learner/model.pth.tar-XX")
    ap.add_argument("--mean", required=True, help="MBLS dir OR full path")
    ap.add_argument("--var", required=True, help="MBLS dir OR full path")
    ap.add_argument("--mcm", required=True, help="MCM dir OR full path")

    ap.add_argument("--epoch", type=int, default=None, help="epoch if not encoded in filename")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--split", choices=["auto","val","test","train"], default="auto")
    ap.add_argument("--trainer-name", type=str, default="CoOp")
    ap.add_argument("--left-thr", type=float, default=1.0)
    ap.add_argument("--right-thr", type=float, default=8.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    methods = {
        "CE": args.ce,
        "Mean": args.mean,
        "Var": args.var,
        "MCM":args.mcm
    }

    rows = {}
    for name, p in methods.items():
        print(f"[•] {name}: {p}")
        directory, epoch = normalize_dir_and_epoch(p, default_epoch=args.epoch)
        if epoch is None:
            raise ValueError(f"Epoch not found in path '{p}'. Provide --epoch.")
        trainer, _ = build_trainer_for_eval(args.root, args.dataset_config_file, args.config_file,
                                            trainer_name=args.trainer_name, seed=args.seed)
        trainer.load_model(directory, epoch=epoch)
        loader = find_eval_loader(trainer, split=args.split)
        logits, labels = collect_logits_labels(trainer, loader)
        mbar = per_sample_mean_ovr_margin(logits, labels)
        rows[name] = summarize_mbar(mbar, left_thr=args.left_thr, right_thr=args.right_thr)

    # Plot & save
    out_png = osp.join(args.outdir, "margin_risk_return_map.png")
    plot_risk_return(rows, out_png, left_thr=args.left_thr, right_thr=args.right_thr)
    print(f"[✓] Figure saved: {out_png}")

    # CSV of metrics
    out_csv = osp.join(args.outdir, "margin_risk_return_map.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","mu","sigma","P(m<=%.0f)"%args.left_thr,"P(m>=%.0f)"%args.right_thr])
        for k in ["CE","Mean","Var","MCM"]:
            if k in rows:
                mu, sigma, pl, pr = rows[k]
                w.writerow([k, f"{mu:.6f}", f"{sigma:.6f}", f"{pl:.6f}", f"{pr:.6f}"])
    print(f"[✓] Table saved : {out_csv}")

if __name__ == "__main__":
    main()
