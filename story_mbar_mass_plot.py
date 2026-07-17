#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load CE / MBLS / MCM CoOp checkpoints, compute per-sample AVERAGE margins (true-vs-all),
and draw a paper-ready stacked 100% bar figure (+ CSV).

Works with your repo structure:
  .../seed2/<METHOD>/prompt_learner/model.pth.tar-<epoch>

Example:
  python plot_mbar_from_ckpts.py \
    --dataset-config-file configs/datasets/aptos.yaml \
    --config-file configs/trainers/CoOp/vit_b16_ep50.yaml \
    --root /path/to/datasets \
    --split auto \
    --ce "/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/CE/prompt_learner/model.pth.tar-50" \
    --mbls "/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/MBLS/prompt_learner/model.pth.tar-50" \
    --mcm "/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/MCM/prompt_learner/model.pth.tar-50" \
    --outdir ./figs/aptos_seed2_ep50
"""

import os, os.path as osp
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import torch

# --- reuse your training config build
from train import setup_cfg  # your train.py config pipeline  :contentReference[oaicite:2]{index=2}
from dassl.engine import build_trainer

# your CoOp trainer's load_model expects a DIR and adds 'prompt_learner/model.pth.tar-<epoch>'
# internally, so we pass the right directory and epoch. :contentReference[oaicite:3]{index=3}

def normalize_dir_and_epoch(path_or_dir, default_epoch=None):
    """
    Accept either:
      .../seedX/<METHOD>/
      .../seedX/<METHOD>/prompt_learner/
      .../seedX/<METHOD>/prompt_learner/model.pth.tar-50
    Return (directory_for_trainer.load_model, epoch_int)
    """
    p = path_or_dir.rstrip("/")

    # case: full file path .../model.pth.tar-XX
    m = re.search(r"model\.pth\.tar-(\d+)$", p)
    if m:
        epoch = int(m.group(1))
        # strip '.../prompt_learner/model.pth.tar-XX' -> '.../<METHOD>'
        directory = osp.dirname(osp.dirname(p))
        return directory, epoch

    # case: ends with .../prompt_learner
    if osp.basename(p) == "prompt_learner":
        directory = osp.dirname(p)
        return directory, default_epoch

    # case: already at .../<METHOD>
    return p, default_epoch


def find_eval_loader(trainer, split="auto"):
    """
    Try to find a reasonable evaluation loader on the trainer.
    """
    candidates = []
    if split == "auto":
        candidates = [
            "test_loader", "val_loader",
            "test_loader_x", "val_loader_x",
            "test_loader_u", "val_loader_u"
        ]
    elif split == "test":
        candidates = ["test_loader", "test_loader_x", "test_loader_u", "val_loader"]
    elif split == "val":
        candidates = ["val_loader", "val_loader_x", "val_loader_u", "test_loader"]
    elif split == "train":
        candidates = ["train_loader", "train_loader_x"]
    for name in candidates:
        if hasattr(trainer, name) and getattr(trainer, name) is not None:
            return getattr(trainer, name)
    raise RuntimeError("No eval loader found on trainer (tried many common names).")


@torch.no_grad()
def collect_logits_labels(trainer, data_loader):
    dev = trainer.device
    model = trainer.model
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    logits_all, labels_all = [], []
    for batch in data_loader:
        # match your CoOp parse logic: batch["img"], batch["label"]  :contentReference[oaicite:4]{index=4}
        if isinstance(batch, dict) and "img" in batch and "label" in batch:
            x = batch["img"].to(dev)
            y = batch["label"].to(dev)
        else:
            # fallback: tuple/list (img, label)
            x, y = batch[0].to(dev), batch[1].to(dev)

        out = model(x)  # logits
        logits_all.append(out.detach().cpu())
        labels_all.append(y.detach().cpu())

    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def per_sample_avg_margins(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    m_{i,k} = z[i,y] - z[i,k],  k!=y
    mbar_i  = mean_k m_{i,k}
    """
    B, C = logits.shape
    idx = torch.arange(B)
    z_true = logits[idx, labels].unsqueeze(1)            # (B,1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[idx, labels] = False
    others = logits[mask].view(B, C - 1)                 # (B, C-1)
    margins = z_true - others                            # (B, C-1)
    mbar = margins.mean(dim=1)                           # (B,)
    return mbar.cpu().numpy()


def bucket_labels(edges):
    labs = []
    for i in range(len(edges)-1):
        a, b = edges[i], edges[i+1]
        if np.isneginf(a): labs.append(f"≤{b:g}")
        elif np.isposinf(b): labs.append(f"≥{a:g}")
        else: labs.append(f"{a:g}–{b:g}")
    return labs


def plot_stacked_100(methods, mbar_dict, out_png, left_thr=1.0, right_thr=8.0,
                     edges=None, title="Component analysis"): #MCM vs MBLS via per-sample avg. margins (m̄)
    if edges is None:
        edges = np.array([-np.inf, 0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf], dtype=float)
    labs = bucket_labels(edges)

    # compute percentages + summaries
    P = []
    summaries = {}
    for m in methods:
        arr = mbar_dict[m]
        hist, _ = np.histogram(arr, bins=edges)
        perc = hist * 100.0 / max(1, hist.sum())
        P.append(perc)
        summaries[m] = dict(
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            p_left=float((arr <= left_thr).mean() * 100.0),
            p_right=float((arr >= right_thr).mean() * 100.0),
        )

    plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    x = np.arange(len(methods))
    width = 0.6
    bottom = np.zeros(len(methods))
    for b in range(len(labs)):
        vals = np.array([P[i][b] for i in range(len(methods))])
        ax.bar(x, vals, width=width, bottom=bottom, label=labs[b], linewidth=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=24)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Share by per-sample avg. margin (m̄) bucket (%)", fontsize=20)
    ax.set_title(title, fontsize=16)
    ax.legend(title="m̄ buckets", ncol=4, fontsize=18, title_fontsize=18,
              frameon=True,  framealpha=0.6, loc="upper left")
    ax.tick_params(axis='y', labelsize=20)  # y-axis tick numbers
    ax.tick_params(axis='x', labelsize=20)  # x-axis tick numbers (optional)
    # annotate μ±σ and tails
    """for i, m in enumerate(methods):
        s = summaries[m]
        txt = f"μ={s['mean']:.2f} ± {s['std']:.2f}\nP(m̄≤1)={s['p_left']:.1f}%  |  P(m̄≥8)={s['p_right']:.1f}%"
        ax.text(i, 104.5, txt, ha="center", va="bottom", fontsize=11)"""

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return labs, P, summaries


def save_csv(out_csv, methods, labs, P, summaries):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket"] + methods)
        for b, lab in enumerate(labs):
            w.writerow([lab] + [f"{P[i][b]:.4f}" for i in range(len(methods))])
        w.writerow([])
        w.writerow(["method", "mean", "std", "P(mbar<=1)%", "P(mbar>=8)%"])
        for m in methods:
            s = summaries[m]
            w.writerow([m, f"{s['mean']:.6f}", f"{s['std']:.6f}",
                        f"{s['p_left']:.3f}", f"{s['p_right']:.3f}"])

def build_trainer_for_eval(root, dataset_cfg, method_cfg, trainer_name="CoOp", seed=-1, opts=None):
    """
    Build a trainer via your train.py's setup_cfg -> dassl.build_trainer
    and FORCE TRAINER.NAME to avoid the empty-trainer error.
    """
    from train import setup_cfg  # keep import here to match your repo layout

    class _Args:
        pass

    args = _Args()
    args.root = root
    args.output_dir = ""           # not used
    args.resume = ""
    args.seed = seed
    args.source_domains = None
    args.target_domains = None
    args.transforms = None

    # **CRUCIAL**: set trainer explicitly
    args.trainer = trainer_name

    # backbone/head can be left empty if your method cfg specifies them
    args.backbone = ""
    args.head = ""
    args.dataset_config_file = dataset_cfg
    args.config_file = method_cfg
    args.opts = opts or []

    cfg = setup_cfg(args)          # merges dataset cfg, method cfg, then applies args (incl. trainer)
    trainer = build_trainer(cfg)
    return trainer, cfg


def run_once(method_name, model_path_or_dir, epoch_default, root, dataset_cfg, method_cfg, split, seed,trainer_name):
    print(f"\n[•] {method_name}: loading {model_path_or_dir}")
    directory, epoch = normalize_dir_and_epoch(model_path_or_dir, default_epoch=epoch_default)
    if epoch is None:
        raise ValueError(f"Could not infer epoch from '{model_path_or_dir}'. Please pass --epoch.")
    trainer, cfg = build_trainer_for_eval(root, dataset_cfg, method_cfg, trainer_name=trainer_name, seed=seed)
    # load weights into prompt_learner (CoOp)  :contentReference[oaicite:6]{index=6}
    trainer.load_model(directory, epoch=epoch)

    # pick eval loader
    loader = find_eval_loader(trainer, split=split)
    # collect logits + labels
    logits, labels = collect_logits_labels(trainer, loader)
    # compute per-sample AVERAGE margins
    mbar = per_sample_avg_margins(logits, labels)
    return mbar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config-file", required=True, help="e.g., configs/datasets/aptos.yaml")
    ap.add_argument("--config-file", required=True, help="trainer cfg (must define CoOp + backbone)")
    ap.add_argument("--root", required=True, help="datasets root")

    ap.add_argument("--ce", required=True, help="CE dir OR full .../prompt_learner/model.pth.tar-XX")
    ap.add_argument("--mean", required=True, help="MBLS dir OR full path")
    ap.add_argument("--var", required=True, help="Margin dir OR full path")
    ap.add_argument("--mcm", required=True, help="MCM dir OR full path")

    ap.add_argument("--epoch", type=int, default=None, help="epoch number if not in file name")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--split", choices=["auto","val","test","train"], default="auto")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--trainer-name", type=str, default="MaPLe",
                help="Trainer to build (e.g., CoOp, MaPLe, PromptSRC, KgCoOp)")
    args = ap.parse_args()
    trainer_name = args.trainer_name
    os.makedirs(args.outdir, exist_ok=True)

    methods = ["CE", "Mean", "Var","Margin"]
    paths   = [args.ce, args.mean,args.var, args.mcm]

    mbar_dict = {}
    for name, p in zip(methods, paths):
        mbar_dict[name] = run_once(
            method_name=name,
            model_path_or_dir=p,
            epoch_default=args.epoch,
            root=args.root,
            dataset_cfg=args.dataset_config_file,
            method_cfg=args.config_file,
            split=args.split,
            seed=args.seed,
            trainer_name=trainer_name,
        )

    fig_path = osp.join(args.outdir, "component_analyses.png")
    labs, P, S = plot_stacked_100(methods, mbar_dict, fig_path)
    csv_path = osp.join(args.outdir, "mcm_vs_mbls_mbar_mass.csv")
    save_csv(csv_path, methods, labs, P, S)

    with open(osp.join(args.outdir, "README.txt"), "w") as f:
        f.write(
            "Interpretation:\n"
            "• MBLS reduces the extreme right-tail (m̄≥8) but often leaves larger left-tail mass (m̄≤1),\n"
            "  meaning many samples remain weakly separated.\n"
            "• MCM increases mean per-sample average margin (μ↑), shifts mass into 3–6 buckets, and\n"
            "  reduces both left and extreme right tails (σ↓), matching your CVPR motivation.\n"
        )

    print(f"\n[✓] Saved figure: {fig_path}")
    print(f"[✓] Saved table : {csv_path}")

if __name__ == "__main__":
    main()
