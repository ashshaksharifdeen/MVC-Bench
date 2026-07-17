#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Values (read from your table)
# -----------------------------
TABLE = {
    "batch": {
        "labels": ["8 Base", "8 Novel", "16 Base", "16 Novel"],
        "acc": {
            "MaPLe": [90.00, 77.79, 88.03, 76.07],
            "Ours":  [90.00, 77.75, 87.62, 77.98],
        },
        "ece": {
            "MaPLe": [2.69, 7.17, 2.41, 6.12],
            "Ours":  [2.01, 4.62, 1.91, 3.45],
        },
    },
    "ctx": {
        "labels": ["3 Base", "3 Novel", "5 Base", "5 Novel"],
        "acc": {
            "MaPLe": [90.77, 77.93, 90.77, 76.78],
            "Ours":  [90.73, 77.07, 90.45, 76.23],
        },
        "ece": {
            "MaPLe": [2.04, 6.30, 2.47, 7.05],
            "Ours":  [1.51, 5.35, 1.64, 6.06],
        },
    },
}


def grouped_bar(
    ax,
    labels,
    series_dict,
    title,
    ylabel,
    *,
    colors=None,
    bar_width=0.35,
    title_fs=16,
    label_fs=13,
    tick_fs=11,
    legend_fs=11,
    show_legend=False,
    legend_loc="upper right",
    grid=True,
    ylim=None,
    rotate=15,
):
    methods = list(series_dict.keys())
    n = len(labels)
    x = np.arange(n)

    # Center bars around each x
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width

    for i, m in enumerate(methods):
        c = None if colors is None else colors.get(m, None)
        ax.bar(
            x + offsets[i],
            series_dict[m],
            width=bar_width,
            label=m,
            color=c,
        )

    ax.set_title(title, fontsize=title_fs, pad=8)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right", fontsize=tick_fs)
    ax.tick_params(axis="y", labelsize=tick_fs)

    if ylim is not None:
        ax.set_ylim(*ylim)

    if grid:
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    if show_legend:
        ax.legend(fontsize=legend_fs, loc=legend_loc, frameon=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_fig", help="output directory")
    ap.add_argument("--name", default="table_to_figure", help="output filename stem")
    ap.add_argument("--dpi", type=int, default=200, help="dpi for png")
    ap.add_argument("--save_pdf", action="store_true", help="also save PDF")

    # ---- style knobs ----
    ap.add_argument("--fig_w", type=float, default=15.0, help="figure width (inches)")
    ap.add_argument("--fig_h", type=float, default=10.0, help="figure height (inches)")
    ap.add_argument("--title_fs", type=float, default=18, help="subplot title font size")
    ap.add_argument("--label_fs", type=float, default=20, help="axis label font size")
    ap.add_argument("--tick_fs", type=float, default=24, help="tick label font size")
    ap.add_argument("--legend_fs", type=float, default=16, help="legend font size")
    ap.add_argument("--bar_width", type=float, default=0.35, help="bar width")
    ap.add_argument("--xrot", type=float, default=15, help="x tick label rotation")
    ap.add_argument("--no_grid", action="store_true", help="disable grid")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Colors to match your attached figure (orange + green)
    colors = {
        "MaPLe": "tab:orange",
        "Ours": "tab:green",
    }

    fig, axs = plt.subplots(2, 2, figsize=(args.fig_w, args.fig_h))

    # Top-left: Batch size (Accuracy)
    grouped_bar(
        axs[0, 0],
        TABLE["batch"]["labels"],
        TABLE["batch"]["acc"],
        title="Batch size comparison (Accuracy)",
        ylabel="Accuracy (%)",
        colors=colors,
        bar_width=args.bar_width,
        title_fs=args.title_fs,
        label_fs=args.label_fs,
        tick_fs=args.tick_fs,
        legend_fs=args.legend_fs,
        show_legend=True,                 # legend only here (like your example)
        legend_loc="upper right",
        grid=not args.no_grid,
        ylim=(70, 95),
        rotate=args.xrot,
    )

    # Top-right: Context length/Vocab size (Accuracy)
    grouped_bar(
        axs[0, 1],
        TABLE["ctx"]["labels"],
        TABLE["ctx"]["acc"],
        title="Context length/Vocab size comparison (Accuracy)",
        ylabel="Accuracy (%)",
        colors=colors,
        bar_width=args.bar_width,
        title_fs=args.title_fs,
        label_fs=args.label_fs,
        tick_fs=args.tick_fs,
        legend_fs=args.legend_fs,
        show_legend=False,
        grid=not args.no_grid,
        ylim=(70, 95),
        rotate=args.xrot,
    )

    # Bottom-left: Batch size (ECE)
    grouped_bar(
        axs[1, 0],
        TABLE["batch"]["labels"],
        TABLE["batch"]["ece"],
        title="Batch size comparison (ECE)",
        ylabel="ECE",
        colors=colors,
        bar_width=args.bar_width,
        title_fs=args.title_fs,
        label_fs=args.label_fs,
        tick_fs=args.tick_fs,
        legend_fs=args.legend_fs,
        show_legend=False,
        grid=not args.no_grid,
        ylim=(0, 8),
        rotate=args.xrot,
    )

    # Bottom-right: Context length/Vocab size (ECE)
    grouped_bar(
        axs[1, 1],
        TABLE["ctx"]["labels"],
        TABLE["ctx"]["ece"],
        title="Context length/Vocab size comparison (ECE)",
        ylabel="ECE",
        colors=colors,
        bar_width=args.bar_width,
        title_fs=args.title_fs,
        label_fs=args.label_fs,
        tick_fs=args.tick_fs,
        legend_fs=args.legend_fs,
        show_legend=False,
        grid=not args.no_grid,
        ylim=(0, 8),
        rotate=args.xrot,
    )

    fig.tight_layout()

    out_png = os.path.join(args.outdir, f"{args.name}.png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print("Saved:", out_png)

    if args.save_pdf:
        out_pdf = os.path.join(args.outdir, f"{args.name}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
        print("Saved:", out_pdf)

    plt.close(fig)


if __name__ == "__main__":
    main()
