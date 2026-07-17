import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def quantile_table(df: pd.DataFrame, cols):
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append(
            {
                "metric": c,
                "median": float(s.quantile(0.50)),
                "q10": float(s.quantile(0.10)),
                "q90": float(s.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to scale_grad_log.csv")
    ap.add_argument("--outdir", required=True, help="where to save plot+table")

    # ---- STYLE KNOBS (edit these) ----
    ap.add_argument("--fig_w", type=float, default=10.0, help="figure width (inches)")
    ap.add_argument("--fig_h", type=float, default=8.0, help="figure height (inches)")
    ap.add_argument("--title_fs", type=float, default=16, help="title font size")
    ap.add_argument("--label_fs", type=float, default=20, help="x/y label font size")
    ap.add_argument("--tick_fs", type=float, default=24, help="tick label font size")
    ap.add_argument("--legend_fs", type=float, default=16, help="legend font size")
    ap.add_argument("--lw", type=float, default=2.2, help="line width")
    ap.add_argument("--dpi", type=int, default=300, help="save dpi")
    ap.add_argument("--x_major_step", type=int, default=0, help="0=auto else step (e.g., 50)")
    ap.add_argument("--y_major_step", type=float, default=0.0, help="0=auto else step (e.g., 10)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # ---- PLOT: three curves vs iteration ----
    x = df["iter"].values

    fig, ax = plt.subplots(figsize=(args.fig_w, args.fig_h))

    ax.plot(x, df["rho_margin"].values, label=r"$\rho_{\mathrm{margin}}$", linewidth=args.lw)
    ax.plot(x, df["rho_mom"].values, label=r"$\rho_{\mathrm{mom}}$", linewidth=args.lw)
    ax.plot(
        x,
        df["rho_mom_margin"].values,
        label=r"$\rho_{\mathrm{mom/margin}}$",
        linewidth=args.lw,
    )

    ax.set_title("Gradient-norm ratios vs iteration", fontsize=args.title_fs, pad=10)
    ax.set_xlabel("iteration", fontsize=args.label_fs, labelpad=6)
    ax.set_ylabel("ratio", fontsize=args.label_fs, labelpad=6)

    # tick number size
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fs)

    # optional: x ticks as integers + configurable spacing
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if args.x_major_step and args.x_major_step > 0:
        xmin, xmax = int(x.min()), int(x.max())
        ax.set_xticks(list(range(xmin - (xmin % args.x_major_step), xmax + 1, args.x_major_step)))

    # optional: configurable y spacing
    if args.y_major_step and args.y_major_step > 0:
        ymin, ymax = ax.get_ylim()
        start = (ymin // args.y_major_step) * args.y_major_step
        ticks = []
        t = start
        while t <= ymax + 1e-9:
            ticks.append(t)
            t += args.y_major_step
        ax.set_yticks(ticks)

    ax.legend(fontsize=args.legend_fs, loc="upper right", frameon=True)

    fig.tight_layout()
    plot_path = os.path.join(args.outdir, "rho_curves.png")
    fig.savefig(plot_path, dpi=args.dpi)
    plt.close(fig)

    # ---- TABLE: median + 10–90% quantiles ----
    cols = ["rho_margin", "rho_mom", "rho_mom_margin", "rho_margin_ce", "rho_mom_ce"]
    tab = quantile_table(df, cols)

    csv_path = os.path.join(args.outdir, "rho_quantiles.csv")
    tab.to_csv(csv_path, index=False)

    tex_path = os.path.join(args.outdir, "rho_quantiles.tex")
    with open(tex_path, "w") as f:
        f.write(tab.to_latex(index=False, float_format=lambda v: f"{v:.3g}"))

    print("Saved:")
    print(" -", plot_path)
    print(" -", csv_path)
    print(" -", tex_path)


if __name__ == "__main__":
    main()




"""import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def quantile_table(df: pd.DataFrame, cols):
    rows = []
    for c in cols:
        s = df[c].dropna()
        rows.append({
            "metric": c,
            "median": float(s.quantile(0.50)),
            "q10": float(s.quantile(0.10)),
            "q90": float(s.quantile(0.90)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to scale_grad_log.csv")
    ap.add_argument("--outdir", required=True, help="where to save plot+table")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # ---- PLOT: three curves vs iteration ----
    x = df["iter"].values

    plt.figure()
    plt.plot(x, df["rho_margin"].values, label=r"$\rho_{\mathrm{margin}}$")
    plt.plot(x, df["rho_mom"].values, label=r"$\rho_{\mathrm{mom}}$")
    plt.plot(x, df["rho_mom_margin"].values, label=r"$\rho_{\mathrm{mom/margin}}$")
    plt.xlabel("iteration")
    plt.ylabel("ratio")
    plt.title("Gradient-norm ratios vs iteration")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "rho_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # ---- TABLE: median + 10–90% quantiles ----
    cols = ["rho_margin", "rho_mom", "rho_mom_margin", "rho_margin_ce", "rho_mom_ce"]
    tab = quantile_table(df, cols)

    csv_path = os.path.join(args.outdir, "rho_quantiles.csv")
    tab.to_csv(csv_path, index=False)

    tex_path = os.path.join(args.outdir, "rho_quantiles.tex")
    with open(tex_path, "w") as f:
        f.write(tab.to_latex(index=False, float_format=lambda v: f"{v:.3g}"))

    print("Saved:")
    print(" -", plot_path)
    print(" -", csv_path)
    print(" -", tex_path)


if __name__ == "__main__":
    main()"""
