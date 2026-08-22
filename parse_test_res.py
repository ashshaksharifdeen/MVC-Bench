import re
import argparse
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    """95% CI using Student’s t for small n."""
    res = np.asarray(res, dtype=float)
    n = len(res)
    if n <= 1:
        return 0.0
    s = np.std(res, ddof=1)
    t_map = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_star = t_map.get(n, 1.96)
    return float(t_star * s / np.sqrt(n))


def _safe_append(csv_path: str, row_df: pd.DataFrame, col_order=None):
    """Append row_df to csv_path, creating file if needed; upgrade missing cols safely."""
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    if osp.exists(csv_path):
        old = pd.read_csv(csv_path)
        # add missing columns on both sides
        for c in row_df.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in row_df.columns:
                row_df[c] = np.nan
        out = pd.concat([old, row_df], ignore_index=True)
        if col_order:
            rest = [c for c in out.columns if c not in col_order]
            out = out[col_order] + out[rest]
        out.to_csv(csv_path, index=False)
    else:
        if col_order:
            rest = [c for c in row_df.columns if c not in col_order]
            row_df = row_df[col_order] + row_df[rest]
        row_df.to_csv(csv_path, index=False)


def results_to_csv(args, directory, key, mean_value: float, std_value: float):
    """Dispatch CSV writer for legacy per-run logs (can be disabled)."""
    if args.disable_per_run_csv:
        return
    if 'train_base' in directory or 'test_new' in directory:
        base2new_results_to_csv(args, directory, key, mean_value, std_value)
    elif 'xd_test' in directory or 'xd_train' in directory:
        xd_results_to_csv(args, directory, key, mean_value, std_value)
    else:
        generic_results_to_csv(args, directory, key, mean_value, std_value)


def _augment_algorithm_name_with_calibration(algorithm: str, args) -> str:
    """Reproduce your calibration suffix logic safely."""
    if not getattr(args, "calibration_config", ""):
        return algorithm
    try:
        calibration_cfgs = json.loads(args.calibration_config)
        if calibration_cfgs.get('BASE_CALIBRATION_MODE'):
            if calibration_cfgs.get('SCALING_CONFIG'):
                algorithm += f"+{calibration_cfgs.get('SCALING_CALIBRATOR_NAME','')}"
            if calibration_cfgs.get('BIN_CALIBRATOR_NAME'):
                algorithm += f"+{calibration_cfgs['BIN_CALIBRATOR_NAME']}"
        if calibration_cfgs.get('IF_DAC'):
            algorithm += '+DAC'
        if calibration_cfgs.get('IF_PROCAL'):
            algorithm += '+ProCal'
    except Exception:
        pass
    return algorithm


def base2new_results_to_csv(args, directory, key, mean_value: float, std_value: float):
    parts = directory.split("/")
    # .../<split>/<dataset>/shots_K/<algorithm>/<cfgs>/seedX/log.txt
    split = parts[2]
    dataset = parts[3]
    shot = int(parts[4].split("_")[1])
    algorithm = parts[5]
    cfgs = parts[6]
    algorithm = _augment_algorithm_name_with_calibration(algorithm, args)

    df = pd.DataFrame({
        "dataset": [dataset],
        "split": [split],
        "shot": [shot],
        "algorithm": [algorithm],
        "cfgs": [cfgs],
        "metrics": [key],
        "results_mean": [float(mean_value)],
        "results_std": [float(std_value)],
    })

    csv_file = "output/base2new/logs_base2new.csv"
    desired_order = ["dataset", "split", "shot", "algorithm", "cfgs", "metrics", "results_mean", "results_std"]
    _safe_append(csv_file, df, col_order=desired_order)


def xd_results_to_csv(args, directory, key, mean_value: float, std_value: float):
    parts = directory.split("/")
    # .../<split>/<algorithm>/<cfgs>/<dataset>/seedX/log.txt
    split = parts[2]
    algorithm = parts[3]
    cfgs = parts[4]
    dataset = parts[5]

    calib_label = getattr(args, "calibration", "")
    if calib_label:
        algorithm = algorithm + '+' + calib_label

    df = pd.DataFrame({
        "dataset": [dataset],
        "split": [split],
        "algorithm": [algorithm],
        "cfgs": [cfgs],
        "metrics": [key],
        "results_mean": [float(mean_value)],
        "results_std": [float(std_value)],
    })

    csv_file = "output/xd/logs_xd.csv"
    desired_order = ["dataset", "split", "algorithm", "cfgs", "metrics", "results_mean", "results_std"]
    _safe_append(csv_file, df, col_order=desired_order)


def generic_results_to_csv(args, directory, key, mean_value: float, std_value: float):
    df = pd.DataFrame({
        "directory": [directory],
        "metrics": [key],
        "results_mean": [float(mean_value)],
        "results_std": [float(std_value)]
    })
    csv_file = "output/logs_generic.csv"
    desired_order = ["directory", "metrics", "results_mean", "results_std"]
    _safe_append(csv_file, df, col_order=desired_order)


def _build_metrics_from_args(args):
    """Support multiple metrics via --keywords, fallback to --keyword."""
    if args.keywords:
        ks = [k.strip() for k in args.keywords.split(",") if k.strip()]
    else:
        ks = [args.keyword]
    # pattern like: "* accuracy: 82.34%"
    return [{"name": k, "regex": re.compile(rf"\* {re.escape(k)}: ([\.\deE+-]+)%")} for k in ks]


def _is_result_dir(directory):
    """A result dir contains seed*/log*.txt."""
    try:
        for s in listdir_nohidden(directory, sort=True):
            sd = osp.join(directory, s)
            if osp.isdir(sd):
                for fname in os.listdir(sd):
                    if fname.startswith("log") and fname.endswith(".txt"):
                        return True
    except Exception:
        return False
    return False


def _find_result_dirs(root, path_filter_regex=None):
    hits = []
    for cur, dirs, files in os.walk(root):
        if _is_result_dir(cur):
            if (path_filter_regex is None) or re.search(path_filter_regex, cur):
                hits.append(cur)
            dirs[:] = []  # don't descend further under a result dir
    return sorted(set(hits))


def _parse_path_meta(directory):
    """Extract dataset/shot/algorithm/cfgs from common layouts."""
    parts = directory.strip("/").split("/")
    meta = {"split": "", "dataset": "", "shot": None, "algorithm": "", "cfgs": ""}
    # base2new-like
    if len(parts) >= 7 and parts[2] in ("train_base", "test_new"):
        meta["split"] = parts[2]
        meta["dataset"] = parts[3]
        meta["shot"] = int(parts[4].split("_")[1])
        meta["algorithm"] = parts[5]
        meta["cfgs"] = parts[6]
        return meta
    # fallback: .../<dataset>/shots_<k>/<algo>/<cfg>/...
    try:
        i = next(i for i, p in enumerate(parts) if p.startswith("shots_"))
        meta["split"] = "all"
        meta["dataset"] = parts[i - 1]
        meta["shot"] = int(parts[i].split("_")[1])
        meta["algorithm"] = parts[i + 1]
        meta["cfgs"] = parts[i + 2]
    except Exception:
        meta["dataset"] = parts[-1]
    return meta


def _interleave_mean_std(mean_piv: pd.DataFrame, std_piv: pd.DataFrame):
    """Return a dataframe with columns interleaved as metric, metric_std."""
    # preserve metric order by appearance in mean_piv columns
    metrics = list(mean_piv.columns)
    interleaved = pd.DataFrame(index=mean_piv.index)
    for m in metrics:
        interleaved[m] = mean_piv[m]
        # std_piv may lack a column if std couldn't be computed (n==1); handle gracefully
        interleaved[f"{m}_std"] = std_piv[m] if m in std_piv.columns else np.nan
    return interleaved


def _save_consolidated(rows, path, mode, wide=False):
    if not path:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    ext = osp.splitext(path)[1].lower()
    df = pd.DataFrame(rows)

    if wide and not df.empty:
        index_cols = ["dataset", "shot", "algorithm", "cfgs"]
        have = [c for c in index_cols if c in df.columns]
        # build mean and std pivot tables
        mean_piv = df.pivot_table(index=have, columns="metric", values="mean", aggfunc="first")
        std_piv = df.pivot_table(index=have, columns="metric", values="std", aggfunc="first")
        wide_df = _interleave_mean_std(mean_piv, std_piv).reset_index()
        out_df = wide_df
    else:
        out_df = df

    if ext == ".csv":
        out_df.to_csv(path, index=False)
    elif ext == ".json":
        with open(path, "w" if mode == "overwrite" else "a") as f:
            for _, r in out_df.iterrows():
                f.write(json.dumps(r.to_dict()) + "\n")
    else:
        with open(path, "w" if mode == "overwrite" else "a") as f:
            if wide and not df.empty:
                f.write("\t".join(out_df.columns) + "\n")
                for _, r in out_df.iterrows():
                    f.write("\t".join(str(r[c]) for c in out_df.columns) + "\n")
            else:
                for r in rows:
                    f.write(
                        f"{r['dataset']}\tshot={r.get('shot','')}\talg={r.get('algorithm','')}\t"
                        f"cfg={r.get('cfgs','')}\tmetric={r['metric']}\tmean={r['mean']:.6f}\tstd={r['std']:.6f}\t"
                        f"{r['dispersion_type']}={r['dispersion']:.6f}\tn={r['n']}\tpath={r['directory']}\n"
                    )


def parse_function(*metrics, directory="", args=None, end_signal=None):
    """Parse one experiment directory (containing seed*/log*.txt)."""
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        base_path = osp.join(directory, subdir)
        base_name = "log"

        # Optional calibration suffixes for log file naming
        calib = args.calibration_config
        if isinstance(calib, str) and calib.strip():
            try:
                calibration_cfgs = json.loads(calib)
                if calibration_cfgs.get('BASE_CALIBRATION_MODE'):
                    if calibration_cfgs.get('SCALING_CONFIG'):
                        base_name += f"_{calibration_cfgs.get('SCALING_CALIBRATOR_NAME','')}"
                    if calibration_cfgs.get('BIN_CALIBRATOR_NAME'):
                        base_name += f"_{calibration_cfgs['BIN_CALIBRATOR_NAME']}"
                if calibration_cfgs.get('IF_DAC'):
                    base_name += "_dac"
                if calibration_cfgs.get('IF_PROCAL'):
                    base_name += "_procal"
            except Exception:
                pass

        base_name = base_name + ".txt"
        fpath = osp.join(base_path, base_name)

        if not check_isfile(fpath):
            print(f"[WARN] Missing log: {fpath}")
            continue

        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                # Gate line
                if end_signal and (end_signal in line):
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    if len(outputs) == 0:
        print(f"[WARN] Nothing found in {directory}")
        return OrderedDict(), []  # keep consolidated flow alive

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()
    summary_rows = []

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        n = len(values)
        base_std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        avg = float(np.mean(values))
        disp = compute_ci95(values) if args.ci95 else base_std
        print(f"* {key}: {avg:.2f}% +- {disp:.2f}% (std={base_std:.2f}%)")
        output_results[key] = avg
        # per-run csvs (optional)
        results_to_csv(args, directory, key, avg, base_std)
        summary_rows.append({
            "metric": key,
            "mean": avg,
            "std": base_std,
            "dispersion": float(disp),
            "n": n
        })
    print("===")

    return output_results, summary_rows


def main(args, end_signal):
    metrics = _build_metrics_from_args(args)

    # Single-directory parse
    if not args.scan_deep:
        parse_function(*metrics, directory=args.directory, args=args, end_signal=end_signal)
        return

    # Consolidated parse
    path_filter_regex = args.path_filter if args.path_filter else None
    result_dirs = _find_result_dirs(args.directory, path_filter_regex)

    if not result_dirs:
        print(f"[WARN] No result directories found under: {args.directory}")
        return

    rows = []
    for rd in result_dirs:
        meta = _parse_path_meta(rd)
        _, out_rows = parse_function(*metrics, directory=rd, args=args, end_signal=end_signal)
        for r in out_rows:
            rows.append({
                "directory": rd,
                "split": meta.get("split", ""),
                "dataset": meta.get("dataset", ""),
                "shot": meta.get("shot", ""),
                "algorithm": meta.get("algorithm", ""),
                "cfgs": meta.get("cfgs", ""),
                "metric": r["metric"],
                "mean": r["mean"],
                "std": r["std"],
                "dispersion": r["dispersion"],
                "dispersion_type": "ci95" if args.ci95 else "std",
                "n": r["n"],
            })

    if args.consolidate_file:
        _save_consolidated(rows, args.consolidate_file, args.save_mode, wide=args.wide)
        print(f"[OK] Wrote consolidated results: {args.consolidate_file}")
    else:
        # Print compact per-dataset table
        by_ds = defaultdict(list)
        for r in rows:
            by_ds[r["dataset"]].append(r)
        for ds, items in by_ds.items():
            print(f"\n## {ds}")
            for it in items:
                print(f"{it['metric']}: {it['mean']:.2f} "
                      f"(std {it['std']:.2f}, {it['dispersion_type']} {it['dispersion']:.2f}, n={it['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to root or single experiment directory")
    parser.add_argument("--ci95", action="store_true", help=r"compute 95% confidence interval")
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs (gate on '=> result')")
    parser.add_argument("--multi-exp", action="store_true", help="(legacy) parse multiple experiments")
    parser.add_argument("--keyword", default="accuracy", type=str, help="single metric (legacy)")
    parser.add_argument("--keywords", type=str, default="", help="comma-separated metrics to extract")
    parser.add_argument("--calibration-config", default="", type=str, help="JSON string for calibration flags")
    parser.add_argument("--calibration", default="", type=str, help="optional label for xd runs")

    # Consolidation / output controls
    parser.add_argument("--scan-deep", action="store_true",
                        help="recursively scan `directory` to find result dirs (contain seed*/log*.txt)")
    parser.add_argument("--path-filter", type=str, default="",
                        help="regex to filter result directories by full path (used with --scan-deep)")
    parser.add_argument("--consolidate-file", type=str, default="",
                        help="write one consolidated file (csv|json|txt)")
    parser.add_argument("--save-mode", choices=["append", "overwrite"], default="overwrite",
                        help="append to or overwrite the consolidate file")
    parser.add_argument("--wide", action="store_true",
                        help="**wide table**: interleaved columns {metric}, {metric}_std")
    parser.add_argument("--disable-per-run-csv", action="store_true",
                        help="disable legacy per-run CSV writers, only produce --consolidate-file")

    args = parser.parse_args()
    end_signal = "=> result" if args.test_log else "Finished training"
    main(args, end_signal)
