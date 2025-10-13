"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std (or 95% CI) across seed folders
3. (NEW) Consolidate averages for ALL metrics across ALL datasets into ONE file

Usage
---
Single directory (backward compatible):
$ python parse_test_res.py output/my_experiment --test-log --keyword accuracy

Multiple metrics + deep scan + one consolidated CSV (recommended):
$ python parse_test_res.py /storagepool/Ashshak/output2/all \
    --scan-deep --test-log \
    --keywords accuracy,confidence,ece,mce,ace,ece_kde \
    --path-filter "shots_8/CoOp/vit_b32_plip_c16_ep50_batch16" \
    --ci95 \
    --consolidate-file /storagepool/Ashshak/output2/summaries/CoOp_vit_b32_plip_c16_ep50_batch16_shots8.csv \
    --wide --save-mode overwrite
"""
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
    # t* (two-sided, 95%) for very small n; else ~1.96
    t_map = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
    t_star = t_map.get(n, 1.96)
    return t_star * s / np.sqrt(n)


def results_to_csv(args, directory, key, results):
    """Original CSV writers (kept for backward compat)."""
    if 'train_base' in directory or 'test_new' in directory:
        base2new_results_to_csv(args, directory, key, results)
    elif 'xd_test' in directory or 'xd_train' in directory:
        xd_results_to_csv(args, directory, key, results)


def base2new_results_to_csv(args, directory, key, results):
    parts = directory.split("/")
    split = parts[2]
    dataset = parts[3]
    shot = int(parts[4].split("_")[1])
    algorithm = parts[5]
    cfgs = parts[6]

    if args.calibration_config:
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

    df = pd.DataFrame({
        "dataset": [dataset],
        "split": [split],
        "shot": [shot],
        "algorithm": [algorithm],
        "cfgs": [cfgs],
        "metrics": [key],
        "results": [results]
    })

    csv_file = "output/base2new/logs_base2new.csv"
    os.makedirs(osp.dirname(csv_file), exist_ok=True)
    if osp.exists(csv_file):
        pd.concat([pd.read_csv(csv_file), df], ignore_index=True).to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, index=False)


def xd_results_to_csv(args, directory, key, results):
    parts = directory.split("/")
    split = parts[2]
    algorithm = parts[3]
    cfgs = parts[4]
    dataset = parts[5]

    # Optional arg; keep safe
    calib_label = getattr(args, "calibration", "")
    if calib_label:
        algorithm = algorithm + '+' + calib_label

    df = pd.DataFrame({
        "dataset": [dataset],
        "split": [split],
        "algorithm": [algorithm],
        "cfgs": [cfgs],
        "metrics": [key],
        "results": [results]
    })

    csv_file = "output/xd/logs_xd.csv"
    os.makedirs(osp.dirname(csv_file), exist_ok=True)
    if osp.exists(csv_file):
        pd.concat([pd.read_csv(csv_file), df], ignore_index=True).to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, index=False)


def _build_metrics_from_args(args):
    """Support multiple metrics via --keywords, fallback to --keyword."""
    if args.keywords:
        ks = [k.strip() for k in args.keywords.split(",") if k.strip()]
    else:
        ks = [args.keyword]
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
    """Extract dataset/shot/algorithm/cfgs from known layouts."""
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


def _save_consolidated(rows, path, mode, wide=False):
    if not path:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    ext = osp.splitext(path)[1].lower()
    df = pd.DataFrame(rows)

    if wide:
        index_cols = ["dataset", "shot", "algorithm", "cfgs"]
        have = [c for c in index_cols if c in df.columns]
        if not df.empty:
            df = df.pivot_table(index=have, columns="metric", values="mean", aggfunc="first").reset_index()

    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".json":
        with open(path, "w" if mode == "overwrite" else "a") as f:
            for _, r in df.iterrows():
                f.write(json.dumps(r.to_dict()) + "\n")
    else:
        with open(path, "w" if mode == "overwrite" else "a") as f:
            if wide and not df.empty:
                f.write("\t".join(df.columns) + "\n")
                for _, r in df.iterrows():
                    f.write("\t".join(str(r[c]) for c in df.columns) + "\n")
            else:
                for r in rows:
                    f.write(
                        f"{r['dataset']}\tshot={r.get('shot','')}\talg={r.get('algorithm','')}\t"
                        f"cfg={r.get('cfgs','')}\tmetric={r['metric']}\tmean={r['mean']:.6f}\t"
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

                # Be flexible on the gate line
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
        base_std = np.std(values, ddof=1) if n > 1 else 0.0
        avg = float(np.mean(values))
        disp = compute_ci95(values) if args.ci95 else base_std
        print(f"* {key}: {avg:.2f}% +- {disp:.2f}%")
        output_results[key] = avg
        results_to_csv(args, directory, key, f"{avg:.2f}")
        summary_rows.append({
            "metric": key,
            "mean": avg,
            "dispersion": float(disp),
            "n": n
        })
    print("===")

    return output_results, summary_rows


def main(args, end_signal):
    metrics = _build_metrics_from_args(args)

    # Single-directory (original behavior)
    if not args.scan_deep:
        parse_function(*metrics, directory=args.directory, args=args, end_signal=end_signal)
        return

    # Consolidated mode
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
                print(f"{it['metric']}: {it['mean']:.2f} ({it['dispersion_type']} {it['dispersion']:.2f}, n={it['n']})")


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

    # NEW consolidated options
    parser.add_argument("--scan-deep", action="store_true",
                        help="recursively scan `directory` to find result dirs (contain seed*/log*.txt)")
    parser.add_argument("--path-filter", type=str, default="",
                        help="regex to filter result directories by full path (used with --scan-deep)")
    parser.add_argument("--consolidate-file", type=str, default="",
                        help="write one consolidated file (csv|json|txt) with averages across datasets/metrics")
    parser.add_argument("--save-mode", choices=["append", "overwrite"], default="overwrite",
                        help="append to or overwrite the consolidate file")
    parser.add_argument("--wide", action="store_true",
                        help="save a wide table: one row per dataset (and config), columns are metrics")

    args = parser.parse_args()

    end_signal = "=> result" if args.test_log else "Finished training"
    main(args, end_signal)
