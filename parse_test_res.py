"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std (or 95% CI) across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run:

    python tools/parse_test_res.py output/my_experiment

Add --ci95 for 95% confidence interval instead of std:

    python tools/parse_test_res.py output/my_experiment --ci95

For multiple experiments (each subfolder is its own experiment), add --multi-exp:

    python tools/parse_test_res.py output/my_experiment --multi-exp
"""

import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    """95% confidence interval half-width."""
    return 1.96 * np.std(res) / np.sqrt(len(res))

def harmonic_mean(base_accuracy, novel_accuracy, eps=1e-12):
    """
    Compute the harmonic mean between base and novel accuracy.

    Both values must be represented using the same scale:
        85.89 and 77.99
    or:
        0.8589 and 0.7799
    """
    base_accuracy = float(base_accuracy)
    novel_accuracy = float(novel_accuracy)

    if not np.isfinite(base_accuracy):
        raise ValueError(
            f"Base accuracy must be finite, got {base_accuracy}"
        )

    if not np.isfinite(novel_accuracy):
        raise ValueError(
            f"Novel accuracy must be finite, got {novel_accuracy}"
        )

    if base_accuracy < 0.0 or novel_accuracy < 0.0:
        raise ValueError(
            "Base and novel accuracies must be non-negative"
        )

    denominator = base_accuracy + novel_accuracy

    if denominator <= eps:
        return 0.0

    return (
        2.0
        * base_accuracy
        * novel_accuracy
        / denominator
    )

def parse_accuracy_from_log(
    log_path,
    test_log=False,
    end_signal="Finish training"
):
    """
    Extract the final evaluation accuracy from one log.txt file.

    test_log=True:
        Used for --eval-only logs. Parsing begins when the evaluator
        prints '=> Total samples:'.

    test_log=False:
        Used for train-then-test logs. Parsing begins after
        'Finish training'.
    """
    accuracy_regex = re.compile(
        r"=>\s*Accuracy:\s*([\d\.eE+-]+)%",
        re.IGNORECASE
    )

    good_to_go = False
    accuracy = None
    all_accuracy_matches = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            # Keep a fallback copy of all accuracy values
            fallback_match = accuracy_regex.search(raw_line)

            if fallback_match:
                all_accuracy_matches.append(
                    float(fallback_match.group(1))
                )

            if test_log:
                if (
                    raw_line.startswith("=> Total samples:")
                    or raw_line.startswith("=> result")
                    or line.startswith("* total:")
                ):
                    good_to_go = True
            else:
                if line == end_signal:
                    good_to_go = True
                    continue

            if not good_to_go:
                continue

            match = accuracy_regex.search(raw_line)

            if match:
                # Retain the final matching test accuracy
                accuracy = float(match.group(1))

    # Support old or manually copied logs without the expected marker
    if accuracy is None and all_accuracy_matches:
        accuracy = all_accuracy_matches[-1]

    if accuracy is None:
        raise RuntimeError(
            f"No accuracy result was found in: {log_path}"
        )

    return accuracy

def collect_accuracy_by_seed(
    directory,
    test_log=False,
    end_signal="Finish training"
):
    """
    Expected directory structure:

        directory/
            seed1/
                log.txt
            seed2/
                log.txt
            seed3/
                log.txt

    A direct directory/log.txt layout is also supported.
    """
    outputs = OrderedDict()

    direct_log = osp.join(directory, "log.txt")

    if check_isfile(direct_log):
        outputs["root"] = {
            "accuracy": parse_accuracy_from_log(
                direct_log,
                test_log=test_log,
                end_signal=end_signal
            ),
            "file": direct_log,
        }
        return outputs

    subdirs = listdir_nohidden(directory, sort=True)

    for subdir in subdirs:
        log_path = osp.join(directory, subdir, "log.txt")

        if not check_isfile(log_path):
            continue

        outputs[subdir] = {
            "accuracy": parse_accuracy_from_log(
                log_path,
                test_log=test_log,
                end_signal=end_signal
            ),
            "file": log_path,
        }

    if not outputs:
        raise RuntimeError(
            f"No valid seed log files were found in: {directory}"
        )

    return outputs

def parse_base_to_novel_hm(
    base_directory,
    novel_directory,
    args,
    end_signal
):
    """
    Pair base and novel results by seed name and calculate HM.
    """
    base_results = collect_accuracy_by_seed(
        directory=base_directory,
        test_log=args.test_log,
        end_signal=end_signal
    )

    novel_results = collect_accuracy_by_seed(
        directory=novel_directory,
        test_log=args.test_log,
        end_signal=end_signal
    )

    base_seeds = set(base_results.keys())
    novel_seeds = set(novel_results.keys())

    common_seeds = sorted(base_seeds & novel_seeds)

    missing_novel = sorted(base_seeds - novel_seeds)
    missing_base = sorted(novel_seeds - base_seeds)

    if missing_novel:
        print(
            "Warning: no novel result for: "
            + ", ".join(missing_novel)
        )

    if missing_base:
        print(
            "Warning: no base result for: "
            + ", ".join(missing_base)
        )

    if not common_seeds:
        raise RuntimeError(
            "No matching seed folder names were found between "
            "the base and novel directories"
        )

    base_accuracies = []
    novel_accuracies = []
    seed_hms = []

    print("======================================")
    print("Base-to-Novel Harmonic Mean")
    print("======================================")

    for seed in common_seeds:
        base_accuracy = base_results[seed]["accuracy"]
        novel_accuracy = novel_results[seed]["accuracy"]

        hm = harmonic_mean(
            base_accuracy,
            novel_accuracy
        )

        base_accuracies.append(base_accuracy)
        novel_accuracies.append(novel_accuracy)
        seed_hms.append(hm)

        print(
            f"{seed}: "
            f"Base={base_accuracy:.2f}%  "
            f"Novel={novel_accuracy:.2f}%  "
            f"HM={hm:.2f}%"
        )

    mean_base = float(np.mean(base_accuracies))
    mean_novel = float(np.mean(novel_accuracies))

    base_spread = (
        compute_ci95(base_accuracies)
        if args.ci95
        else float(np.std(base_accuracies))
    )

    novel_spread = (
        compute_ci95(novel_accuracies)
        if args.ci95
        else float(np.std(novel_accuracies))
    )

    mean_seed_hm = float(np.mean(seed_hms))

    hm_spread = (
        compute_ci95(seed_hms)
        if args.ci95
        else float(np.std(seed_hms))
    )

    # Table-1 style calculation:
    # first average Base and Novel, then calculate HM
    paper_style_hm = harmonic_mean(
        mean_base,
        mean_novel
    )

    spread_name = "95% CI" if args.ci95 else "std"

    print("======================================")
    print(f"Matched seeds: {len(common_seeds)}")
    print(
        f"=> Mean Base Accuracy: "
        f"{mean_base:.2f}% ± {base_spread:.2f}% ({spread_name})"
    )
    print(
        f"=> Mean Novel Accuracy: "
        f"{mean_novel:.2f}% ± {novel_spread:.2f}% ({spread_name})"
    )
    print(
        f"=> Mean Seed-wise HM: "
        f"{mean_seed_hm:.2f}% ± {hm_spread:.2f}% ({spread_name})"
    )
    print(
        f"=> Harmonic Mean (HM): "
        f"{paper_style_hm:.2f}%"
    )
    print("======================================")

    return OrderedDict([
        ("base_accuracy", mean_base),
        ("novel_accuracy", mean_novel),
        ("hm_seedwise_mean", mean_seed_hm),
        ("hm", paper_style_hm),
    ])

def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)
    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath), f"Missing log.txt in {subdir}"

        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()

                if args.test_log:
                    # Works for both old and new evaluator logs
                    if (
                        raw_line.startswith("=> Total samples:")
                        or raw_line.startswith("=> result")
                        or line.startswith("* total:")
                    ):
                        good_to_go = True
                else:
                    if line == end_signal:
                        good_to_go = True
                        continue

                if not good_to_go:
                    continue

                for metric in metrics:
                    m = metric["regex"].search(raw_line)
                    if m:
                        if "file" not in output:
                            output["file"] = fpath
                        output[metric["name"]] = float(m.group(1))

        if output:
            outputs.append(output)

    assert outputs, f"No metrics found in {directory}"

    metrics_results = defaultdict(list)
    metric_units = {m["name"]: m.get("unit", "") for m in metrics}

    for out in outputs:
        msg = ""

        for k, v in out.items():
            if k == "file":
                msg += f"{v}  "
            else:
                unit = metric_units.get(k, "")

                if unit == "%":
                    msg += f"{k}: {v:.2f}%. "
                else:
                    msg += f"{k}: {v:.6f}. "

                metrics_results[k].append(v)

        print(msg)

    print("===")
    print(f"Summary of directory: {directory}")

    for k, vals in metrics_results.items():
        avg = np.mean(vals)
        std = compute_ci95(vals) if args.ci95 else np.std(vals)
        unit = metric_units.get(k, "")

        if unit == "%":
            print(f"* {k}: {avg:.2f}% ± {std:.2f}%")
        else:
            print(f"* {k}: {avg:.6f} ± {std:.6f}")

    print("===")

    return {k: np.mean(v) for k, v in metrics_results.items()}


def main(args, end_signal):
    # Define all metrics to extract:
    """metrics = [
        {"name": "accuracy",      "regex": re.compile(r"Accuracy:\s*([\d\.eE+-]+)%", re.IGNORECASE)},
        {"name": "ece",           "regex": re.compile(r"ECE:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
        {"name": "mce",           "regex": re.compile(r"MCE:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
        {"name": "adaptive_ece",  "regex": re.compile(r"Adaptive ECE:\s*([\d\.eE+-]+)%", re.IGNORECASE)},
        {"name": "macro_f1",      "regex": re.compile(r"Macro-F1:\s*([\d\.eE+-]+)%",    re.IGNORECASE)},
        {"name": "piece",         "regex": re.compile(r"Piece:\s*([\d\.eE+-]+)%",      re.IGNORECASE)},
    ]"""
        # Base-to-novel HM mode
    if args.base_dir or args.novel_dir:
        if not args.base_dir or not args.novel_dir:
            raise ValueError(
                "Both --base-dir and --novel-dir are required "
                "for harmonic-mean calculation"
            )

        parse_base_to_novel_hm(
            base_directory=args.base_dir,
            novel_directory=args.novel_dir,
            args=args,
            end_signal=end_signal
        )
        return
    metrics = [
    {
        "name": "accuracy",
        "regex": re.compile(r"=>\s*Accuracy:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "error_rate",
        "regex": re.compile(r"Error rate:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "macro_f1",
        "regex": re.compile(r"=>\s*Macro-F1:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "ece",
        "regex": re.compile(r"=>\s*ECE(?:@\d+)?:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "mce",
        "regex": re.compile(r"MCE:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "ace",
        "regex": re.compile(r"Adaptive ECE:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    {
        "name": "brier",
        "regex": re.compile(r"=>\s*Brier score:\s*([\d\.eE+-]+)", re.IGNORECASE),
        "unit": ""
    },
    {
        "name": "brier_norm",
        "regex": re.compile(r"Normalized Brier:\s*([\d\.eE+-]+)", re.IGNORECASE),
        "unit": ""
    },

    # ---------------------------------------------------------
    # Class-wise ECE from:
    # => Top-label Class-wise ECE: Macro=10.87% Weighted=10.61% Max=40.25%
    # ---------------------------------------------------------
    {
        "name": "toplabel_ece_macro",
        "regex": re.compile(
            r"=>\s*Top-label Class-wise ECE:\s*Macro=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },
    {
        "name": "toplabel_ece_weighted",
        "regex": re.compile(
            r"=>\s*Top-label Class-wise ECE:.*?Weighted=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },
    {
        "name": "toplabel_ece_max",
        "regex": re.compile(
            r"=>\s*Top-label Class-wise ECE:.*?Max=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },

    # ---------------------------------------------------------
    # One-vs-rest class-wise ECE from:
    # => One-vs-rest Class-wise ECE: Macro=0.45% Weighted=0.44% Max=1.33%
    # ---------------------------------------------------------
    {
        "name": "ovr_ece_macro",
        "regex": re.compile(
            r"=>\s*One-vs-rest Class-wise ECE:\s*Macro=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },
    {
        "name": "ovr_ece_weighted",
        "regex": re.compile(
            r"=>\s*One-vs-rest Class-wise ECE:.*?Weighted=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },
    {
        "name": "ovr_ece_max",
        "regex": re.compile(
            r"=>\s*One-vs-rest Class-wise ECE:.*?Max=([\d\.eE+-]+)%",
            re.IGNORECASE
        ),
        "unit": "%"
    },

    # ---------------------------------------------------------
    # Class-wise Brier from:
    # => Class-wise Brier: Macro=0.194387 Normalized Macro=0.003812
    # ---------------------------------------------------------
    {
        "name": "classwise_brier_macro",
        "regex": re.compile(
            r"=>\s*Class-wise Brier:\s*Macro=([\d\.eE+-]+)",
            re.IGNORECASE
        ),
        "unit": ""
    },
    {
        "name": "classwise_brier_norm_macro",
        "regex": re.compile(
            r"=>\s*Class-wise Brier:.*?Normalized Macro=([\d\.eE+-]+)",
            re.IGNORECASE
        ),
        "unit": ""
    },

    {
        "name": "piece",
        "regex": re.compile(r"=>\s*PIECE:\s*([\d\.eE+-]+)%", re.IGNORECASE),
        "unit": "%"
    },
    ]

    if args.multi_exp:
        # average across multiple experiments
        final = defaultdict(list)
        for exp in listdir_nohidden(args.directory, sort=True):
            exp_dir = osp.join(args.directory, exp)
            res = parse_function(*metrics, directory=exp_dir, args=args, end_signal=end_signal)
            for k, v in res.items():
                final[k].append(v)

        print("Average performance across experiments")
        for k, vals in final.items():
            avg = np.mean(vals)
            std = compute_ci95(vals) if args.ci95 else np.std(vals)
            #print(f"* {k}: {avg:.2f}% ± {std:.2f}%")
            metric_units = {m["name"]: m.get("unit", "") for m in metrics}
            unit = metric_units.get(k, "")

            if unit == "%":
                print(f"* {k}: {avg:.2f}% ± {std:.2f}%")
            else:
                print(f"* {k}: {avg:.6f} ± {std:.6f}")
    else:
        # single experiment (directory = seeds folder)
        parse_function(*metrics, directory=args.directory, args=args, end_signal=end_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("directory", type=str, help="path to experiment folder")
    parser.add_argument(
    "directory",
    type=str,
    nargs="?",
    default="",
    help="path to a normal single-split experiment folder"
    )
    parser.add_argument(
    "--base-dir",
    type=str,
    default="",
    help="directory containing base-class seed result folders"
    )
    parser.add_argument(
    "--novel-dir",
    type=str,
    default="",
    help="directory containing novel-class seed result folders"
    )
    parser.add_argument("--ci95",      action="store_true", help="use 95% CI instead of std")
    parser.add_argument("--test-log",  action="store_true", help="parse evaluation-only logs (=> Total samples)")
    parser.add_argument("--multi-exp", action="store_true", help="treat each subfolder as separate experiment")
    args = parser.parse_args()
    if (
        not args.directory
        and not (args.base_dir and args.novel_dir)
    ):
        parser.error(
            "Provide either a normal experiment directory, "
            "or both --base-dir and --novel-dir"
        )

    # for train+test runs we still look for "Finish training"
    end_signal = "Finish training"
    main(args, end_signal)
