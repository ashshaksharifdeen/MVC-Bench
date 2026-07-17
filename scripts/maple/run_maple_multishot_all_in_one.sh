#!/usr/bin/env bash
set -Eeuo pipefail

# Self-contained sequential multi-shot MaPLe base-to-novel runner.
# Training, evaluation, aggregation, reporting, and checkpoint cleanup are all
# implemented in this single Bash file. The aggregation logic is embedded Python.
#
# For each shot in 1, 2, 4, 8, 16, this script performs:
#   1. Train the base-class model for every dataset and seed.
#   2. Evaluate the same checkpoint on base and novel classes.
#   3. Strictly aggregate all three seeds for that shot.
#   4. Delete only that shot's model checkpoint directories.
#   5. Continue to the next shot.
#
# Evaluation logs and CSV/Markdown reports are retained, so the combined report
# can be regenerated after checkpoints have been removed.
#
# The root cumulative report is refreshed after every successfully completed
# shot. Therefore, after shot 1 it contains shot 1; after shot 2 it contains
# shots 1 and 2; and so on.
#
# Usage:
#   bash run_maple_multishot_all_in_one.sh [GPU_ID] [all|train|eval|report]
#
# Recommended:
#   bash run_maple_multishot_all_in_one.sh 0 all
#
# Useful overrides:
#   CLEANUP_CHECKPOINTS=0  # retain checkpoints
#   FORCE_RERUN=1          # ignore completion markers and rerun
#   PLOT_ANGDIST=True      # enable optional plots

GPU_ID="${1:-${GPU_ID:-0}}"
MODE="${2:-all}"

echo "=================================================================="
echo "Resume status:"
for resume_shot in "${SHOTS[@]}"; do
    if [[ -f "$STATUS_ROOT/shot_${resume_shot}.complete" ]]; then
        echo "  shot $resume_shot : complete"
    elif [[ -f "$STATUS_ROOT/shot_${resume_shot}.aggregation_complete" ]]; then
        echo "  shot $resume_shot : aggregation complete; cleanup pending"
    else
        echo "  shot $resume_shot : incomplete or not started"
    fi
done
echo "=================================================================="

case "$MODE" in
    all|train|eval|report) ;;
    *)
        echo "ERROR: mode must be one of: all, train, eval, report" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/storagepool/Ashshak/output3/base2new}"
REQUESTED_PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-maple}"
AUTO_CONDA_FALLBACK="${AUTO_CONDA_FALLBACK:-1}"
PYTHON_BIN=""
TRAIN_PY="${TRAIN_PY:-$PROJECT_ROOT/train.py}"

TRAINER="${TRAINER:-MaPLe}"
CFG="${CFG:-vit_b16_c2_ep5_batch4_2ctx}"
LOAD_EPOCH="${LOAD_EPOCH:-5}"
MODEL_COMPONENT="${MODEL_COMPONENT:-MultiModalPromptLearner}"

FORCE_RERUN="${FORCE_RERUN:-0}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-1}"
PLOT_ANGDIST="${PLOT_ANGDIST:-False}"

SHOTS=(1 2 4 8 16)
SEEDS=(1 2 3)
DATASETS=(
    caltech101
    food101
    dtd
    ucf101
    oxford_flowers
    oxford_pets
    fgvc_aircraft
    stanford_cars
    eurosat
)
SPLITS=(base new)

REPORT_ROOT="$OUTPUT_ROOT/reports/$TRAINER/$CFG"
STATUS_ROOT="$REPORT_ROOT/shot_status"
mkdir -p "$REPORT_ROOT" "$STATUS_ROOT"

# Current execution context, used by the error trap.
CURRENT_SHOT="-"
CURRENT_DATASET="-"
CURRENT_SEED="-"
CURRENT_STAGE="initialization"

on_error() {
    local exit_code=$?
    local failed_command="${BASH_COMMAND:-unknown}"

    echo >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "EXPERIMENT STOPPED WITH AN ERROR" >&2
    echo "Exit code : $exit_code" >&2
    echo "Stage     : $CURRENT_STAGE" >&2
    echo "Shot      : $CURRENT_SHOT" >&2
    echo "Dataset   : $CURRENT_DATASET" >&2
    echo "Seed      : $CURRENT_SEED" >&2
    echo "Python    : ${PYTHON_REALPATH:-${PYTHON_BIN:-unresolved}}" >&2
    echo "Conda env : ${CONDA_ENV:-not set}" >&2
    echo "Command   : $failed_command" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "Completed shots and their reports have been preserved." >&2
    echo "Fix the reported error, then run the same command again to resume." >&2

    exit "$exit_code"
}

trap on_error ERR

if [[ "$MODE" != "report" && ! -f "$TRAIN_PY" ]]; then
    echo "ERROR: train.py not found at: $TRAIN_PY" >&2
    echo "Set PROJECT_ROOT or TRAIN_PY correctly." >&2
    exit 2
fi

cd "$PROJECT_ROOT"

python_has_torch() {
    local candidate="$1"

    [[ -n "$candidate" ]] || return 1
    [[ -x "$candidate" ]] || return 1

    "$candidate" -c \
        'import torch, torchvision; print(torch.__version__)' \
        >/dev/null 2>&1
}

resolve_python_environment() {
    local candidate=""
    local current_python=""
    local current_python3=""
    local conda_executable=""
    local conda_python=""

    if [[ -n "$REQUESTED_PYTHON_BIN" ]]; then
        if [[ "$REQUESTED_PYTHON_BIN" == */* ]]; then
            candidate="$REQUESTED_PYTHON_BIN"
        else
            candidate="$(command -v "$REQUESTED_PYTHON_BIN" 2>/dev/null || true)"
        fi

        if python_has_torch "$candidate"; then
            PYTHON_BIN="$candidate"
            return 0
        fi

        echo "WARNING: requested PYTHON_BIN cannot import torch:" >&2
        echo "         ${candidate:-$REQUESTED_PYTHON_BIN}" >&2
    fi

    current_python="$(command -v python 2>/dev/null || true)"
    if python_has_torch "$current_python"; then
        PYTHON_BIN="$current_python"
        return 0
    fi

    current_python3="$(command -v python3 2>/dev/null || true)"
    if [[ "$current_python3" != "$current_python" ]] &&
       python_has_torch "$current_python3"; then
        PYTHON_BIN="$current_python3"
        return 0
    fi

    if [[ "$AUTO_CONDA_FALLBACK" == "1" ]]; then
        if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
            conda_executable="$CONDA_EXE"
        else
            conda_executable="$(command -v conda 2>/dev/null || true)"
        fi

        if [[ -n "$conda_executable" ]]; then
            conda_python="$(
                "$conda_executable" run -n "$CONDA_ENV" \
                    python -c 'import sys; print(sys.executable)' \
                    2>/dev/null | tail -n 1
            )"

            if python_has_torch "$conda_python"; then
                PYTHON_BIN="$conda_python"
                echo "[ENV] Current Python has no PyTorch."
                echo "[ENV] Using Conda environment '$CONDA_ENV': $PYTHON_BIN"
                return 0
            fi
        fi
    fi

    echo "ERROR: no usable Python interpreter with PyTorch was found." >&2
    echo "Current python          : ${current_python:-not found}" >&2
    echo "Requested PYTHON_BIN    : ${REQUESTED_PYTHON_BIN:-not set}" >&2
    echo "Conda fallback env      : $CONDA_ENV" >&2
    echo >&2
    echo "Activate the correct environment:" >&2
    echo "  conda activate $CONDA_ENV" >&2
    echo "  bash scripts/maple/run_maple_multishot_all_in_one.sh 0 all" >&2
    echo >&2
    echo "Or provide the interpreter directly:" >&2
    echo "  PYTHON_BIN=/path/to/env/bin/python bash scripts/maple/run_maple_multishot_all_in_one.sh 0 all" >&2
    exit 2
}

resolve_python_environment

PYTHON_REALPATH="$(
    "$PYTHON_BIN" -c 'import os, sys; print(os.path.realpath(sys.executable))'
)"
TORCH_VERSION="$(
    "$PYTHON_BIN" -c 'import torch; print(torch.__version__)'
)"
TORCH_CUDA_AVAILABLE="$(
    "$PYTHON_BIN" -c 'import torch; print(torch.cuda.is_available())'
)"

if ! "$PYTHON_BIN" - <<'PY_ENV_CHECK'
import torch
import torchvision
import yaml
print("Environment preflight passed")
PY_ENV_CHECK
then
    echo "ERROR: Python environment preflight failed: $PYTHON_BIN" >&2
    exit 2
fi

echo "=================================================================="
echo "MaPLe multi-shot experiment"
echo "Mode       : $MODE"
echo "GPU        : $GPU_ID"
echo "Python     : $PYTHON_REALPATH"
echo "Torch      : $TORCH_VERSION"
echo "Torch CUDA : $TORCH_CUDA_AVAILABLE"
echo "Conda env  : $CONDA_ENV"
echo "Shots      : ${SHOTS[*]}"
echo "Seeds      : ${SEEDS[*]}"
echo "Datasets   : ${DATASETS[*]}"
echo "Output root: $OUTPUT_ROOT"
echo "Report root: $REPORT_ROOT"
echo "=================================================================="

log_has_accuracy() {
    local log_path="$1"

    [[ -f "$log_path" ]] || return 1

    "$PYTHON_BIN" - "$log_path" <<'PY_METRIC_CHECK'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
text = re.sub(r"\x1b\[[0-9;]*m", "", text)

patterns = [
    r"(?i)(?:\*\s*)?(?:test[\s_/-]+)?accuracy\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?",
    r"(?i)(?:top[\s_-]*1|acc(?:uracy)?[\s_-]*1|acc@1)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%?",
]

found = any(re.search(pattern, text) for pattern in patterns)
raise SystemExit(0 if found else 1)
PY_METRIC_CHECK
}

run_training_job() {
    local dataset="$1"
    local shot="$2"
    local seed="$3"

    local run_dir="$OUTPUT_ROOT/train_base/$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}"
    local checkpoint="$run_dir/$MODEL_COMPONENT/model.pth.tar-${LOAD_EPOCH}"
    local done_marker="$run_dir/.train_done_epoch_${LOAD_EPOCH}"

    if [[ "$FORCE_RERUN" != "1" && -f "$done_marker" && -f "$checkpoint" ]]; then
        echo "[SKIP][TRAIN] dataset=$dataset shot=$shot seed=$seed"
        return 0
    fi

    if [[ "$FORCE_RERUN" != "1" && -f "$checkpoint" ]]; then
        touch "$done_marker"
        echo "[SKIP][TRAIN] Final checkpoint already exists: $checkpoint"
        return 0
    fi

    mkdir -p "$run_dir"
    rm -f "$done_marker"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="$dataset"
    CURRENT_SEED="$seed"
    CURRENT_STAGE="training"

    echo "=================================================================="
    echo "[TRAIN] dataset=$dataset shot=$shot seed=$seed GPU=$GPU_ID"
    echo "Output: $run_dir"
    echo "=================================================================="

    "$PYTHON_BIN" "$TRAIN_PY" \
        --root "$DATA_ROOT" \
        --seed "$seed" \
        --trainer "$TRAINER" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "$run_dir" \
        DATASET.NUM_SHOTS "$shot" \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.MAPLE.PLOT_ANGDIST "$PLOT_ANGDIST"

    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: training completed but the expected checkpoint is missing:" >&2
        echo "       $checkpoint" >&2
        echo "Check LOAD_EPOCH=$LOAD_EPOCH and MAX_EPOCH in the YAML config." >&2
        exit 1
    fi

    touch "$done_marker"
}

run_evaluation_job() {
    local dataset="$1"
    local shot="$2"
    local seed="$3"
    local split="$4"

    local common_dir="$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}"
    local model_dir="$OUTPUT_ROOT/train_base/$common_dir"
    local checkpoint="$model_dir/$MODEL_COMPONENT/model.pth.tar-${LOAD_EPOCH}"
    local eval_dir="$OUTPUT_ROOT/test_${split}/$common_dir"
    local done_marker="$eval_dir/.eval_done_epoch_${LOAD_EPOCH}"

    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: cannot evaluate because checkpoint is missing:" >&2
        echo "       $checkpoint" >&2
        exit 1
    fi

    local eval_log="$eval_dir/log.txt"
    local train_log="$model_dir/log.txt"

    # Dassl's original base-to-novel workflow normally records the base-class
    # test result in train_base/.../log.txt after training. Reuse it when valid.
    if [[ "$split" == "base" && "$FORCE_RERUN" != "1" ]] &&        log_has_accuracy "$train_log"; then
        echo "[SKIP][EVAL-base] valid base metric found in training log: $train_log"
        return 0
    fi

    # Do not trust only a marker or the existence of log.txt. The log must
    # contain a parsable accuracy metric; otherwise rerun the evaluation.
    if [[ "$FORCE_RERUN" != "1" && -f "$done_marker" ]] &&        log_has_accuracy "$eval_log"; then
        echo "[SKIP][EVAL-$split] valid metric found: $eval_log"
        return 0
    fi

    if [[ -f "$done_marker" ]] && ! log_has_accuracy "$eval_log"; then
        echo "[STALE][EVAL-$split] marker/log exists but accuracy is missing; rerunning."
    fi

    mkdir -p "$eval_dir"
    rm -f "$done_marker"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="$dataset"
    CURRENT_SEED="$seed"
    CURRENT_STAGE="evaluation-$split"

    echo "=================================================================="
    echo "[EVAL-$split] dataset=$dataset shot=$shot seed=$seed GPU=$GPU_ID"
    echo "Model : $model_dir"
    echo "Output: $eval_dir"
    echo "=================================================================="

    "$PYTHON_BIN" "$TRAIN_PY" \
        --root "$DATA_ROOT" \
        --seed "$seed" \
        --trainer "$TRAINER" \
        --dataset-config-file "configs/datasets/${dataset}.yaml" \
        --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "$eval_dir" \
        --model-dir "$model_dir" \
        --load-epoch "$LOAD_EPOCH" \
        --eval-only \
        DATASET.NUM_SHOTS "$shot" \
        DATASET.SUBSAMPLE_CLASSES "$split" \
        TRAINER.MAPLE.PLOT_ANGDIST "$PLOT_ANGDIST" \
        TRAINER.MAPLE.ANGDIST_MAX_BATCHES 50 \
        TRAINER.MAPLE.ANGDIST_MAX_CLASSES 0

    if ! log_has_accuracy "$eval_log"; then
        echo "ERROR: evaluation finished but no parsable accuracy metric was found:" >&2
        echo "       $eval_log" >&2
        echo "Inspect the final evaluation output before rerunning." >&2
        exit 1
    fi

    touch "$done_marker"
}

embedded_aggregate() {
    local report_dir="$1"
    shift
    local selected_shots=("$@")

    # The Python program is supplied through stdin, so no separate .py file is
    # required. All values after "-" are normal command-line arguments.
    "$PYTHON_BIN" - \
        --output-root "$OUTPUT_ROOT" \
        --trainer "$TRAINER" \
        --config "$CFG" \
        --shots "${selected_shots[@]}" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        --report-dir "$report_dir" \
        --strict <<'PY_AGGREGATOR'
"""Aggregate base-to-novel MaPLe accuracy over datasets, shots, and seeds.

Expected directory layout:

  Base: OUTPUT_ROOT/test_base/.../log.txt OR OUTPUT_ROOT/train_base/.../log.txt
  Novel: OUTPUT_ROOT/test_new/.../log.txt

The script writes:
  1. per_seed_results.csv
  2. per_dataset_shot_summary.csv
  3. shot_seed_benchmark_averages.csv
  4. shot_overall_summary.csv
  5. base2new_multishot_report.md
  6. missing_or_invalid_runs.csv (only when something is missing/invalid)
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class RunResult:
    dataset: str
    shot: int
    seed: int
    base_accuracy: Optional[float]
    novel_accuracy: Optional[float]
    harmonic_mean: Optional[float]
    base_log: str
    novel_log: str


def mean(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return statistics.fmean(vals) if vals else None


def sample_std(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    if len(vals) == 1:
        return 0.0
    return statistics.stdev(vals)


def harmonic(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a + b == 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def fmt(value: Optional[float], digits: int = 2) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def fmt_pm(mu: Optional[float], sigma: Optional[float]) -> str:
    if mu is None:
        return "N/A"
    if sigma is None:
        return f"{mu:.2f}"
    return f"{mu:.2f} ± {sigma:.2f}"


def extract_metric(log_path: Path, metric: str = "accuracy") -> Optional[float]:
    """Extract the last accuracy-like metric from a Dassl-style log."""
    if not log_path.is_file():
        return None

    text = ANSI_RE.sub(
        "",
        log_path.read_text(encoding="utf-8", errors="replace"),
    )

    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    patterns = [
        re.compile(
            rf"(?i)(?:\*\s*)?(?:test[\s_/-]+)?{re.escape(metric)}"
            rf"\s*[:=]\s*{number}\s*(%)?"
        ),
        re.compile(
            rf"(?i)(?:top[\s_-]*1|acc(?:uracy)?[\s_-]*1|acc@1)"
            rf"\s*[:=]\s*{number}\s*(%)?"
        ),
    ]

    matches: list[tuple[str, str]] = []
    for pattern in patterns:
        matches.extend(pattern.findall(text))

    if not matches:
        return None

    raw_value, percent_marker = matches[-1]
    value = float(raw_value)

    if not percent_marker and 0.0 <= value <= 1.0:
        value *= 100.0

    return value


def first_valid_metric(
    candidates: Sequence[Path],
    metric: str,
) -> tuple[Optional[float], Optional[Path]]:
    for candidate in candidates:
        value = extract_metric(candidate, metric)
        if value is not None:
            return value, candidate
    return None, None


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trainer", default="MaPLe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error when any expected base/new log or metric is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[RunResult] = []
    problems: list[dict] = []

    for shot in args.shots:
        for dataset in args.datasets:
            for seed in args.seeds:
                common = (
                    Path(dataset)
                    / f"shots_{shot}"
                    / args.trainer
                    / args.config
                    / f"seed{seed}"
                )
                # Base accuracy can come from an explicit test_base run or,
                # as in the original Dassl scripts, from train_base/log.txt.
                base_candidates = [
                    args.output_root / "test_base" / common / "log.txt",
                    args.output_root / "train_base" / common / "log.txt",
                ]
                novel_candidates = [
                    args.output_root / "test_new" / common / "log.txt",
                ]

                base_acc, selected_base_log = first_valid_metric(
                    base_candidates,
                    args.metric,
                )
                novel_acc, selected_novel_log = first_valid_metric(
                    novel_candidates,
                    args.metric,
                )

                if base_acc is None:
                    problems.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "split": "base",
                            "log_path": " | ".join(map(str, base_candidates)),
                            "problem": "metric not found in test_base or train_base log",
                        }
                    )
                if novel_acc is None:
                    problems.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "split": "new",
                            "log_path": " | ".join(map(str, novel_candidates)),
                            "problem": "metric not found in test_new log",
                        }
                    )

                run_results.append(
                    RunResult(
                        dataset=dataset,
                        shot=shot,
                        seed=seed,
                        base_accuracy=base_acc,
                        novel_accuracy=novel_acc,
                        harmonic_mean=harmonic(base_acc, novel_acc),
                        base_log=(
                            str(selected_base_log)
                            if selected_base_log is not None
                            else " | ".join(map(str, base_candidates))
                        ),
                        novel_log=(
                            str(selected_novel_log)
                            if selected_novel_log is not None
                            else " | ".join(map(str, novel_candidates))
                        ),
                    )
                )

    per_seed_rows = [
        {
            "dataset": r.dataset,
            "shot": r.shot,
            "seed": r.seed,
            "base_accuracy": fmt(r.base_accuracy, 6),
            "novel_accuracy": fmt(r.novel_accuracy, 6),
            "harmonic_mean": fmt(r.harmonic_mean, 6),
            "base_log": r.base_log,
            "novel_log": r.novel_log,
        }
        for r in run_results
    ]
    write_csv(
        args.report_dir / "per_seed_results.csv",
        [
            "dataset",
            "shot",
            "seed",
            "base_accuracy",
            "novel_accuracy",
            "harmonic_mean",
            "base_log",
            "novel_log",
        ],
        per_seed_rows,
    )

    dataset_summary_rows: list[dict] = []
    dataset_summary_lookup: dict[tuple[int, str], dict] = {}

    for shot in args.shots:
        for dataset in args.datasets:
            selected = [r for r in run_results if r.shot == shot and r.dataset == dataset]
            base_vals = [r.base_accuracy for r in selected if r.base_accuracy is not None]
            novel_vals = [r.novel_accuracy for r in selected if r.novel_accuracy is not None]
            h_vals = [r.harmonic_mean for r in selected if r.harmonic_mean is not None]

            base_mu, base_sd = mean(base_vals), sample_std(base_vals)
            novel_mu, novel_sd = mean(novel_vals), sample_std(novel_vals)
            h_seed_mu, h_seed_sd = mean(h_vals), sample_std(h_vals)
            h_of_means = harmonic(base_mu, novel_mu)

            row = {
                "dataset": dataset,
                "shot": shot,
                "n_expected_seeds": len(args.seeds),
                "n_base_seeds": len(base_vals),
                "n_novel_seeds": len(novel_vals),
                "n_paired_seeds": len(h_vals),
                "base_mean": fmt(base_mu, 6),
                "base_std": fmt(base_sd, 6),
                "novel_mean": fmt(novel_mu, 6),
                "novel_std": fmt(novel_sd, 6),
                "harmonic_mean_of_seed_pairs": fmt(h_seed_mu, 6),
                "harmonic_std_of_seed_pairs": fmt(h_seed_sd, 6),
                "harmonic_of_base_novel_means": fmt(h_of_means, 6),
            }
            dataset_summary_rows.append(row)
            dataset_summary_lookup[(shot, dataset)] = {
                "base_mean": base_mu,
                "base_std": base_sd,
                "novel_mean": novel_mu,
                "novel_std": novel_sd,
                "h_seed_mean": h_seed_mu,
                "h_seed_std": h_seed_sd,
                "h_of_means": h_of_means,
                "n_paired": len(h_vals),
            }

    write_csv(
        args.report_dir / "per_dataset_shot_summary.csv",
        [
            "dataset",
            "shot",
            "n_expected_seeds",
            "n_base_seeds",
            "n_novel_seeds",
            "n_paired_seeds",
            "base_mean",
            "base_std",
            "novel_mean",
            "novel_std",
            "harmonic_mean_of_seed_pairs",
            "harmonic_std_of_seed_pairs",
            "harmonic_of_base_novel_means",
        ],
        dataset_summary_rows,
    )

    # Benchmark average for each individual seed: average across datasets first.
    # This yields a meaningful three-seed standard deviation for the overall row.
    shot_seed_rows: list[dict] = []
    shot_seed_numeric: dict[tuple[int, int], tuple[Optional[float], Optional[float], Optional[float], int]] = {}

    for shot in args.shots:
        for seed in args.seeds:
            selected = [r for r in run_results if r.shot == shot and r.seed == seed]
            paired = [r for r in selected if r.base_accuracy is not None and r.novel_accuracy is not None]
            base_avg = mean(r.base_accuracy for r in paired if r.base_accuracy is not None)
            novel_avg = mean(r.novel_accuracy for r in paired if r.novel_accuracy is not None)
            h_avg = harmonic(base_avg, novel_avg)
            shot_seed_numeric[(shot, seed)] = (base_avg, novel_avg, h_avg, len(paired))
            shot_seed_rows.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "n_complete_datasets": len(paired),
                    "n_expected_datasets": len(args.datasets),
                    "base_dataset_average": fmt(base_avg, 6),
                    "novel_dataset_average": fmt(novel_avg, 6),
                    "harmonic_of_dataset_averages": fmt(h_avg, 6),
                }
            )

    write_csv(
        args.report_dir / "shot_seed_benchmark_averages.csv",
        [
            "shot",
            "seed",
            "n_complete_datasets",
            "n_expected_datasets",
            "base_dataset_average",
            "novel_dataset_average",
            "harmonic_of_dataset_averages",
        ],
        shot_seed_rows,
    )

    shot_summary_rows: list[dict] = []
    shot_summary_numeric: dict[int, dict] = {}

    for shot in args.shots:
        seed_values = [shot_seed_numeric[(shot, seed)] for seed in args.seeds]
        complete_seed_values = [v for v in seed_values if v[3] == len(args.datasets)]

        # In non-strict mode, still summarize available data for each seed.
        usable = complete_seed_values or [v for v in seed_values if v[0] is not None and v[1] is not None]
        base_seed_avgs = [v[0] for v in usable if v[0] is not None]
        novel_seed_avgs = [v[1] for v in usable if v[1] is not None]
        h_seed_avgs = [v[2] for v in usable if v[2] is not None]

        base_mu, base_sd = mean(base_seed_avgs), sample_std(base_seed_avgs)
        novel_mu, novel_sd = mean(novel_seed_avgs), sample_std(novel_seed_avgs)
        h_mu, h_sd = mean(h_seed_avgs), sample_std(h_seed_avgs)
        h_of_means = harmonic(base_mu, novel_mu)

        row = {
            "shot": shot,
            "n_expected_seeds": len(args.seeds),
            "n_complete_seeds": len(complete_seed_values),
            "n_expected_datasets_per_seed": len(args.datasets),
            "base_mean": fmt(base_mu, 6),
            "base_std_across_seeds": fmt(base_sd, 6),
            "novel_mean": fmt(novel_mu, 6),
            "novel_std_across_seeds": fmt(novel_sd, 6),
            "harmonic_mean_across_seeds": fmt(h_mu, 6),
            "harmonic_std_across_seeds": fmt(h_sd, 6),
            "harmonic_of_overall_base_novel_means": fmt(h_of_means, 6),
        }
        shot_summary_rows.append(row)
        shot_summary_numeric[shot] = {
            "base_mean": base_mu,
            "base_std": base_sd,
            "novel_mean": novel_mu,
            "novel_std": novel_sd,
            "h_mean": h_mu,
            "h_std": h_sd,
            "h_of_means": h_of_means,
            "complete_seeds": len(complete_seed_values),
        }

    write_csv(
        args.report_dir / "shot_overall_summary.csv",
        [
            "shot",
            "n_expected_seeds",
            "n_complete_seeds",
            "n_expected_datasets_per_seed",
            "base_mean",
            "base_std_across_seeds",
            "novel_mean",
            "novel_std_across_seeds",
            "harmonic_mean_across_seeds",
            "harmonic_std_across_seeds",
            "harmonic_of_overall_base_novel_means",
        ],
        shot_summary_rows,
    )

    missing_report = args.report_dir / "missing_or_invalid_runs.csv"
    if problems:
        write_csv(
            missing_report,
            ["dataset", "shot", "seed", "split", "log_path", "problem"],
            problems,
        )
    elif missing_report.exists():
        missing_report.unlink()

    report_lines = [
        "# MaPLe Multi-shot Base-to-Novel Results",
        "",
        f"- Trainer: `{args.trainer}`",
        f"- Config: `{args.config}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Shots: `{', '.join(map(str, args.shots))}`",
        f"- Metric parsed from logs: `{args.metric}`",
        "- Values are percentages.",
        "- `H` is the harmonic mean of base and novel accuracy.",
        "- Base accuracy is read from test_base when available, otherwise train_base.",
        "",
        "## Overall benchmark average by shot",
        "",
        "The dataset average is computed separately for each seed; the table then reports mean ± sample standard deviation across seeds.",
        "",
        "| Shot | Base | Novel | H | Complete seeds |",
        "|---:|---:|---:|---:|---:|",
    ]

    for shot in args.shots:
        s = shot_summary_numeric[shot]
        report_lines.append(
            f"| {shot} | {fmt_pm(s['base_mean'], s['base_std'])} | "
            f"{fmt_pm(s['novel_mean'], s['novel_std'])} | "
            f"{fmt_pm(s['h_mean'], s['h_std'])} | "
            f"{s['complete_seeds']}/{len(args.seeds)} |"
        )

    for shot in args.shots:
        report_lines.extend(
            [
                "",
                f"## {shot}-shot per-dataset results",
                "",
                "| Dataset | Base | Novel | H (from means) | Paired seeds |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in args.datasets:
            d = dataset_summary_lookup[(shot, dataset)]
            report_lines.append(
                f"| {dataset} | {fmt_pm(d['base_mean'], d['base_std'])} | "
                f"{fmt_pm(d['novel_mean'], d['novel_std'])} | "
                f"{fmt(d['h_of_means']) or 'N/A'} | "
                f"{d['n_paired']}/{len(args.seeds)} |"
            )

    if problems:
        report_lines.extend(
            [
                "",
                "## Missing or invalid runs",
                "",
                f"Found {len(problems)} missing/invalid split results. See `missing_or_invalid_runs.csv`.",
            ]
        )

    report_path = args.report_dir / "base2new_multishot_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote reports to: {args.report_dir}")
    print(f"Main report: {report_path}")

    if problems:
        print(f"WARNING: {len(problems)} expected results were missing or invalid.", file=sys.stderr)
        print(f"See: {missing_report}", file=sys.stderr)
        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY_AGGREGATOR
}

aggregate_one_shot() {
    local shot="$1"
    local shot_report_dir="$REPORT_ROOT/shot_${shot}"
    local aggregation_marker="$STATUS_ROOT/shot_${shot}.aggregation_complete"

    mkdir -p "$shot_report_dir"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="strict-shot-aggregation"

    echo "=================================================================="
    echo "[AGGREGATE] shot=$shot across seeds: ${SEEDS[*]}"
    echo "Report: $shot_report_dir"
    echo "=================================================================="

    # Strict aggregation must succeed before cleanup is permitted.
    embedded_aggregate "$shot_report_dir" "$shot"

    touch "$aggregation_marker"
}

cleanup_one_shot_checkpoints() {
    local shot="$1"
    local deleted_count=0
    local checkpoint_dir

    if [[ "$CLEANUP_CHECKPOINTS" != "1" ]]; then
        echo "[KEEP] CLEANUP_CHECKPOINTS=$CLEANUP_CHECKPOINTS; retaining shot=$shot checkpoints."
        return 0
    fi

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="checkpoint-cleanup"

    echo "=================================================================="
    echo "[CLEANUP] Removing model checkpoints for shot=$shot"
    echo "Training logs and evaluation results will be retained."
    echo "=================================================================="

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            checkpoint_dir="$OUTPUT_ROOT/train_base/$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}/$MODEL_COMPONENT"

            if [[ -d "$checkpoint_dir" ]]; then
                # Safety guard: only remove a directory that is located below the
                # expected train_base tree and has the exact model component name.
                case "$checkpoint_dir" in
                    "$OUTPUT_ROOT"/train_base/*/shots_"$shot"/"$TRAINER"/"$CFG"/seed*/"$MODEL_COMPONENT")
                        rm -rf -- "$checkpoint_dir"
                        deleted_count=$((deleted_count + 1))
                        ;;
                    *)
                        echo "ERROR: refusing unsafe cleanup path: $checkpoint_dir" >&2
                        exit 1
                        ;;
                esac
            fi
        done
    done

    echo "[CLEANUP] Deleted $deleted_count checkpoint directorie(s) for shot=$shot."
    touch "$STATUS_ROOT/shot_${shot}.checkpoints_deleted"
}

refresh_cumulative_report() {
    local completed_shots=()
    local candidate_shot

    for candidate_shot in "${SHOTS[@]}"; do
        if [[ -f "$STATUS_ROOT/shot_${candidate_shot}.complete" ]]; then
            completed_shots+=("$candidate_shot")
        fi
    done

    if (( ${#completed_shots[@]} == 0 )); then
        echo "[CUMULATIVE] No completed shots are available yet."
        return 0
    fi

    CURRENT_SHOT="${completed_shots[*]}"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="cumulative-report"

    echo "=================================================================="
    echo "[CUMULATIVE REPORT] completed shots: ${completed_shots[*]}"
    echo "Report: $REPORT_ROOT"
    echo "=================================================================="

    embedded_aggregate "$REPORT_ROOT" "${completed_shots[@]}"

    printf '%s\n' "${completed_shots[@]}" > "$REPORT_ROOT/completed_shots.txt"
}

process_shot_all() {
    local shot="$1"
    local shot_complete_marker="$STATUS_ROOT/shot_${shot}.complete"
    local aggregation_marker="$STATUS_ROOT/shot_${shot}.aggregation_complete"

    if [[ "$FORCE_RERUN" != "1" && -f "$shot_complete_marker" ]]; then
        echo "[SKIP][SHOT] shot=$shot was already evaluated, aggregated, and cleaned."

        # Revalidate retained evaluation logs before trusting the marker.
        aggregate_one_shot "$shot"
        refresh_cumulative_report
        return 0
    fi

    # Recovery path: aggregation already succeeded in a previous invocation, but
    # cleanup or final marker creation was interrupted. Revalidate the logs,
    # finish cleanup idempotently, and do not retrain the models.
    if [[ "$FORCE_RERUN" != "1" && -f "$aggregation_marker" ]]; then
        echo "[RECOVER][SHOT] shot=$shot already has a strict aggregate report."
        aggregate_one_shot "$shot"
        cleanup_one_shot_checkpoints "$shot"
        touch "$shot_complete_marker"
        refresh_cumulative_report
        return 0
    fi

    echo "##################################################################"
    echo "# STARTING SHOT: $shot"
    echo "##################################################################"

    # Train and immediately evaluate each individual checkpoint. The checkpoint
    # remains available until strict aggregation of the complete shot succeeds.
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_training_job "$dataset" "$shot" "$seed"
            run_evaluation_job "$dataset" "$shot" "$seed" base
            run_evaluation_job "$dataset" "$shot" "$seed" new
        done
    done

    aggregate_one_shot "$shot"
    cleanup_one_shot_checkpoints "$shot"
    touch "$shot_complete_marker"
    refresh_cumulative_report

    echo "##################################################################"
    echo "# COMPLETED SHOT: $shot"
    echo "# The next shot will start now."
    echo "##################################################################"
}

process_shot_train_only() {
    local shot="$1"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_training_job "$dataset" "$shot" "$seed"
        done
    done
}

process_shot_eval_only() {
    local shot="$1"
    local shot_complete_marker="$STATUS_ROOT/shot_${shot}.complete"

    if [[ "$FORCE_RERUN" != "1" && -f "$shot_complete_marker" ]]; then
        echo "[SKIP][SHOT] shot=$shot was already evaluated, aggregated, and cleaned."
        return 0
    fi

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_evaluation_job "$dataset" "$shot" "$seed" base
            run_evaluation_job "$dataset" "$shot" "$seed" new
        done
    done

    aggregate_one_shot "$shot"
    cleanup_one_shot_checkpoints "$shot"
    touch "$shot_complete_marker"
    refresh_cumulative_report
}

aggregate_all_shots() {
    CURRENT_SHOT="${SHOTS[*]}"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="final-all-shot-aggregation"

    echo "=================================================================="
    echo "[FINAL AGGREGATE] shots: ${SHOTS[*]}"
    echo "Report: $REPORT_ROOT"
    echo "=================================================================="

    embedded_aggregate "$REPORT_ROOT" "${SHOTS[@]}"
}

case "$MODE" in
    all)
        for shot in "${SHOTS[@]}"; do
            process_shot_all "$shot"
        done
        aggregate_all_shots
        ;;

    train)
        echo "WARNING: train mode does not evaluate or delete checkpoints." >&2
        echo "Use mode=all for the sequential space-saving workflow." >&2
        for shot in "${SHOTS[@]}"; do
            process_shot_train_only "$shot"
        done
        ;;

    eval)
        for shot in "${SHOTS[@]}"; do
            process_shot_eval_only "$shot"
        done
        aggregate_all_shots
        ;;

    report)
        aggregate_all_shots
        ;;
esac

printf '\nCompleted mode=%s on GPU=%s\n' "$MODE" "$GPU_ID"
printf 'Python interpreter       : %s\n' "$PYTHON_REALPATH"
printf 'Combined report directory: %s\n' "$REPORT_ROOT"
