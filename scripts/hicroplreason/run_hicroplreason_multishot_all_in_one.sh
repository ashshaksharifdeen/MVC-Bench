#!/usr/bin/env bash
set -Eeuo pipefail

# ================================================================
# HiCroPLReason sequential multi-shot base-to-novel experiment runner
# ================================================================
#
# For each shot in 1, 2, 4, 8, 16:
#   1. Train base classes for all datasets and seeds.
#   2. Reuse the base result from train_base/log.txt when available.
#   3. Evaluate novel classes.
#   4. Validate every accuracy value.
#   5. Aggregate all three seeds.
#   6. Generate per-shot and cumulative reports.
#   7. Delete only the VLPromptLearner checkpoint directory.
#   8. Continue to the next shot.
#
# Usage:
#   bash scripts/hicroplreason/run_hicroplreason_multishot_all_in_one.sh \
#       [GPU_ID] [all|train|eval|report]
#
# Recommended:
#   bash scripts/hicroplreason/run_hicroplreason_multishot_all_in_one.sh 0 all
#
# Optional environment overrides:
#
#   CONDA_ENV=maple
#   PYTHON_BIN=/path/to/python
#   CLEANUP_CHECKPOINTS=0
#   FORCE_RERUN=1
#   DATA_ROOT=/path/to/datasets
#   OUTPUT_ROOT=/path/to/output
#
# ================================================================


# ----------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------

GPU_ID="${1:-${GPU_ID:-0}}"
MODE="${2:-all}"

case "$MODE" in
    all|train|eval|report)
        ;;
    *)
        echo "ERROR: unsupported mode: $MODE" >&2
        echo "Allowed modes: all, train, eval, report" >&2
        exit 2
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"


# ----------------------------------------------------------------
# Paths and experiment configuration
# ----------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

DATA_ROOT="${DATA_ROOT:-/l/users/ashshak.sharifdeen/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/l/users/ashshak.sharifdeen/output2/base2new}"

TRAIN_PY="${TRAIN_PY:-$PROJECT_ROOT/train.py}"

TRAINER="${TRAINER:-HiCroPLReason}"
CFG="${CFG:-vit_b16_c2_ep50_batch32_16ctx}"
LOAD_EPOCH="${LOAD_EPOCH:-50}"

# HiCroPLReason registers the model as VLPromptLearner.
# Expected checkpoint: VLPromptLearner/model.pth.tar-<epoch>
MODEL_COMPONENT="${MODEL_COMPONENT:-VLPromptLearner}"

FORCE_RERUN="${FORCE_RERUN:-0}"
CLEANUP_CHECKPOINTS="${CLEANUP_CHECKPOINTS:-1}"

# Python environment configuration.
REQUESTED_PYTHON_BIN="${PYTHON_BIN:-}"
CONDA_ENV="${CONDA_ENV:-maple}"
AUTO_CONDA_FALLBACK="${AUTO_CONDA_FALLBACK:-1}"

PYTHON_BIN=""


# ----------------------------------------------------------------
# Shots, seeds and datasets
# ----------------------------------------------------------------

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
    sun397
    eurosat
)

# Dataset-specific probe weights from the HiCroPLReason training script.
declare -A PROBE_LAMBDA_BY_DATASET=(
    [caltech101]=5.0
    [food101]=12.0
    [dtd]=5.0
    [ucf101]=0.5
    [oxford_flowers]=3.0
    [oxford_pets]=12.0
    [fgvc_aircraft]=5.0
    [stanford_cars]=4.0
    [sun397]=4.0
    [eurosat]=4.0
)

# Evaluation-time overrides from the HiCroPLReason test script.
PROBE_ENABLE_EVAL="${PROBE_ENABLE_EVAL:-False}"
DAPT_SAVE_PROTOTYPES_EVAL="${DAPT_SAVE_PROTOTYPES_EVAL:-False}"
DAPT_INTRA_ENABLE_EVAL="${DAPT_INTRA_ENABLE_EVAL:-False}"


# ----------------------------------------------------------------
# Report and status paths
# ----------------------------------------------------------------

REPORT_ROOT="$OUTPUT_ROOT/reports/$TRAINER/$CFG"
STATUS_ROOT="$REPORT_ROOT/shot_status"

mkdir -p "$REPORT_ROOT"
mkdir -p "$STATUS_ROOT"


# ----------------------------------------------------------------
# Current execution state for error reporting
# ----------------------------------------------------------------

CURRENT_SHOT="-"
CURRENT_DATASET="-"
CURRENT_SEED="-"
CURRENT_STAGE="initialization"

PYTHON_REALPATH=""
TORCH_VERSION=""
TORCH_CUDA_AVAILABLE=""


# ----------------------------------------------------------------
# Error handler
# ----------------------------------------------------------------

on_error() {
    local exit_code=$?
    local failed_command="${BASH_COMMAND:-unknown}"

    echo >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "HICROPLREASON EXPERIMENT STOPPED WITH AN ERROR" >&2
    echo "Exit code : $exit_code" >&2
    echo "Stage     : $CURRENT_STAGE" >&2
    echo "Shot      : $CURRENT_SHOT" >&2
    echo "Dataset   : $CURRENT_DATASET" >&2
    echo "Seed      : $CURRENT_SEED" >&2
    echo "Python    : ${PYTHON_REALPATH:-${PYTHON_BIN:-unresolved}}" >&2
    echo "Conda env : ${CONDA_ENV:-not set}" >&2
    echo "Command   : $failed_command" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "Completed shots and generated reports were preserved." >&2
    echo "Run the same command again after correcting the error." >&2

    exit "$exit_code"
}

trap on_error ERR


# ----------------------------------------------------------------
# Initial file validation
# ----------------------------------------------------------------

if [[ "$MODE" != "report" && ! -f "$TRAIN_PY" ]]; then
    echo "ERROR: train.py was not found:" >&2
    echo "       $TRAIN_PY" >&2
    echo "Set PROJECT_ROOT or TRAIN_PY correctly." >&2
    exit 2
fi

HICROPLREASON_CONFIG_FILE="$PROJECT_ROOT/configs/trainers/$TRAINER/$CFG.yaml"

if [[ "$MODE" != "report" && ! -f "$HICROPLREASON_CONFIG_FILE" ]]; then
    echo "ERROR: HiCroPLReason configuration file was not found:" >&2
    echo "       $HICROPLREASON_CONFIG_FILE" >&2
    exit 2
fi

cd "$PROJECT_ROOT"


# ----------------------------------------------------------------
# Python environment resolution
# ----------------------------------------------------------------

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

    # Explicit PYTHON_BIN has highest priority.
    if [[ -n "$REQUESTED_PYTHON_BIN" ]]; then
        if [[ "$REQUESTED_PYTHON_BIN" == */* ]]; then
            candidate="$REQUESTED_PYTHON_BIN"
        else
            candidate="$(
                command -v "$REQUESTED_PYTHON_BIN" 2>/dev/null || true
            )"
        fi

        if python_has_torch "$candidate"; then
            PYTHON_BIN="$candidate"
            return 0
        fi

        echo "WARNING: requested PYTHON_BIN cannot import PyTorch:" >&2
        echo "         ${candidate:-$REQUESTED_PYTHON_BIN}" >&2
    fi

    # Try current python.
    current_python="$(command -v python 2>/dev/null || true)"

    if python_has_torch "$current_python"; then
        PYTHON_BIN="$current_python"
        return 0
    fi

    # Try current python3.
    current_python3="$(command -v python3 2>/dev/null || true)"

    if [[ "$current_python3" != "$current_python" ]] &&
       python_has_torch "$current_python3"; then
        PYTHON_BIN="$current_python3"
        return 0
    fi

    # Fall back to a Conda environment without requiring conda activate.
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
                    2>/dev/null |
                    tail -n 1
            )"

            if python_has_torch "$conda_python"; then
                PYTHON_BIN="$conda_python"

                echo "[ENV] Current Python cannot import PyTorch."
                echo "[ENV] Using Conda environment '$CONDA_ENV'."
                echo "[ENV] Python: $PYTHON_BIN"

                return 0
            fi
        fi
    fi

    echo "ERROR: no Python interpreter with PyTorch was found." >&2
    echo >&2
    echo "Current python       : ${current_python:-not found}" >&2
    echo "Requested PYTHON_BIN : ${REQUESTED_PYTHON_BIN:-not set}" >&2
    echo "Conda environment    : $CONDA_ENV" >&2
    echo >&2
    echo "Activate the correct environment:" >&2
    echo "  conda activate $CONDA_ENV" >&2
    echo "  bash scripts/hicroplreason/run_hicroplreason_multishot_all_in_one.sh 0 all" >&2
    echo >&2
    echo "Or provide the Python path directly:" >&2
    echo "  PYTHON_BIN=/path/to/env/bin/python \\" >&2
    echo "  bash scripts/hicroplreason/run_hicroplreason_multishot_all_in_one.sh 0 all" >&2

    exit 2
}


resolve_python_environment

PYTHON_REALPATH="$(
    "$PYTHON_BIN" -c \
        'import os, sys; print(os.path.realpath(sys.executable))'
)"

TORCH_VERSION="$(
    "$PYTHON_BIN" -c \
        'import torch; print(torch.__version__)'
)"

TORCH_CUDA_AVAILABLE="$(
    "$PYTHON_BIN" -c \
        'import torch; print(torch.cuda.is_available())'
)"


# ----------------------------------------------------------------
# Environment preflight
# ----------------------------------------------------------------

CURRENT_STAGE="environment-preflight"

if ! "$PYTHON_BIN" - <<'PY_ENV_CHECK'
import torch
import torchvision
import yaml

print("Environment preflight passed")
PY_ENV_CHECK
then
    echo "ERROR: Python environment preflight failed:" >&2
    echo "       $PYTHON_BIN" >&2
    exit 2
fi


# ----------------------------------------------------------------
# Experiment summary
# ----------------------------------------------------------------

echo "=================================================================="
echo "HiCroPLReason multi-shot base-to-novel experiment"
echo "Mode       : $MODE"
echo "GPU        : $GPU_ID"
echo "Python     : $PYTHON_REALPATH"
echo "Torch      : $TORCH_VERSION"
echo "Torch CUDA : $TORCH_CUDA_AVAILABLE"
echo "Conda env  : $CONDA_ENV"
echo "Trainer    : $TRAINER"
echo "Config     : $CFG"
echo "Config file: $HICROPLREASON_CONFIG_FILE"
echo "Epoch      : $LOAD_EPOCH"
echo "Component  : $MODEL_COMPONENT"
echo "Shots      : ${SHOTS[*]}"
echo "Seeds      : ${SEEDS[*]}"
echo "Datasets   : ${DATASETS[*]}"
echo "Data root  : $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Report root: $REPORT_ROOT"
echo "Cleanup    : $CLEANUP_CHECKPOINTS"
echo "Eval probe : $PROBE_ENABLE_EVAL"
echo "Eval save prototypes: $DAPT_SAVE_PROTOTYPES_EVAL"
echo "Eval DAPT intra     : $DAPT_INTRA_ENABLE_EVAL"
echo "Probe lambda map:"
for lambda_dataset in "${DATASETS[@]}"; do
    echo "  $lambda_dataset = ${PROBE_LAMBDA_BY_DATASET[$lambda_dataset]}"
done
echo "=================================================================="


# ----------------------------------------------------------------
# Accuracy validation helper
# ----------------------------------------------------------------

log_has_accuracy() {
    local log_path="$1"

    [[ -f "$log_path" ]] || return 1

    "$PYTHON_BIN" - "$log_path" <<'PY_METRIC_CHECK'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])

text = path.read_text(
    encoding="utf-8",
    errors="replace",
)

text = re.sub(
    r"\x1b\[[0-9;]*m",
    "",
    text,
)

number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

patterns = [
    re.compile(
        rf"(?i)(?:\*\s*)?(?:test[\s_/-]+)?accuracy"
        rf"\s*[:=]\s*{number}\s*%?"
    ),
    re.compile(
        rf"(?i)(?:top[\s_-]*1|acc(?:uracy)?[\s_-]*1|acc@1)"
        rf"\s*[:=]\s*{number}\s*%?"
    ),
]

found = any(
    pattern.search(text)
    for pattern in patterns
)

raise SystemExit(0 if found else 1)
PY_METRIC_CHECK
}


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

run_training_job() {
    local dataset="$1"
    local shot="$2"
    local seed="$3"

    local run_dir
    local checkpoint
    local done_marker
    local probe_lambda
    local lambda_file

    if [[ ! -v "PROBE_LAMBDA_BY_DATASET[$dataset]" ]]; then
        echo "ERROR: PROBE_LAMBDA is not defined for dataset: $dataset" >&2
        exit 1
    fi

    probe_lambda="${PROBE_LAMBDA_BY_DATASET[$dataset]}"

    run_dir="$OUTPUT_ROOT/train_base/$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}"

    checkpoint="$run_dir/$MODEL_COMPONENT/model.pth.tar-${LOAD_EPOCH}"

    done_marker="$run_dir/.train_done_epoch_${LOAD_EPOCH}"
    lambda_file="$run_dir/.probe_lambda"

    # If the lambda record exists, it must match the requested dataset value.
    if [[ -f "$checkpoint" && -f "$lambda_file" ]]; then
        if [[ "$(tr -d '[:space:]' < "$lambda_file")" != "$probe_lambda" ]]; then
            echo "ERROR: checkpoint PROBE_LAMBDA does not match this run:" >&2
            echo "       $checkpoint" >&2
            echo "Stored   : $(cat "$lambda_file")" >&2
            echo "Requested: $probe_lambda" >&2
            exit 1
        fi
    fi

    # Skip only when both the marker and expected checkpoint exist.
    if [[ "$FORCE_RERUN" != "1" &&
          -f "$done_marker" &&
          -f "$checkpoint" ]]; then
        echo "[SKIP][TRAIN] dataset=$dataset shot=$shot seed=$seed"
        return 0
    fi

    # If the final checkpoint exists, create the marker and continue.
    if [[ "$FORCE_RERUN" != "1" &&
          -f "$checkpoint" ]]; then
        touch "$done_marker"

        echo "[SKIP][TRAIN] final checkpoint already exists:"
        echo "              $checkpoint"

        return 0
    fi

    mkdir -p "$run_dir"
    rm -f "$done_marker"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="$dataset"
    CURRENT_SEED="$seed"
    CURRENT_STAGE="training"

    echo "=================================================================="
    echo "[TRAIN]"
    echo "Dataset : $dataset"
    echo "Shot    : $shot"
    echo "Seed    : $seed"
    echo "GPU     : $GPU_ID"
    echo "PROBE_LAMBDA: $probe_lambda"
    echo "Output  : $run_dir"
    echo "=================================================================="

    "$PYTHON_BIN" "$TRAIN_PY" \
        --root "$DATA_ROOT" \
        --seed "$seed" \
        --trainer "$TRAINER" \
        --dataset-config-file \
            "configs/datasets/${dataset}.yaml" \
        --config-file \
            "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "$run_dir" \
        DATASET.NUM_SHOTS "$shot" \
        DATASET.SUBSAMPLE_CLASSES base \
        TRAINER.HICROPLReason.PROBE_LAMBDA "$probe_lambda"

    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: training completed but the expected checkpoint is missing:" >&2
        echo "       $checkpoint" >&2
        echo >&2
        echo "Verify:" >&2
        echo "  LOAD_EPOCH=$LOAD_EPOCH" >&2
        echo "  MODEL_COMPONENT=$MODEL_COMPONENT" >&2
        echo "  MAX_EPOCH in $HICROPLREASON_CONFIG_FILE" >&2

        exit 1
    fi

    printf '%s\n' "$probe_lambda" > "$lambda_file"
    touch "$done_marker"
}


# ----------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------

run_evaluation_job() {
    local dataset="$1"
    local shot="$2"
    local seed="$3"
    local split="$4"

    local common_dir
    local model_dir
    local checkpoint
    local eval_dir
    local done_marker
    local eval_log
    local train_log

    common_dir="$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}"

    model_dir="$OUTPUT_ROOT/train_base/$common_dir"

    checkpoint="$model_dir/$MODEL_COMPONENT/model.pth.tar-${LOAD_EPOCH}"

    eval_dir="$OUTPUT_ROOT/test_${split}/$common_dir"

    done_marker="$eval_dir/.eval_done_epoch_${LOAD_EPOCH}"

    eval_log="$eval_dir/log.txt"
    train_log="$model_dir/log.txt"

    if [[ ! -f "$checkpoint" ]]; then
        echo "ERROR: cannot evaluate because the checkpoint is missing:" >&2
        echo "       $checkpoint" >&2
        exit 1
    fi

    # Base-class accuracy is normally already available in train_base/log.txt.
    if [[ "$split" == "base" &&
          "$FORCE_RERUN" != "1" ]] &&
       log_has_accuracy "$train_log"; then
        echo "[SKIP][EVAL-base] valid base metric found:"
        echo "                  $train_log"

        return 0
    fi

    # Skip a previous explicit evaluation only when its log contains accuracy.
    if [[ "$FORCE_RERUN" != "1" &&
          -f "$done_marker" ]] &&
       log_has_accuracy "$eval_log"; then
        echo "[SKIP][EVAL-$split] valid metric found:"
        echo "                  $eval_log"

        return 0
    fi

    # Remove stale markers whose log contains no parsable metric.
    if [[ -f "$done_marker" ]] &&
       ! log_has_accuracy "$eval_log"; then
        echo "[STALE][EVAL-$split] marker or log exists without valid accuracy."
        echo "                     Rerunning dataset=$dataset shot=$shot seed=$seed"
    fi

    mkdir -p "$eval_dir"
    rm -f "$done_marker"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="$dataset"
    CURRENT_SEED="$seed"
    CURRENT_STAGE="evaluation-$split"

    echo "=================================================================="
    echo "[EVAL-$split]"
    echo "Dataset : $dataset"
    echo "Shot    : $shot"
    echo "Seed    : $seed"
    echo "GPU     : $GPU_ID"
    echo "Model   : $model_dir"
    echo "Output  : $eval_dir"
    echo "=================================================================="

    "$PYTHON_BIN" "$TRAIN_PY" \
        --root "$DATA_ROOT" \
        --seed "$seed" \
        --trainer "$TRAINER" \
        --dataset-config-file \
            "configs/datasets/${dataset}.yaml" \
        --config-file \
            "configs/trainers/${TRAINER}/${CFG}.yaml" \
        --output-dir "$eval_dir" \
        --model-dir "$model_dir" \
        --load-epoch "$LOAD_EPOCH" \
        --eval-only \
        DATASET.NUM_SHOTS "$shot" \
        DATASET.SUBSAMPLE_CLASSES "$split" \
        TRAINER.HICROPLReason.PROBE_ENABLE "$PROBE_ENABLE_EVAL" \
        TRAINER.HICROPLReason.DAPT_SAVE_PROTOTYPES "$DAPT_SAVE_PROTOTYPES_EVAL" \
        TRAINER.HICROPLReason.DAPT_INTRA_ENABLE "$DAPT_INTRA_ENABLE_EVAL"

    if ! log_has_accuracy "$eval_log"; then
        echo "ERROR: evaluation completed but no parsable accuracy was found:" >&2
        echo "       $eval_log" >&2
        exit 1
    fi

    touch "$done_marker"
}


# ----------------------------------------------------------------
# Embedded result aggregation
# ----------------------------------------------------------------

embedded_aggregate() {
    local report_dir="$1"
    shift

    local selected_shots=("$@")

    "$PYTHON_BIN" - \
        --output-root "$OUTPUT_ROOT" \
        --trainer "$TRAINER" \
        --config "$CFG" \
        --shots "${selected_shots[@]}" \
        --seeds "${SEEDS[@]}" \
        --datasets "${DATASETS[@]}" \
        --report-dir "$report_dir" \
        --strict <<'PY_AGGREGATOR'
from __future__ import annotations

import argparse
import csv
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

    if not vals:
        return None

    return statistics.fmean(vals)


def sample_std(values: Iterable[float]) -> Optional[float]:
    vals = list(values)

    if not vals:
        return None

    if len(vals) == 1:
        return 0.0

    return statistics.stdev(vals)


def harmonic(
    base: Optional[float],
    novel: Optional[float],
) -> Optional[float]:
    if base is None or novel is None:
        return None

    if base + novel == 0:
        return 0.0

    return 2.0 * base * novel / (base + novel)


def fmt(
    value: Optional[float],
    digits: int = 2,
) -> str:
    if value is None:
        return ""

    return f"{value:.{digits}f}"


def fmt_pm(
    mu: Optional[float],
    sigma: Optional[float],
) -> str:
    if mu is None:
        return "N/A"

    if sigma is None:
        return f"{mu:.2f}"

    return f"{mu:.2f} ± {sigma:.2f}"


def extract_metric(
    log_path: Path,
    metric: str = "accuracy",
) -> Optional[float]:
    if not log_path.is_file():
        return None

    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    text = ANSI_RE.sub("", text)

    number = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"

    patterns = [
        re.compile(
            rf"(?i)(?:\*\s*)?(?:test[\s_/-]+)?"
            rf"{re.escape(metric)}\s*[:=]\s*"
            rf"{number}\s*(%)?"
        ),
        re.compile(
            rf"(?i)(?:top[\s_-]*1|"
            rf"acc(?:uracy)?[\s_-]*1|acc@1)"
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


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--trainer",
        default="HiCroPLReason",
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--metric",
        default="accuracy",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--strict",
        action="store_true",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    args.report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

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

                base_candidates = [
                    args.output_root
                    / "test_base"
                    / common
                    / "log.txt",

                    args.output_root
                    / "train_base"
                    / common
                    / "log.txt",
                ]

                novel_candidates = [
                    args.output_root
                    / "test_new"
                    / common
                    / "log.txt",
                ]

                base_accuracy, selected_base_log = first_valid_metric(
                    base_candidates,
                    args.metric,
                )

                novel_accuracy, selected_novel_log = first_valid_metric(
                    novel_candidates,
                    args.metric,
                )

                if base_accuracy is None:
                    problems.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "split": "base",
                            "log_path": " | ".join(
                                map(str, base_candidates)
                            ),
                            "problem": (
                                "metric not found in "
                                "test_base or train_base"
                            ),
                        }
                    )

                if novel_accuracy is None:
                    problems.append(
                        {
                            "dataset": dataset,
                            "shot": shot,
                            "seed": seed,
                            "split": "new",
                            "log_path": " | ".join(
                                map(str, novel_candidates)
                            ),
                            "problem": (
                                "metric not found in test_new"
                            ),
                        }
                    )

                run_results.append(
                    RunResult(
                        dataset=dataset,
                        shot=shot,
                        seed=seed,
                        base_accuracy=base_accuracy,
                        novel_accuracy=novel_accuracy,
                        harmonic_mean=harmonic(
                            base_accuracy,
                            novel_accuracy,
                        ),
                        base_log=(
                            str(selected_base_log)
                            if selected_base_log is not None
                            else " | ".join(
                                map(str, base_candidates)
                            )
                        ),
                        novel_log=(
                            str(selected_novel_log)
                            if selected_novel_log is not None
                            else " | ".join(
                                map(str, novel_candidates)
                            )
                        ),
                    )
                )

    per_seed_rows = [
        {
            "dataset": result.dataset,
            "shot": result.shot,
            "seed": result.seed,
            "base_accuracy": fmt(
                result.base_accuracy,
                6,
            ),
            "novel_accuracy": fmt(
                result.novel_accuracy,
                6,
            ),
            "harmonic_mean": fmt(
                result.harmonic_mean,
                6,
            ),
            "base_log": result.base_log,
            "novel_log": result.novel_log,
        }
        for result in run_results
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
            selected = [
                result
                for result in run_results
                if result.shot == shot
                and result.dataset == dataset
            ]

            base_values = [
                result.base_accuracy
                for result in selected
                if result.base_accuracy is not None
            ]

            novel_values = [
                result.novel_accuracy
                for result in selected
                if result.novel_accuracy is not None
            ]

            harmonic_values = [
                result.harmonic_mean
                for result in selected
                if result.harmonic_mean is not None
            ]

            base_mean = mean(base_values)
            base_std = sample_std(base_values)

            novel_mean = mean(novel_values)
            novel_std = sample_std(novel_values)

            harmonic_seed_mean = mean(harmonic_values)
            harmonic_seed_std = sample_std(harmonic_values)

            harmonic_from_means = harmonic(
                base_mean,
                novel_mean,
            )

            row = {
                "dataset": dataset,
                "shot": shot,
                "n_expected_seeds": len(args.seeds),
                "n_base_seeds": len(base_values),
                "n_novel_seeds": len(novel_values),
                "n_paired_seeds": len(harmonic_values),
                "base_mean": fmt(base_mean, 6),
                "base_std": fmt(base_std, 6),
                "novel_mean": fmt(novel_mean, 6),
                "novel_std": fmt(novel_std, 6),
                "harmonic_mean_of_seed_pairs": fmt(
                    harmonic_seed_mean,
                    6,
                ),
                "harmonic_std_of_seed_pairs": fmt(
                    harmonic_seed_std,
                    6,
                ),
                "harmonic_of_base_novel_means": fmt(
                    harmonic_from_means,
                    6,
                ),
            }

            dataset_summary_rows.append(row)

            dataset_summary_lookup[(shot, dataset)] = {
                "base_mean": base_mean,
                "base_std": base_std,
                "novel_mean": novel_mean,
                "novel_std": novel_std,
                "harmonic_seed_mean": harmonic_seed_mean,
                "harmonic_seed_std": harmonic_seed_std,
                "harmonic_from_means": harmonic_from_means,
                "n_paired": len(harmonic_values),
            }

    write_csv(
        args.report_dir
        / "per_dataset_shot_summary.csv",
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

    shot_seed_rows: list[dict] = []

    shot_seed_numeric: dict[
        tuple[int, int],
        tuple[
            Optional[float],
            Optional[float],
            Optional[float],
            int,
        ],
    ] = {}

    for shot in args.shots:
        for seed in args.seeds:
            selected = [
                result
                for result in run_results
                if result.shot == shot
                and result.seed == seed
            ]

            paired = [
                result
                for result in selected
                if result.base_accuracy is not None
                and result.novel_accuracy is not None
            ]

            base_average = mean(
                result.base_accuracy
                for result in paired
                if result.base_accuracy is not None
            )

            novel_average = mean(
                result.novel_accuracy
                for result in paired
                if result.novel_accuracy is not None
            )

            harmonic_average = harmonic(
                base_average,
                novel_average,
            )

            shot_seed_numeric[(shot, seed)] = (
                base_average,
                novel_average,
                harmonic_average,
                len(paired),
            )

            shot_seed_rows.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "n_complete_datasets": len(paired),
                    "n_expected_datasets": len(args.datasets),
                    "base_dataset_average": fmt(
                        base_average,
                        6,
                    ),
                    "novel_dataset_average": fmt(
                        novel_average,
                        6,
                    ),
                    "harmonic_of_dataset_averages": fmt(
                        harmonic_average,
                        6,
                    ),
                }
            )

    write_csv(
        args.report_dir
        / "shot_seed_benchmark_averages.csv",
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
        seed_values = [
            shot_seed_numeric[(shot, seed)]
            for seed in args.seeds
        ]

        complete_seed_values = [
            value
            for value in seed_values
            if value[3] == len(args.datasets)
        ]

        usable_values = (
            complete_seed_values
            or [
                value
                for value in seed_values
                if value[0] is not None
                and value[1] is not None
            ]
        )

        base_seed_averages = [
            value[0]
            for value in usable_values
            if value[0] is not None
        ]

        novel_seed_averages = [
            value[1]
            for value in usable_values
            if value[1] is not None
        ]

        harmonic_seed_averages = [
            value[2]
            for value in usable_values
            if value[2] is not None
        ]

        base_mean = mean(base_seed_averages)
        base_std = sample_std(base_seed_averages)

        novel_mean = mean(novel_seed_averages)
        novel_std = sample_std(novel_seed_averages)

        harmonic_mean = mean(harmonic_seed_averages)
        harmonic_std = sample_std(harmonic_seed_averages)

        harmonic_from_means = harmonic(
            base_mean,
            novel_mean,
        )

        row = {
            "shot": shot,
            "n_expected_seeds": len(args.seeds),
            "n_complete_seeds": len(
                complete_seed_values
            ),
            "n_expected_datasets_per_seed": len(
                args.datasets
            ),
            "base_mean": fmt(base_mean, 6),
            "base_std_across_seeds": fmt(
                base_std,
                6,
            ),
            "novel_mean": fmt(novel_mean, 6),
            "novel_std_across_seeds": fmt(
                novel_std,
                6,
            ),
            "harmonic_mean_across_seeds": fmt(
                harmonic_mean,
                6,
            ),
            "harmonic_std_across_seeds": fmt(
                harmonic_std,
                6,
            ),
            "harmonic_of_overall_base_novel_means": fmt(
                harmonic_from_means,
                6,
            ),
        }

        shot_summary_rows.append(row)

        shot_summary_numeric[shot] = {
            "base_mean": base_mean,
            "base_std": base_std,
            "novel_mean": novel_mean,
            "novel_std": novel_std,
            "harmonic_mean": harmonic_mean,
            "harmonic_std": harmonic_std,
            "harmonic_from_means": harmonic_from_means,
            "complete_seeds": len(
                complete_seed_values
            ),
        }

    write_csv(
        args.report_dir
        / "shot_overall_summary.csv",
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

    missing_report = (
        args.report_dir
        / "missing_or_invalid_runs.csv"
    )

    if problems:
        write_csv(
            missing_report,
            [
                "dataset",
                "shot",
                "seed",
                "split",
                "log_path",
                "problem",
            ],
            problems,
        )
    elif missing_report.exists():
        missing_report.unlink()

    report_lines = [
        "# HiCroPLReason Multi-shot Base-to-Novel Results",
        "",
        f"- Trainer: `{args.trainer}`",
        f"- Config: `{args.config}`",
        f"- Seeds: `{', '.join(map(str, args.seeds))}`",
        f"- Shots: `{', '.join(map(str, args.shots))}`",
        f"- Parsed metric: `{args.metric}`",
        "- Values are percentages.",
        "- H is the harmonic mean of base and novel accuracy.",
        "- Base accuracy uses test_base first and train_base as fallback.",
        "",
        "## Overall benchmark average by shot",
        "",
        (
            "Dataset averages are computed separately for each seed; "
            "the table reports mean ± sample standard deviation "
            "across seeds."
        ),
        "",
        "| Shot | Base | Novel | H | Complete seeds |",
        "|---:|---:|---:|---:|---:|",
    ]

    for shot in args.shots:
        summary = shot_summary_numeric[shot]

        report_lines.append(
            f"| {shot} | "
            f"{fmt_pm(summary['base_mean'], summary['base_std'])} | "
            f"{fmt_pm(summary['novel_mean'], summary['novel_std'])} | "
            f"{fmt_pm(summary['harmonic_mean'], summary['harmonic_std'])} | "
            f"{summary['complete_seeds']}/{len(args.seeds)} |"
        )

    for shot in args.shots:
        report_lines.extend(
            [
                "",
                f"## {shot}-shot per-dataset results",
                "",
                (
                    "| Dataset | Base | Novel | "
                    "H (from means) | Paired seeds |"
                ),
                "|---|---:|---:|---:|---:|",
            ]
        )

        for dataset in args.datasets:
            summary = dataset_summary_lookup[
                (shot, dataset)
            ]

            report_lines.append(
                f"| {dataset} | "
                f"{fmt_pm(summary['base_mean'], summary['base_std'])} | "
                f"{fmt_pm(summary['novel_mean'], summary['novel_std'])} | "
                f"{fmt(summary['harmonic_from_means']) or 'N/A'} | "
                f"{summary['n_paired']}/{len(args.seeds)} |"
            )

    if problems:
        report_lines.extend(
            [
                "",
                "## Missing or invalid runs",
                "",
                (
                    f"Found {len(problems)} missing or invalid results. "
                    "See `missing_or_invalid_runs.csv`."
                ),
            ]
        )

    report_path = (
        args.report_dir
        / "base2new_multishot_report.md"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote reports to: {args.report_dir}")
    print(f"Main report: {report_path}")

    if problems:
        print(
            (
                f"WARNING: {len(problems)} expected "
                "results were missing or invalid."
            ),
            file=sys.stderr,
        )

        print(
            f"See: {missing_report}",
            file=sys.stderr,
        )

        if args.strict:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY_AGGREGATOR
}


# ----------------------------------------------------------------
# Per-shot aggregation
# ----------------------------------------------------------------

aggregate_one_shot() {
    local shot="$1"

    local shot_report_dir
    local aggregation_marker

    shot_report_dir="$REPORT_ROOT/shot_${shot}"

    aggregation_marker="$STATUS_ROOT/shot_${shot}.aggregation_complete"

    mkdir -p "$shot_report_dir"

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="strict-shot-aggregation"

    echo "=================================================================="
    echo "[AGGREGATE]"
    echo "Shot   : $shot"
    echo "Seeds  : ${SEEDS[*]}"
    echo "Report : $shot_report_dir"
    echo "=================================================================="

    embedded_aggregate \
        "$shot_report_dir" \
        "$shot"

    touch "$aggregation_marker"
}


# ----------------------------------------------------------------
# Checkpoint cleanup
# ----------------------------------------------------------------

cleanup_one_shot_checkpoints() {
    local shot="$1"

    local deleted_count=0
    local checkpoint_dir=""

    if [[ "$CLEANUP_CHECKPOINTS" != "1" ]]; then
        echo "[KEEP] Checkpoints retained for shot=$shot."
        return 0
    fi

    CURRENT_SHOT="$shot"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="checkpoint-cleanup"

    echo "=================================================================="
    echo "[CLEANUP]"
    echo "Removing HiCroPLReason VLPromptLearner checkpoints for shot=$shot"
    echo "Training and evaluation logs will be retained."
    echo "=================================================================="

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            checkpoint_dir="$OUTPUT_ROOT/train_base/$dataset/shots_${shot}/$TRAINER/$CFG/seed${seed}/$MODEL_COMPONENT"

            if [[ -d "$checkpoint_dir" ]]; then
                case "$checkpoint_dir" in
                    "$OUTPUT_ROOT"/train_base/*/shots_"$shot"/"$TRAINER"/"$CFG"/seed*/"$MODEL_COMPONENT")
                        rm -rf -- "$checkpoint_dir"

                        deleted_count=$((deleted_count + 1))
                        ;;

                    *)
                        echo "ERROR: refusing unsafe cleanup path:" >&2
                        echo "       $checkpoint_dir" >&2
                        exit 1
                        ;;
                esac
            fi
        done
    done

    echo "[CLEANUP] Deleted $deleted_count VLPromptLearner directories."

    touch "$STATUS_ROOT/shot_${shot}.checkpoints_deleted"
}


# ----------------------------------------------------------------
# Cumulative report
# ----------------------------------------------------------------

refresh_cumulative_report() {
    local completed_shots=()
    local candidate_shot=""

    for candidate_shot in "${SHOTS[@]}"; do
        if [[ -f "$STATUS_ROOT/shot_${candidate_shot}.complete" ]]; then
            completed_shots+=("$candidate_shot")
        fi
    done

    if (( ${#completed_shots[@]} == 0 )); then
        echo "[CUMULATIVE] No completed shots available."
        return 0
    fi

    CURRENT_SHOT="${completed_shots[*]}"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="cumulative-report"

    echo "=================================================================="
    echo "[CUMULATIVE REPORT]"
    echo "Completed shots: ${completed_shots[*]}"
    echo "Report root    : $REPORT_ROOT"
    echo "=================================================================="

    embedded_aggregate \
        "$REPORT_ROOT" \
        "${completed_shots[@]}"

    printf '%s\n' \
        "${completed_shots[@]}" \
        > "$REPORT_ROOT/completed_shots.txt"
}


# ----------------------------------------------------------------
# Complete shot workflow
# ----------------------------------------------------------------

process_shot_all() {
    local shot="$1"

    local shot_complete_marker
    local aggregation_marker

    shot_complete_marker="$STATUS_ROOT/shot_${shot}.complete"

    aggregation_marker="$STATUS_ROOT/shot_${shot}.aggregation_complete"

    # Completed shots are validated and skipped.
    if [[ "$FORCE_RERUN" != "1" &&
          -f "$shot_complete_marker" ]]; then
        echo "[SKIP][SHOT] shot=$shot is already complete."

        aggregate_one_shot "$shot"
        refresh_cumulative_report

        return 0
    fi

    # Recover when aggregation completed but cleanup was interrupted.
    if [[ "$FORCE_RERUN" != "1" &&
          -f "$aggregation_marker" ]]; then
        echo "[RECOVER][SHOT] shot=$shot was already aggregated."

        aggregate_one_shot "$shot"
        cleanup_one_shot_checkpoints "$shot"

        touch "$shot_complete_marker"

        refresh_cumulative_report

        return 0
    fi

    echo "##################################################################"
    echo "# STARTING SHOT: $shot"
    echo "##################################################################"

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_training_job \
                "$dataset" \
                "$shot" \
                "$seed"

            run_evaluation_job \
                "$dataset" \
                "$shot" \
                "$seed" \
                base

            run_evaluation_job \
                "$dataset" \
                "$shot" \
                "$seed" \
                new
        done
    done

    aggregate_one_shot "$shot"

    cleanup_one_shot_checkpoints "$shot"

    touch "$shot_complete_marker"

    refresh_cumulative_report

    echo "##################################################################"
    echo "# COMPLETED SHOT: $shot"
    echo "# Continuing to the next shot."
    echo "##################################################################"
}


# ----------------------------------------------------------------
# Training-only workflow
# ----------------------------------------------------------------

process_shot_train_only() {
    local shot="$1"

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_training_job \
                "$dataset" \
                "$shot" \
                "$seed"
        done
    done
}


# ----------------------------------------------------------------
# Evaluation-only workflow
# ----------------------------------------------------------------

process_shot_eval_only() {
    local shot="$1"

    local shot_complete_marker

    shot_complete_marker="$STATUS_ROOT/shot_${shot}.complete"

    if [[ "$FORCE_RERUN" != "1" &&
          -f "$shot_complete_marker" ]]; then
        echo "[SKIP][SHOT] shot=$shot is already complete."
        return 0
    fi

    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_evaluation_job \
                "$dataset" \
                "$shot" \
                "$seed" \
                base

            run_evaluation_job \
                "$dataset" \
                "$shot" \
                "$seed" \
                new
        done
    done

    aggregate_one_shot "$shot"

    cleanup_one_shot_checkpoints "$shot"

    touch "$shot_complete_marker"

    refresh_cumulative_report
}


# ----------------------------------------------------------------
# Final all-shot report
# ----------------------------------------------------------------

aggregate_all_shots() {
    CURRENT_SHOT="${SHOTS[*]}"
    CURRENT_DATASET="all"
    CURRENT_SEED="all"
    CURRENT_STAGE="final-all-shot-aggregation"

    echo "=================================================================="
    echo "[FINAL AGGREGATE]"
    echo "Shots  : ${SHOTS[*]}"
    echo "Report : $REPORT_ROOT"
    echo "=================================================================="

    embedded_aggregate \
        "$REPORT_ROOT" \
        "${SHOTS[@]}"
}


# ----------------------------------------------------------------
# Resume status
# ----------------------------------------------------------------

echo "=================================================================="
echo "Resume status"

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


# ----------------------------------------------------------------
# Main mode dispatch
# ----------------------------------------------------------------

case "$MODE" in
    all)
        for shot in "${SHOTS[@]}"; do
            process_shot_all "$shot"
        done

        aggregate_all_shots
        ;;

    train)
        echo "WARNING: train mode does not evaluate or clean checkpoints." >&2
        echo "Use all mode for the full sequential workflow." >&2

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


# ----------------------------------------------------------------
# Completion summary
# ----------------------------------------------------------------

CURRENT_STAGE="completed"

printf '\nCompleted mode=%s on GPU=%s\n' \
    "$MODE" \
    "$GPU_ID"

printf 'Python interpreter       : %s\n' \
    "$PYTHON_REALPATH"

printf 'Combined report directory: %s\n' \
    "$REPORT_ROOT"

printf 'Combined CSV report      : %s\n' \
    "$REPORT_ROOT/shot_overall_summary.csv"

printf 'Combined Markdown report : %s\n' \
    "$REPORT_ROOT/base2new_multishot_report.md"

printf 'Completed shots file     : %s\n' \
    "$REPORT_ROOT/completed_shots.txt"