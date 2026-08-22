#!/bin/bash
# CUDA

set -Eeuo pipefail

############################################
# GPU
############################################
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# dataset
DATA_DIR=/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA #"/storagepool/Ashshak/DR"
# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false
# Datasets to use
datasets=("kather" "pannuke" "digestpath") #(aptos eyepacs messidor messidor_2)
seeds=(1 2 3)
SHOTS=16
# model
BACKBONE=vit_b32_plip
# trainer
TRAINER=ZeroshotCLIP
# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'ace' 'mce' 'ece_kde')

ROOT_OUT="/storagepool/Ashshak/output_zero/all"
SUMMARY_DIR="/storagepool/Ashshak/output_zero/summaries"
mkdir -p "$SUMMARY_DIR"
# Build trainer config with specific parameters
BATCH_SIZE=100  # Test batch size
TRAINER_CFG="${BACKBONE}_batch${BATCH_SIZE}"

############################################
# Sanity checks
############################################
if ! python3 - <<'PY' >/dev/null 2>&1
import yaml  # PyYAML check
PY
then
  echo "[WARN] PyYAML not installed in this environment; LOSS_DIR tag may be empty."
fi

############################################
# Helpers
############################################
join_by_comma() {
  local IFS=,
  echo "$*"
}

# Extract LOSS_DIR tag from YAML safely (NO trailing args after heredoc!)
get_loss_dir() {
  local cfg="$1"
  python3 - "$cfg" <<'PY'
import sys, yaml
cfg_path = sys.argv[1]
try:
    with open(cfg_path, 'r') as f:
        c = yaml.safe_load(f)
    # Adjust these keys if your YAML differs
    losses  = c['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES']
    weights = [str(c['TRAINER']['COOP']['LOSS'][L]['WEIGHT']) for L in losses]
    print(f"losses_{'_'.join(losses)}_weights_{'_'.join(weights)}")
except Exception:
    # Print empty so the caller can continue without a tag
    print("")
PY
}

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Evaluating PLIP on dataset: ${dataset} (seed: ${seed})"
        # evaluates on all classes
        bash scripts/classification/all_zeroshot_plip.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}
    done
    
done
# parse results
#echo "Parsing results for dataset: ${dataset}"
#RESULTS_DIR="output/all_zeroshot/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG}"
#for keyword in "${KEYWORDS[@]}"; do
#        python parse_test_res.py ${RESULTS_DIR} --test-log --keyword ${keyword}
#done
##########################################
# CONSOLIDATE (one file: averages for all metrics per dataset)
##########################################
FILTER="shots_${SHOTS}/${TRAINER}/${TRAINER_CFG}"
#[[ -n "$LOSS_DIR" ]] && FILTER="${FILTER}/${LOSS_DIR}"

KEYWORDS_CSV="$(join_by_comma "${KEYWORDS[@]}")"
CONSOL_FILE="${SUMMARY_DIR}/${TRAINER}_${TRAINER_CFG}_shots${SHOTS}.csv"

EXTRA_ARGS=()
#[[ -n "$CALIBRATION_CONFIG_JSON" ]] && EXTRA_ARGS+=( --calibration-config "$CALIBRATION_CONFIG_JSON" )

python3 parse_test_res.py "$ROOT_OUT" \
    --scan-deep --test-log \
    --keywords "$KEYWORDS_CSV" \
    --path-filter "$FILTER" \
    --ci95 \
    --consolidate-file "$CONSOL_FILE" \
    --wide --save-mode overwrite \
    "${EXTRA_ARGS[@]}"

echo "[OK] Consolidated metrics saved to: $CONSOL_FILE"