#!/bin/bash
# CUDA
GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# dataset
DATA_DIR="/storagepool/Ashshak/DR"
# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false
# Datasets to use
new_class_datasets=(aptos eyepacs messidor messidor_2)

seeds=(1 2 3)

SHOTS=16

# model
BACKBONE=vit_b32_biomedclip
# trainer

TRAINERS=('CoOp_BioMedCLIP')

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'mce' 'ace' 'ece_kde')

# Optional: calibration config JSON for log filename suffixes (leave empty if unused)
CALIBRATION_CONFIG_JSON=""
# Example:
# CALIBRATION_CONFIG_JSON='{"BASE_CALIBRATION_MODE": true, "SCALING_CONFIG": true, "SCALING_CALIBRATOR_NAME": "ts", "BIN_CALIBRATOR_NAME": "", "IF_DAC": false, "IF_PROCAL": false}'

############################################
# OUTPUT ROOTS
############################################
ROOT_OUT="/storagepool/Ashshak/output/all"
SUMMARY_DIR="/storagepool/Ashshak/output/summaries"
mkdir -p "$SUMMARY_DIR"

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

############################################
# Main
############################################
for TRAINER in "${TRAINERS[@]}"; do
  case "$TRAINER" in
    "CoOp_BioMedCLIP")
      EPOCH=50; BATCH_SIZE=16; N_CTX=16
      ;;
    *)
      echo "[ERR] Unknown trainer: $TRAINER" >&2
      exit 1
      ;;
  esac

  TRAINER_CFG="${BACKBONE}_c${N_CTX}_ep${EPOCH}_batch${BATCH_SIZE}"
  CONFIG_FILE="configs/trainers/${TRAINER}/${TRAINER_CFG}.yaml"

  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERR] Missing config: $CONFIG_FILE" >&2
    exit 1
  fi

  LOSS_DIR="$(get_loss_dir "$CONFIG_FILE")"
  if [[ -n "$LOSS_DIR" ]]; then
    echo "[INFO] LOSS_DIR=$LOSS_DIR"
  else
    echo "[WARN] LOSS_DIR is empty (YAML keys missing or parse failed). Continuing."
  fi

  ##########################################
  # TRAIN per dataset x seed
  ##########################################
  for dataset in "${new_class_datasets[@]}"; do
    for seed in "${seeds[@]}"; do
      bash scripts/classification/all_fewshot_biomedclip.sh \
        "$TRAINER" "$TRAINER_CFG" "$dataset" "$DATA_DIR" "$SHOTS" "$seed"
    done
  done

  ##########################################
  # CONSOLIDATE (one file: averages for all metrics per dataset)
  ##########################################
  FILTER="shots_${SHOTS}/${TRAINER}/${TRAINER_CFG}"
  [[ -n "$LOSS_DIR" ]] && FILTER="${FILTER}/${LOSS_DIR}"

  KEYWORDS_CSV="$(join_by_comma "${KEYWORDS[@]}")"
  CONSOL_FILE="${SUMMARY_DIR}/${TRAINER}_${TRAINER_CFG}_shots${SHOTS}$( [[ -n "$LOSS_DIR" ]] && echo "_${LOSS_DIR}" ).csv"

  EXTRA_ARGS=()
  [[ -n "$CALIBRATION_CONFIG_JSON" ]] && EXTRA_ARGS+=( --calibration-config "$CALIBRATION_CONFIG_JSON" )

  python3 parse_test_res.py "$ROOT_OUT" \
    --scan-deep --test-log \
    --keywords "$KEYWORDS_CSV" \
    --path-filter "$FILTER" \
    --ci95 \
    --consolidate-file "$CONSOL_FILE" \
    --wide --save-mode overwrite \
    "${EXTRA_ARGS[@]}"

  echo "[OK] Consolidated metrics saved to: $CONSOL_FILE"
done
