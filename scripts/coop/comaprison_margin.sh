#!/usr/bin/env bash
set -euo pipefail

# === config (matches your command) ===
DATASET="oxford_flowers"

ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
METHOD_CFG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml"
DATASET_CFG="configs/datasets/${DATASET}.yaml"

BASE_MODEL_DIR="/storagepool/Ashshak/output_test/base2new/train_base/${DATASET}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
OUTPUT_DIR="./margin_figs/${DATASET}_base"

SUBSPLIT="base"
EVAL_SPLIT="auto"
METHODS=(CE Mbls Margin)
SEED=2
LOAD_EPOCH=5

SCRIPT="comparison_analyse.py"

# === optional: activate your env ===
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate maple

mkdir -p "$OUTPUT_DIR"

echo "Running margin analysis for ${DATASET} (${SUBSPLIT})..."
echo "Base model dir: $BASE_MODEL_DIR"
echo "Output dir    : $OUTPUT_DIR"
echo "Methods       : ${METHODS[*]}"
echo "Eval split    : $EVAL_SPLIT"

# quick sanity listing (helpful if paths are off)
ls -lah "$BASE_MODEL_DIR" || true
ls -lah "$BASE_MODEL_DIR/seed${SEED}" || true
ls -lah "$BASE_MODEL_DIR/seed${SEED}/CE" || true
ls -lah "$BASE_MODEL_DIR/seed${SEED}/Mbls" || true
ls -lah "$BASE_MODEL_DIR/seed${SEED}/rmargin" || true

python "$SCRIPT" \
  --root "$ROOT" \
  --config-file "$METHOD_CFG" \
  --dataset-config-file "$DATASET_CFG" \
  --base-model-dir "$BASE_MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --subsample-classes "$SUBSPLIT" \
  --eval-split "$EVAL_SPLIT" \
  --methods "${METHODS[@]}" \
  --seed "$SEED" \
  --load-epoch "$LOAD_EPOCH"

echo "Done."
