#!/usr/bin/env bash
set -euo pipefail

# You can override with: PYTHON=python3 ./run_plot_mbar_from_ckpts.sh
PYTHON="${PYTHON:-python}"

# --- Configs & dataset root ---
DATASET_CFG="configs/datasets/food101.yaml"
TRAINER_CFG="configs/trainers/MaPLe/vit_b16_c2_ep5_batch4_2ctx_comp.yaml"
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"

# --- Checkpoints (CE / MBLS / MCM) ---
CE="/storagepool/Ashshak/output2/base2new/train_base/eurosat/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx_comp/seed1/CE/MultiModalPromptLearner/model.pth.tar-5"
Mean="/storagepool/Ashshak/output2/base2new/train_base/eurosat/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx_comp/seed1/ECCV_ZS/MultiModalPromptLearner/model.pth.tar-5"
Var="/storagepool/Ashshak/output2/base2new/train_base/caltech101/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx_comp/seed2/ECCV_penalty/MultiModalPromptLearner/model.pth.tar-5"
Margin="/storagepool/Ashshak/output2/base2new/train_base/food101/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx_comp/seed1/Margin/MultiModalPromptLearner/model.pth.tar-5"

# --- Output directory ---
OUTDIR="./cvpr_margin_maple26eccvcomp/food101_seed2_ep5"
mkdir -p "$OUTDIR"

# --- (Optional) epoch override if your file path doesn't include '-50'
# EPOCH=50

# --- Run ---
"$PYTHON" story_mbar_mass_plot_sals.py \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAINER_CFG" \
  --root "$ROOT" \
  --split auto \
  --ce   "$CE" \
  --mean "$Mean" \
  --var "$Var" \
  --mcm  "$Margin" \
  --outdir "$OUTDIR"
  # If needed, append: --epoch "$EPOCH"
 # #--ce   "$CE" \