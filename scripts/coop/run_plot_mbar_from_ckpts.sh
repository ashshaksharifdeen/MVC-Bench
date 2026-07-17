#!/usr/bin/env bash
set -euo pipefail

# You can override with: PYTHON=python3 ./run_plot_mbar_from_ckpts.sh
PYTHON="${PYTHON:-python}"

# --- Configs & dataset root ---
DATASET_CFG="configs/datasets/aptos.yaml"
TRAINER_CFG="configs/trainers/CoOp/vit_b16_ep50.yaml"
ROOT="/storagepool/Ashshak/DR"

# --- Checkpoints (CE / MBLS / MCM) ---
CE="/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/CE/prompt_learner/model.pth.tar-50"
Mean="/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/Mean/prompt_learner/model.pth.tar-50"
Var="/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed3/Var/prompt_learner/model.pth.tar-50"
MCM="/storagepool/Ashshak/output/base2new/train_base/aptos/shots_16/CoOp/vit_b16_ep50/seed2/MCM/prompt_learner/model.pth.tar-50"

# --- Output directory ---
OUTDIR="./cvpr_margin/aptos_seed2_ep50"
mkdir -p "$OUTDIR"

# --- (Optional) epoch override if your file path doesn't include '-50'
# EPOCH=50

# --- Run ---
"$PYTHON" story_mbar_mass_plot.py \
  --dataset-config-file "$DATASET_CFG" \
  --config-file "$TRAINER_CFG" \
  --root "$ROOT" \
  --split auto \
  --ce   "$CE" \
  --mean "$Mean" \
  --var "$Var" \
  --mcm  "$MCM" \
  --outdir "$OUTDIR"
  # If needed, append: --epoch "$EPOCH"
