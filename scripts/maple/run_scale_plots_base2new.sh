#!/bin/bash
set -euo pipefail

# Match your training script settings
TRAINER="MaPLe"
CFG="vit_b16_c2_ep5_batch4_2ctx"
SHOTS=16

# Same datasets/seeds used in base2new_train_maple_datasets.sh
DATASETS=(caltech101 food101 dtd eurosat)
SEEDS=(1 2 3)

# Root of your training outputs (from your script)
OUT_ROOT="/storagepool/Ashshak/output4/base2new/train_base"

# Filenames created by your logger + analysis output folder
CSV_NAME="scale_grad_log.csv"
ANALYSIS_DIRNAME="scale_analysis"

for DATASET in "${DATASETS[@]}"; do
  for SEED in "${SEEDS[@]}"; do

    RUN_DIR="${OUT_ROOT}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"
    CSV_PATH="${RUN_DIR}/${CSV_NAME}"
    OUTDIR="${RUN_DIR}/${ANALYSIS_DIRNAME}"

    echo "--------------------------------------------------"
    echo "Dataset: ${DATASET} | Seed: ${SEED}"
    echo "CSV   : ${CSV_PATH}"
    echo "Outdir: ${OUTDIR}"
    echo "--------------------------------------------------"

    if [[ ! -f "${CSV_PATH}" ]]; then
      echo "[WARN] Missing ${CSV_NAME} in ${RUN_DIR}. Skipping..."
      continue
    fi

    python plot_scale_grad_ratios.py \
      --csv "${CSV_PATH}" \
      --outdir "${OUTDIR}"

  done
done

echo "[DONE] Finished generating scale plots/tables."
