#!/bin/bash
# Set variables that remain constant across datasets
ROOT="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
CONFIG="configs/trainers/CoOp/vit_b32_plip_c16_ep50_batch16.yaml"
LOAD_EPOCH=50
SUBSAMPLE="all"

# List of datasets to loop through
#"fgvc_aircraft" "stanford_cars" "sun397"
datasets=("oxford_flowers")

# Loop through each dataset in the list
for dataset in "${datasets[@]}"; do
  echo "Evaluating dataset: ${dataset}"

  # Set dataset-specific variables
  DATASET_CFG="configs/datasets/${dataset}.yaml"
  MODEL_DIR="/storagepool/Ashshak/output/base2new/train_base/${dataset}/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx"
  OUTPUT_DIR="reliability_diagram/${dataset}_base"

  # Run the evaluation script with the specified arguments for the current dataset
  python reliability_diagram.py \
    --root ${ROOT} \
    --config-file ${CONFIG} \
    --dataset-config-file ${DATASET_CFG} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOAD_EPOCH} \
    --output-dir ${OUTPUT_DIR} \
    --subsample-classes ${SUBSAMPLE}
done
