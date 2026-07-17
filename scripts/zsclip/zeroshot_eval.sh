#!/bin/bash
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base config
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset" #"/storagepool/Ashshak/DR"   #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset" #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=ZeroshotCLIP
CFG=vit_b16    #rn101_ep50 
SUB=new
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets and seeds
#aptos eyepacs messidor messidor_2
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)  #(aptos eyepacs messidor messidor_2)   #(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)
SEEDS=(1 2 3)

# Loop through datasets and seeds
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COMMON_DIR=${DATASET}/${TRAINER}/${CFG}/seed${SEED}
        DIR=/storagepool/Ashshak/output/base2new/test_${SUB}/${COMMON_DIR}

        echo "---------------------------------------------"
        echo "Evaluating ${DATASET} | Seed ${SEED}"
        echo "Output dir: ${DIR}"
        echo "Model dir : ${MODEL_DIR}"
        echo "---------------------------------------------"

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/CoOp/${CFG}.yaml \
            --output-dir ${DIR} \
            --eval-only \
            DATASET.SUBSAMPLE_CLASSES ${SUB} \
            TEST.PLOT_ANGDIST True
    done
done