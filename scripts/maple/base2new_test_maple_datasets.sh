#!/bin/bash
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base config
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset" #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"   #"/storagepool/Ashshak/DR" #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx
SHOTS=16
LOADEP=5
SUB=new
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets and seeds
#aptos eyepacs messidor messidor_2
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)   #("pannuke" "kather" "digestpath") #(caltech101 food101 dtd eurosat)
SEEDS=(1 2 3)

# Loop through datasets and seeds
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=/storagepool/Ashshak/output3/base2new/train_base/${COMMON_DIR}
        DIR=/storagepool/Ashshak/output3/base2new/test_${SUB}/${COMMON_DIR}

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
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB} \
            TRAINER.MAPLE.PLOT_ANGDIST True \
            TRAINER.MAPLE.ANGDIST_MAX_BATCHES 50 \
            TRAINER.MAPLE.ANGDIST_MAX_CLASSES 0
    done
done
