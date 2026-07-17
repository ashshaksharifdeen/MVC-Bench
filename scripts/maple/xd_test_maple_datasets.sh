#!/bin/bash
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base config
DATA="/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA"  #"/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA" #"/storagepool/Ashshak/DR"
TRAINER=MaPLe
CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16
LOADEP=2
SUB=all
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets and seeds
DATASETS=(covid)   #(aptos eyepacs messidor_2)
SEEDS=(1 2 3)

# Loop through datasets and seeds
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        COMMON_DIR=rsna18/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=/storagepool/Ashshak/output/base2new/train_all/${COMMON_DIR}
        DIR=/storagepool/Ashshak/output/base2new/test_${SUB}/${OUTPUT_DIR}

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
            --eval-only 
           
           
    done
done
