#!/bin/bash
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base config
DATA="/l/users/ashshak.sharifdeen/dataset"  #"/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA" #"/storagepool/Ashshak/DR"
TRAINER=HiCroPLReason
CFG=vit_b16_c2_ep5_batch32_2ctx_cross_datasets    #vit_b16_c2_ep50_batch32_16ctx 
SHOTS=16
LOADEP=5
SUB=all
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
#imagenet_a imagenet_r imagenet_sketch imagenetv2  
# List of datasets and seeds
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)   #(imagenet_a imagenet_r imagenet_sketch imagenetv2)   #(aptos eyepacs messidor_2)
SEEDS=(1 2 3)

# Loop through datasets and seeds
for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        COMMON_DIR=imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=/l/users/ashshak.sharifdeen/output/base2new/train_all/${COMMON_DIR}
        DIR=/l/users/ashshak.sharifdeen/output3/base2new/test_${SUB}/${OUTPUT_DIR}

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
