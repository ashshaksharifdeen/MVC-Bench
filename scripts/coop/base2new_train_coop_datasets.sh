#!/bin/bash
GPU_ID="${1:-2}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base data path and trainer
DATA="/storagepool/Ashshak/DR"  #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"  #"/storagepool/Ashshak/DR"
TRAINER=CoOp
CFG=vit_b16_ep50 #rn101_ep50    #rn101_ep50    #rn50_ep50    #vit_b16_ep50    #rn50_ep50    #vit_b32_ep50        #vit_b16_ep50
CTP=middle 
NCTX=16
SHOTS=16
CSC=False
# List of datasets to loop over
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft
#aptos eyepacs messidor messidor_2
DATASETS=(aptos eyepacs messidor messidor_2) #(rsna18,covid) ("pannuke" "kather" "digestpath")
# List of seeds to loop over
SEEDS=(1 2 3)
CAL_BINS=20
SAVE_CLASSWISE=True

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        DIR=/storagepool/Ashshak/output4/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Resuming..."
        else
            echo "Running training for dataset: ${DATASET}, seed: ${SEED}"
        fi

        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.SUBSAMPLE_CLASSES all \
            TEST.CALIBRATION_BINS ${CAL_BINS} \
            TEST.SAVE_CLASSWISE_CALIBRATION ${SAVE_CLASSWISE}
    done
done
