#!/bin/bash
GPU_ID="${1:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base data path and trainer
DATA="/storagepool/Ashshak/DR"   #"/storagepool/Ashshak/DR"   #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=ProGrad
CFG=rn50_ep100
CTP=end 
NCTX=16
SHOTS=16
CSC=False
LAMBDA=1.0
# List of datasets to loop over
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft
DATASETS=(aptos messidor messidor_2 eyepacs) #("pannuke" "kather" "digestpath")  #(aptos messidor messidor_2 eyepacs)   #(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat)

# List of seeds to loop over
SEEDS=(1 2 3)

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        DIR=/storagepool/Ashshak/output3/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        
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
            LOSS.LAMBDA ${LAMBDA} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC ${CSC} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.SUBSAMPLE_CLASSES all
    done
done
