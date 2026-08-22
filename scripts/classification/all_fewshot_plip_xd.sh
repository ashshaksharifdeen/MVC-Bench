#!/bin/bash

#cd ../..

# custom config

# trainer
TRAINER=$1
CFG=$2

# dataset
DATASET=$3
DATA=$4
SHOTS=$5
SUB=all

SEED=$6
TARGET_DATASET=$7
LOADEP=50
# Extract loss information from config file
LOSS_DIR=$(python -c "
import yaml
config = yaml.safe_load(open('configs/trainers/${TRAINER}/${CFG}.yaml'))
losses = '_'.join(config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'])
weights = '_'.join(str(config['TRAINER']['COOP']['LOSS'][loss]['WEIGHT']) for loss in config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'])
print(f'losses_{losses}_weights_{weights}')" 2>/dev/null)


if [ -z "$LOSS_DIR" ]; then
    LOSS_DIR="default"
fi

echo "Using loss directory: ${LOSS_DIR}"

# Create directory with loss information
DIR=/storagepool/Ashshak/output_plip/all/${TARGET_DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${LOSS_DIR}/seed${SEED}





if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${TARGET_DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir /storagepool/Ashshak/output_plip/all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${LOSS_DIR}/seed${SEED} \
    --output-dir ${DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.NAME "plip" \
    MODEL_ROOT "/home/ashashak/CalibPrompt/models"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${TARGET_DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --model-dir /storagepool/Ashshak/output_plip/all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/${LOSS_DIR}/seed${SEED} \
    --output-dir ${DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB} \
    MODEL.NAME "plip" \
    MODEL_ROOT "/home/ashashak/CalibPrompt/models"
fi