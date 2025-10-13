#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

DATA_DIR=/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/\$DATA
datasets=("oxford_flowers")
seed=1
SHOTS=16
BACKBONE=vit_b32
TRAINER=ZeroshotCLIP
KEYWORDS=('accuracy' 'confidence' 'ece' 'ace' 'mce' 'piece')
CFG=$BACKBONE

for dataset in "${datasets[@]}"; do
    # All classes
    bash scripts/classification/all_zeroshot_plip.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}
    
    for keyword in "${KEYWORDS[@]}"; do
        python parse_test_res.py output/all/zeroshot/${dataset}/shots_${SHOTS}/plip/${CFG} --test-log --keyword ${keyword}
    done
done

