#!/bin/bash

# CUDA
export CUDA_VISIBLE_DEVICES=$1

# dataset
DATA_DIR=/home/abhishek/desktop/VLM_Cal/CLIP_Calibration/\$DATA #/mnt/sharedata/ssd/common/datasets/
# new_class_datasets=("caltech101" "oxford_pets" "stanford_cars" "oxford_flowers" "food101" "fgvc_aircraft" "sun397" "dtd" "eurosat" "ucf101" "imagenet")
new_class_datasets=("kather")
seed=1
SHOTS=16

# model
BACKBONE=vit_b32 # ("rn50" "vit_b32" "vit_b16" "vit_l14")

# trainer
TRAINER=ZeroshotCLIP

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'ace' 'mce' 'piece')


CFG=$BACKBONE

for dataset in "${new_class_datasets[@]}"; do

    # evaluates on all classes
    bash scripts/classification/all_zeroshot_clip.sh ${TRAINER} ${CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}

    for keyword in "${KEYWORDS[@]}"; do
        # prints averaged results for all classes
        python parse_test_res.py output/all/zeroshot/${dataset}/shots_${SHOTS}/clip/${CFG} --test-log --keyword ${keyword}
    done
done
