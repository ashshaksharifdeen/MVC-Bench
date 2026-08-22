#!/bin/bash
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Base data path and trainer
DATA="/l/users/ashshak.sharifdeen/dataset" #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset" #"/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA"   #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"   #"/home/abhishek/desktop/VLM_Cal/CalibPrompt/DATA"#"/storagepool/Ashshak/DR" #"/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset" #"/storagepool/Ashshak/DR"
TRAINER=HiCroPLReason    #MaPLe
CFG=vit_b16_c2_ep50_batch32_16ctx   #vit_b16_c2_ep5_batch4_2ctx
SHOTS=16

# List of datasets to loop over
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft
#aptos eyepacs messidor messidor_2
DATASETS=(caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars eurosat)  #("pannuke" "kather" "digestpath") #(rsna18 covid)   #("pannuke" "kather" "digestpath")   #(caltech101 food101 dtd eurosat) #("pannuke" "kather" "digestpath") #(rsna18 covid)

# List of seeds to loop over
SEEDS=(1 2 3)

declare -A PROBE_LAMBDA_BY_DATASET=(
    [caltech101]=5.0
    [imagenet]=5.0
    [food101]=12.0
    [dtd]=5.0
    [ucf101]=0.5
    [oxford_flowers]=3.0
    [oxford_pets]=12.0
    [fgvc_aircraft]=5.0
    [stanford_cars]=4.0
    [sun397]=4.0
    [eurosat]=4.0
)

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do

    if [[ ! -v "PROBE_LAMBDA_BY_DATASET[$DATASET]" ]]; then
        echo "ERROR: PROBE_LAMBDA is not defined for dataset: ${DATASET}"
        exit 1
    fi

    PROBE_LAMBDA="${PROBE_LAMBDA_BY_DATASET[$DATASET]}"

    echo "Dataset      : ${DATASET}"
    echo "PROBE_LAMBDA : ${PROBE_LAMBDA}"
    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        DIR=/l/users/ashshak.sharifdeen/output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        
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
            --profile-gflops \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base \
            TRAINER.HICROPLReason.PROBE_LAMBDA "${PROBE_LAMBDA}"
    done
done
