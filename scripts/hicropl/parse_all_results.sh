#!/bin/bash
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
#caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
# List of datasets to process
DATASETS=(aptos messidor messidor_2 eyepacs)  #("pannuke" "kather" "digestpath") #(rsna18 covid)   #("pannuke" "kather" "digestpath")  #(caltech101 food101 dtd eurosat)   #(aptos eyepacs messidor messidor_2)

# Common settings
SHOTS=16
TRAINER=HiCroPL
CFG=vit_b16_c2_ep50_batch32_16ctx  #vit_b16_c2_ep5_batch4_2ctx #vit_b16_c2_ep5_batch4_2ctx_cross_datasets #vit_b16_c2_ep5_batch4_2ctx

# Timestamped logfile name
TIMESTAMP=$(date +%F_%H-%M-%S)
LOGFILE=parse_results_${TIMESTAMP}.txt

# Start logging
echo "Logging all results to $LOGFILE"
echo "========== ALL RESULTS ($TIMESTAMP) ==========" > $LOGFILE
echo "" >> $LOGFILE

# Loop through each dataset
for DATASET in "${DATASETS[@]}"; do
    echo "Parsing results for dataset: ${DATASET}" | tee -a $LOGFILE
    echo "--- Base classes ---" | tee -a $LOGFILE

    python parse_test_res.py /l/users/ashshak.sharifdeen/output2/base2new/train_all/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG} \
        | tee -a $LOGFILE

    echo "-----------------------------" | tee -a $LOGFILE
    echo "" >> $LOGFILE
done
