#!/bin/bash

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Datasets to process
DATASETS=(
    caltech101 food101 dtd ucf101 oxford_flowers oxford_pets fgvc_aircraft stanford_cars sun397 eurosat
)

# Common settings
SHOTS=16
TRAINER="HiCroPLReason"
CFG="vit_b16_c2_ep50_batch32_16ctx"

# Common result root
RESULT_ROOT="/storagepool/Ashshak/output2/base2new"

# Timestamped log file
TIMESTAMP=$(date +%F_%H-%M-%S)
LOGFILE="parse_results_${TIMESTAMP}.txt"

echo "Logging all results to $LOGFILE"

{
    echo "========== ALL RESULTS ($TIMESTAMP) =========="
    echo ""
} > "$LOGFILE"

for DATASET in "${DATASETS[@]}"; do

    BASE_DIR="${RESULT_ROOT}/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}"
    NOVEL_DIR="${RESULT_ROOT}/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}"

    echo "Parsing results for dataset: ${DATASET}" | tee -a "$LOGFILE"

    # Check that both result folders exist
    if [ ! -d "$BASE_DIR" ]; then
        echo "ERROR: Base directory not found:" | tee -a "$LOGFILE"
        echo "$BASE_DIR" | tee -a "$LOGFILE"
        echo "-----------------------------" | tee -a "$LOGFILE"
        echo "" >> "$LOGFILE"
        continue
    fi

    if [ ! -d "$NOVEL_DIR" ]; then
        echo "ERROR: Novel directory not found:" | tee -a "$LOGFILE"
        echo "$NOVEL_DIR" | tee -a "$LOGFILE"
        echo "-----------------------------" | tee -a "$LOGFILE"
        echo "" >> "$LOGFILE"
        continue
    fi

    # ---------------------------------------------------------
    # Base-class metrics
    # ---------------------------------------------------------
    echo "--- Base classes ---" | tee -a "$LOGFILE"

    python parse_test_res.py \
        "$BASE_DIR" \
        | tee -a "$LOGFILE"

    # ---------------------------------------------------------
    # Novel-class metrics
    # ---------------------------------------------------------
    echo "--- Novel classes ---" | tee -a "$LOGFILE"

    python parse_test_res.py \
        "$NOVEL_DIR" \
        --test-log \
        | tee -a "$LOGFILE"

    # ---------------------------------------------------------
    # Base-to-novel harmonic mean
    # ---------------------------------------------------------
    echo "--- Harmonic Mean ---" | tee -a "$LOGFILE"

    python parse_test_res.py \
        --base-dir "$BASE_DIR" \
        --novel-dir "$NOVEL_DIR" \
        --test-log \
        | tee -a "$LOGFILE"

    echo "-----------------------------" | tee -a "$LOGFILE"
    echo "" >> "$LOGFILE"

done

echo "All results saved to: $LOGFILE"