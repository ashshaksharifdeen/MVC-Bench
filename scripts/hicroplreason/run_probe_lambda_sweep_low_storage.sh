#!/bin/bash
set -euo pipefail

# =========================================================
# Low-storage PROBE_LAMBDA sweep for HiCroPLReason
#
# Process:
#   1. For one lambda:
#       - train base classes
#       - evaluate novel classes
#       - delete checkpoint files after each seed eval
#       - keep logs temporarily
#   2. Parse averaged base + novel result for that lambda
#   3. Save averaged report with lambda value in filename
#   4. Delete that lambda's train/test folders
#   5. Move to next lambda
# =========================================================

GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---------------------------------------------------------
# Basic settings
# ---------------------------------------------------------
DATA="/storagepool/Ashshak/Vlm-calibration/C-TPT/dataset"
TRAINER=HiCroPLReason
CFG=vit_b16_c2_ep50_batch32_16ctx
SHOTS=16
LOADEP=50

# Update datasets here as needed
DATASETS=(sun397)

# Update seeds here as needed
SEEDS=(1 2 3)

# Lambda values to sweep
LAMBDAS=(0.01 0.1 0.2 0.5 3.0 4.0 5.0 6.0 7.0 10.0 12.0)

# ---------------------------------------------------------
# Output root
# ---------------------------------------------------------
# Use a separate sweep root so deletion is safe and does not touch old experiments.
SWEEP_ROOT="/storagepool/Ashshak/output2/base2new_probe_lambda_sweep"

# Permanent reports are saved here and will NOT be deleted.
REPORT_DIR="${SWEEP_ROOT}/reports"
mkdir -p "${REPORT_DIR}"

TIMESTAMP=$(date +%F_%H-%M-%S)

MASTER_REPORT="${REPORT_DIR}/MASTER_probe_lambda_sweep_${TIMESTAMP}.txt"

# ---------------------------------------------------------
# Deletion options
# ---------------------------------------------------------
# DELETE_MODE=full
#   After parsing each lambda, delete full lambda folders.
#
# DELETE_MODE=checkpoints
#   After parsing each lambda, delete only checkpoint files and keep logs.
#
# Recommended for your storage constraint: full
DELETE_MODE="${DELETE_MODE:-full}"

# Delete model checkpoint files immediately after each seed has been evaluated.
# This keeps only logs until parsing.
DELETE_CKPT_AFTER_EVAL="${DELETE_CKPT_AFTER_EVAL:-true}"

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
lambda_tag () {
    # 0.01 -> probe_lambda_0p01
    # 6.0  -> probe_lambda_6p0
    echo "probe_lambda_${1//./p}"
}

safe_rm_dir () {
    local target="$1"
    local lambda_tag_value="$2"

    if [[ -z "${target}" ]]; then
        echo "[ERROR] Empty deletion path refused."
        exit 1
    fi

    if [[ "${target}" != "${SWEEP_ROOT}/"* ]]; then
        echo "[ERROR] Unsafe deletion refused: ${target}"
        echo "[ERROR] Target is not inside SWEEP_ROOT=${SWEEP_ROOT}"
        exit 1
    fi

    if [[ "${target}" != *"${lambda_tag_value}"* ]]; then
        echo "[ERROR] Unsafe deletion refused: ${target}"
        echo "[ERROR] Target does not contain lambda tag: ${lambda_tag_value}"
        exit 1
    fi

    if [[ -d "${target}" ]]; then
        echo "[CLEANUP] Removing directory: ${target}"
        rm -rf "${target}"
    else
        echo "[CLEANUP] Directory not found, skipping: ${target}"
    fi
}

delete_checkpoint_files_only () {
    local target="$1"

    if [[ -d "${target}" ]]; then
        echo "[CLEANUP] Deleting checkpoint files inside: ${target}"

        find "${target}" -type f \( \
            -name "model*.pth.tar*" -o \
            -name "optimizer*.pth.tar*" -o \
            -name "optim*.pth.tar*" -o \
            -name "scheduler*.pth.tar*" -o \
            -name "sched*.pth.tar*" \
        \) -print -delete
    else
        echo "[CLEANUP] Directory not found, skipping checkpoint cleanup: ${target}"
    fi
}

cleanup_lambda_after_parse () {
    local lambda_tag_value="$1"

    local base_lambda_root="${SWEEP_ROOT}/train_base/${lambda_tag_value}"
    local novel_lambda_root="${SWEEP_ROOT}/test_new/${lambda_tag_value}"

    echo ""
    echo "================================================="
    echo "Final cleanup for ${lambda_tag_value}"
    echo "DELETE_MODE=${DELETE_MODE}"
    echo "================================================="

    if [[ "${DELETE_MODE}" == "full" ]]; then
        safe_rm_dir "${base_lambda_root}" "${lambda_tag_value}"
        safe_rm_dir "${novel_lambda_root}" "${lambda_tag_value}"

    elif [[ "${DELETE_MODE}" == "checkpoints" ]]; then
        delete_checkpoint_files_only "${base_lambda_root}"
        delete_checkpoint_files_only "${novel_lambda_root}"

    else
        echo "[ERROR] Unknown DELETE_MODE=${DELETE_MODE}"
        echo "Use DELETE_MODE=full or DELETE_MODE=checkpoints"
        exit 1
    fi
}

# ---------------------------------------------------------
# Master report header
# ---------------------------------------------------------
{
    echo "================================================="
    echo "PROBE_LAMBDA LOW-STORAGE SEQUENTIAL SWEEP"
    echo "Started at              : ${TIMESTAMP}"
    echo "GPU_ID                  : ${GPU_ID}"
    echo "CUDA_VISIBLE_DEVICES    : ${CUDA_VISIBLE_DEVICES}"
    echo "DATA                    : ${DATA}"
    echo "TRAINER                 : ${TRAINER}"
    echo "CFG                     : ${CFG}"
    echo "SHOTS                   : ${SHOTS}"
    echo "LOADEP                  : ${LOADEP}"
    echo "SWEEP_ROOT              : ${SWEEP_ROOT}"
    echo "REPORT_DIR              : ${REPORT_DIR}"
    echo "DELETE_MODE             : ${DELETE_MODE}"
    echo "DELETE_CKPT_AFTER_EVAL  : ${DELETE_CKPT_AFTER_EVAL}"
    echo "DATASETS                : ${DATASETS[*]}"
    echo "SEEDS                   : ${SEEDS[*]}"
    echo "LAMBDAS                 : ${LAMBDAS[*]}"
    echo "================================================="
    echo ""
} | tee -a "${MASTER_REPORT}"

# =========================================================
# Main lambda loop
# =========================================================
for LAMBDA in "${LAMBDAS[@]}"; do

    LAMBDA_TAG=$(lambda_tag "${LAMBDA}")

    # This file name directly references the lambda value.
    # Example:
    # avg_report_PROBE_LAMBDA_0.01_tag_probe_lambda_0p01_2026-06-10_12-30-00.txt
    LAMBDA_REPORT="${REPORT_DIR}/avg_report_PROBE_LAMBDA_${LAMBDA}_tag_${LAMBDA_TAG}_${TIMESTAMP}.txt"

    {
        echo ""
        echo "#################################################"
        echo "STARTING PROBE_LAMBDA=${LAMBDA}"
        echo "LAMBDA_TAG=${LAMBDA_TAG}"
        echo "AVERAGED_REPORT_FILE=${LAMBDA_REPORT}"
        echo "#################################################"
        echo ""
    } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

    # =====================================================
    # Train + evaluate seed-by-seed
    # This reduces storage because each seed checkpoint can
    # be removed immediately after novel evaluation.
    # =====================================================
    for DATASET in "${DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            COMMON_DIR="${LAMBDA_TAG}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}"

            TRAIN_DIR="${SWEEP_ROOT}/train_base/${COMMON_DIR}"
            TEST_DIR="${SWEEP_ROOT}/test_new/${COMMON_DIR}"

            # -------------------------------------------------
            # 1. Train base classes
            # -------------------------------------------------
            {
                echo ""
                echo "================================================="
                echo "TRAINING BASE"
                echo "Dataset      : ${DATASET}"
                echo "Seed         : ${SEED}"
                echo "PROBE_LAMBDA : ${LAMBDA}"
                echo "Train dir    : ${TRAIN_DIR}"
                echo "================================================="
                echo ""
            } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

            python train.py \
                --root "${DATA}" \
                --seed "${SEED}" \
                --trainer "${TRAINER}" \
                --dataset-config-file "configs/datasets/${DATASET}.yaml" \
                --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
                --output-dir "${TRAIN_DIR}" \
                DATASET.NUM_SHOTS "${SHOTS}" \
                DATASET.SUBSAMPLE_CLASSES base \
                TRAINER.HICROPLReason.PROBE_ENABLE False \
                TRAINER.HICROPLReason.PROBE_LAMBDA "${LAMBDA}" \
                TRAINER.HICROPLReason.DAPT_SAVE_PROTOTYPES False \
                TRAINER.HICROPLReason.DAPT_INTRA_ENABLE False

            # -------------------------------------------------
            # 2. Evaluate novel classes using this seed checkpoint
            # -------------------------------------------------
            {
                echo ""
                echo "================================================="
                echo "EVALUATING NOVEL"
                echo "Dataset      : ${DATASET}"
                echo "Seed         : ${SEED}"
                echo "PROBE_LAMBDA : ${LAMBDA}"
                echo "Model dir    : ${TRAIN_DIR}"
                echo "Test dir     : ${TEST_DIR}"
                echo "================================================="
                echo ""
            } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

            python train.py \
                --root "${DATA}" \
                --seed "${SEED}" \
                --trainer "${TRAINER}" \
                --dataset-config-file "configs/datasets/${DATASET}.yaml" \
                --config-file "configs/trainers/${TRAINER}/${CFG}.yaml" \
                --output-dir "${TEST_DIR}" \
                --model-dir "${TRAIN_DIR}" \
                --load-epoch "${LOADEP}" \
                --eval-only \
                DATASET.NUM_SHOTS "${SHOTS}" \
                DATASET.SUBSAMPLE_CLASSES new \
                TRAINER.HICROPLReason.PROBE_ENABLE False \
                TRAINER.HICROPLReason.PROBE_LAMBDA "${LAMBDA}" \
                TRAINER.HICROPLReason.DAPT_SAVE_PROTOTYPES False \
                TRAINER.HICROPLReason.DAPT_INTRA_ENABLE False

            # -------------------------------------------------
            # 3. Delete checkpoint files immediately after eval
            #    Keep logs for parse_test_res.py.
            # -------------------------------------------------
            if [[ "${DELETE_CKPT_AFTER_EVAL}" == "true" ]]; then
                {
                    echo ""
                    echo "================================================="
                    echo "IMMEDIATE CHECKPOINT CLEANUP AFTER EVAL"
                    echo "Dataset      : ${DATASET}"
                    echo "Seed         : ${SEED}"
                    echo "PROBE_LAMBDA : ${LAMBDA}"
                    echo "Keeping logs for averaging."
                    echo "================================================="
                    echo ""
                } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

                delete_checkpoint_files_only "${TRAIN_DIR}" | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"
                delete_checkpoint_files_only "${TEST_DIR}" | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"
            fi

        done
    done

    # =====================================================
    # 4. Parse averaged base and novel results for this lambda
    # =====================================================
    {
        echo ""
        echo "================================================="
        echo "PARSING AVERAGED RESULTS"
        echo "PROBE_LAMBDA=${LAMBDA}"
        echo "LAMBDA_TAG=${LAMBDA_TAG}"
        echo "Averaged report file: ${LAMBDA_REPORT}"
        echo "================================================="
        echo ""
    } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

    for DATASET in "${DATASETS[@]}"; do

        BASE_PARSE_DIR="${SWEEP_ROOT}/train_base/${LAMBDA_TAG}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}"
        NOVEL_PARSE_DIR="${SWEEP_ROOT}/test_new/${LAMBDA_TAG}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}"

        {
            echo ""
            echo "-------------------------------------------------"
            echo "Dataset: ${DATASET}"
            echo "PROBE_LAMBDA=${LAMBDA}"
            echo "-------------------------------------------------"
            echo ""
            echo "--- Base classes average ---"
        } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

        python parse_test_res.py "${BASE_PARSE_DIR}" \
            | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

        {
            echo ""
            echo "--- Novel classes average ---"
        } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

        python parse_test_res.py "${NOVEL_PARSE_DIR}" --test-log \
            | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

        {
            echo ""
            echo "-----------------------------"
        } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

    done

    {
        echo ""
        echo "================================================="
        echo "FINISHED PROBE_LAMBDA=${LAMBDA}"
        echo "Averaged report saved at:"
        echo "${LAMBDA_REPORT}"
        echo "================================================="
        echo ""
    } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

    # =====================================================
    # 5. Final cleanup for this lambda
    # =====================================================
    cleanup_lambda_after_parse "${LAMBDA_TAG}" | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

    {
        echo ""
        echo "================================================="
        echo "Storage cleanup completed for PROBE_LAMBDA=${LAMBDA}"
        echo "Moving to next lambda."
        echo "================================================="
        echo ""
    } | tee -a "${MASTER_REPORT}" "${LAMBDA_REPORT}"

done

{
    echo ""
    echo "================================================="
    echo "ALL PROBE_LAMBDA SWEEP RUNS COMPLETED"
    echo "Master report:"
    echo "${MASTER_REPORT}"
    echo "All per-lambda averaged reports are in:"
    echo "${REPORT_DIR}"
    echo "================================================="
} | tee -a "${MASTER_REPORT}"