#!/bin/bash
# Create output directories
mkdir -p ../../output/cosine_experiments_plip
mkdir -p ../../output/cosine_experiments_quiltnet

# Define a function to wait until a GPU is available
wait_for_gpu() {
    local gpu=$1
    while true; do
        # Check GPU memory usage
        used_memory=$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used_memory" -lt 1000 ]; then # Less than 1GB usage
            break
        fi
        sleep 10 # Wait 10 seconds before checking again
    done
}

# Function to run experiment and handle GPU allocation
run_experiment() {
    local gpu=$1
    local model=$2
    local dataset=$3
    wait_for_gpu $gpu
    echo "Starting $model on $dataset using GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python cosine_experiments.py --model $model --dataset $dataset
}

# Run all experiments truly in parallel
run_experiment 0 plip kather &
run_experiment 0 plip pannuke &
run_experiment 1 plip digestpath &
run_experiment 1 quiltnet kather &
run_experiment 2 quiltnet pannuke &
run_experiment 2 quiltnet digestpath &

# Wait for all processes to complete
wait

echo "All experiments completed!"