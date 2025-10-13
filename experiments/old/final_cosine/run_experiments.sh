#!/bin/bash

# Create output directories
mkdir -p experiments/final_cosine/output/cosine_experiments_plip
mkdir -p experiments/final_cosine/output/cosine_experiments_quiltnet

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
    echo "Starting $model on $dataset using GPU $gpu at $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python final_cosine_experiments.py --model $model --dataset $dataset
}

echo "Starting experiments at $(date)"

# PLIP Experiments
echo "Starting PLIP experiments..."
run_experiment 0 plip kather &   # Kather with alpha=0.07/0.03/0.05
run_experiment 0 plip pannuke &  # PanNuke with alpha=0.2
run_experiment 1 plip digestpath & # DigestPath with alpha=0.03

# QUILTNET Experiments
echo "Starting QUILTNET experiments..."
run_experiment 1 quiltnet kather &   # Kather with alpha=0.07/0.03/0.01
run_experiment 2 quiltnet pannuke &  # PanNuke with alpha=0.1
run_experiment 2 quiltnet digestpath & # DigestPath with alpha=0.05

# Wait for all processes to complete
wait
echo "All experiments completed at $(date)!"

# Print summary
echo "Experiment Summary:"
echo "==================="
echo "PLIP Experiments:"
echo "- Kather: CE+SLMDCA(0.07), FL+SLMDCA(0.03), LS+SLMDCA(0.05)"
echo "- PanNuke: CE+SLMDCA(0.2), FL+SLMDCA(0.2), LS+SLMDCA(0.2)"
echo "- DigestPath: CE+SLMDCA(0.03), FL+SLMDCA(0.03), LS+SLMDCA(0.03)"
echo ""
echo "QUILTNET Experiments:"
echo "- Kather: CE+SLMDCA(0.07), FL+SLMDCA(0.03), LS+SLMDCA(0.01)"
echo "- PanNuke: CE+SLMDCA(0.1), FL+SLMDCA(0.1), LS+SLMDCA(0.1)"
echo "- DigestPath: CE+SLMDCA(0.05), FL+SLMDCA(0.05), LS+SLMDCA(0.05)"
echo ""
echo "Results can be found in: experiments/final_cosine/output/"