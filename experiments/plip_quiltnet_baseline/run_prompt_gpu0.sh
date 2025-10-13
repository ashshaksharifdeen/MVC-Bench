#!/bin/bash

# Create output directories
mkdir -p ../../output/baseline_prompt_plip
mkdir -p ../../output/baseline_prompt_quiltnet

# Define a function to wait until GPU 0 is available
wait_for_gpu() {
    while true; do
        # Check GPU memory usage
        used_memory=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used_memory" -lt 10000 ]; then # Less than 1GB usage
            break
        fi
        sleep 10 # Wait 10 seconds before checking again
    done
}

# Function to run experiment on GPU 0
run_experiment() {
    local model=$1
    local dataset=$2
    
    wait_for_gpu
    echo "Starting $model prompt learning on $dataset using GPU 0 at $(date)"
    CUDA_VISIBLE_DEVICES=0 python ./baseline_prompt_experiments.py --model $model --dataset $dataset
    
    # No need for & since we're running sequentially
    echo "Completed $model prompt learning on $dataset at $(date)"
}

echo "Starting prompt learning experiments on GPU 0 at $(date)"

echo "Running experiments sequentially on GPU 0..."

# PLIP Experiments
run_experiment plip kather
run_experiment plip pannuke
run_experiment plip digestpath

# QUILTNET Experiments
run_experiment quiltnet kather
run_experiment quiltnet pannuke
run_experiment quiltnet digestpath

echo "All prompt learning experiments completed at $(date)!"

# Print summary
echo "Experiment Summary:"
echo "==================="
echo "Prompt Learning on GPU 0 only"
echo ""
echo "PLIP Experiments:"
echo "- Kather: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo "- PanNuke: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo "- DigestPath: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo ""
echo "QUILTNET Experiments:"
echo "- Kather: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo "- PanNuke: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo "- DigestPath: CE, CE+DCA, CE+MMCE, CE+MDCA, CE+SLMDCA, CE+SLMDCA+CL, FL, FL+MDCA, FL+SLMDCA, FL+SLMDCA+CL, LS, LS+MDCA, LS+SLMDCA, LS+SLMDCA+CL"
echo ""
echo "Results can be found in: ../../output/baseline_prompt_*/"