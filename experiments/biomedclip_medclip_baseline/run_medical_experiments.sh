#!/bin/bash
# Set the base directory
BASE_DIR="/home/abhishek/desktop/VLM_Cal/CalibPrompt"
EXPERIMENT_DIR="${BASE_DIR}/experiments/biomedclip_medclip_baseline"

# Create output directories
mkdir -p ${BASE_DIR}/output/baseline_prompt_medclip
mkdir -p ${BASE_DIR}/output/baseline_prompt_biomedclip

# Define a function to wait until specified GPU is available
wait_for_gpu() {
    local gpu_id=$1
    while true; do
        # Check GPU memory usage
        used_memory=$(nvidia-smi -i ${gpu_id} --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used_memory" -lt 10000 ]; then # Less than 10GB usage
            break
        fi
        echo "GPU ${gpu_id} is busy (${used_memory}MB used). Waiting..."
        sleep 30 # Wait 30 seconds before checking again
    done
}

# Function to run experiment on specified GPU
run_experiment() {
    local model=$1
    local dataset=$2
    local gpu_id=$3
    
    wait_for_gpu ${gpu_id}
    echo "Starting $model prompt learning on $dataset using GPU ${gpu_id} at $(date)"
    python ${EXPERIMENT_DIR}/baseline_experiments.py --model $model --dataset $dataset --gpu ${gpu_id}
    
    echo "Completed $model prompt learning on $dataset at $(date)"
}

echo "Starting comprehensive medical prompt learning experiments at $(date)"
echo "Running experiments in parallel across GPUs 0, 1, and 2..."

# Run experiments in parallel on different GPUs
run_experiment medclip rsna18 0 &
run_experiment medclip covid 1 &
run_experiment biomedclip rsna18 2 &
wait  # Wait for the first batch to complete

run_experiment biomedclip covid 0 &
wait  # Wait for the second batch to complete

echo "All medical prompt learning experiments completed at $(date)!"

# Print summary
echo "Experiment Summary:"
echo "==================="
echo "Medical Prompt Learning with all loss function configurations"
echo ""
echo "Loss Function Combinations:"
echo "1. CE (Cross Entropy only)"
echo "2. CE + DCA (DCA weight 9.0)"
echo "3. CE + MMCE (MMCE weight 2.0)"
echo "4. CE + MDCA (MDCA weight 1.0)"
echo "5. CE + SMAC (SMAC alpha 0.1)"
echo "6. CE + SMAC + AS (SMAC alpha 0.1, AS weight 3.0)"
echo "7. FL (Focal Loss only, gamma 3.0)"
echo "8. FL + MDCA (FL gamma 3.0, MDCA weight 1.0)"
echo "9. FL + SMAC (FL gamma 3.0, SMAC alpha 0.1)"
echo "10. FL + SMAC + AS (FL gamma 3.0, SMAC alpha 0.1, AS weight 3.0)"
echo "11. LS (Label Smoothing alpha 0.2)"
echo "12. LS + MDCA (LS alpha 0.2, MDCA weight 1.0)"
echo "13. LS + SMAC (LS alpha 0.2, SMAC alpha varies by model)"
echo "14. LS + SMAC + AS (LS alpha 0.2, SMAC alpha varies, AS weight varies)"
echo ""
echo "Results can be found in: ${BASE_DIR}/output/baseline_prompt_*/"

# Check for completed experiments
for model in "medclip" "biomedclip"; do
    for dataset in "rsna18" "covid"; do
        directory="${BASE_DIR}/output/baseline_prompt_${model}/${dataset}_${model}_prompt_results.csv"
        if [ -f "$directory" ]; then
            echo "✓ ${model} on ${dataset} completed successfully"
        else
            echo "✗ ${model} on ${dataset} FAILED or not completed"
        fi
    done
done