#!/bin/bash
# Script to run MedCLIP and BioMedCLIP experiments in parallel across multiple GPUs

# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Define output directory
OUTPUT_DIR="../../output/medclip_biomedclip_baseline"
mkdir -p $OUTPUT_DIR

# Log file for experiment progress
LOG_FILE="$OUTPUT_DIR/experiment_progress.log"
echo "Starting parallel experiments at $(date)" > $LOG_FILE

# Define available GPUs
GPUS=(0 1 2)

# Function to run experiments with progress tracking
run_experiment() {
    model=$1
    dataset=$2
    seed=$3
    gpu_id=$4
    
    log_prefix="[GPU $gpu_id | $model | $dataset | seed $seed]"
    
    echo "$log_prefix Started at: $(date)" | tee -a $LOG_FILE
    
    # Set specific GPU for this experiment
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Run experiment with a single seed
    python baseline_prompt_medclip_biomedclip.py --model $model --dataset $dataset --seeds $seed
    
    echo "$log_prefix Finished at: $(date)" | tee -a $LOG_FILE
}

# Function to run experiments in parallel
run_parallel_experiments() {
    # Define experiment combinations
    declare -a experiments=(
        "medclip covid 1"
        "medclip rsna18 1"
        "biomedclip covid 1"
        "biomedclip rsna18 1"
    )
    
    echo "=== STARTING PARALLEL EXPERIMENTS ACROSS ${#GPUS[@]} GPUs ===" | tee -a $LOG_FILE
    
    # Launch experiments in parallel, distributing across GPUs
    pids=()
    for i in "${!experiments[@]}"; do
        # Determine which GPU to use for this experiment
        gpu_idx=$((i % ${#GPUS[@]}))
        gpu_id=${GPUS[$gpu_idx]}
        
        # Parse experiment details
        exp=(${experiments[$i]})
        model=${exp[0]}
        dataset=${exp[1]}
        seed=${exp[2]}
        
        # Run experiment in background
        run_experiment "$model" "$dataset" "$seed" "$gpu_id" &
        pids+=($!)
        
        # Small delay to avoid race conditions
        sleep 2
    done
    
    # Wait for all experiments to complete
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo "All parallel experiments completed at $(date)" | tee -a $LOG_FILE
}

# Run all experiments in parallel
run_parallel_experiments

# Generate combined results table
echo "Creating combined results summary..." | tee -a $LOG_FILE

# Python script to combine results
python - <<'EOF'
import pandas as pd
import os

# Load results
output_dir = "../../output/medclip_biomedclip_baseline"
models = ["medclip", "biomedclip"]
datasets = ["covid", "rsna18"]
combined_results = []

# Collect all summary results
for model in models:
    model_dir = f"{output_dir}/prompt_{model}"
    summary_path = os.path.join(model_dir, f"prompt_summary_{model}.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df["Model"] = model
        combined_results.append(df)

# Combine and save
if combined_results:
    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_path = os.path.join(output_dir, "combined_prompt_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to {combined_path}")
    
    # Create a comparison table showing various loss functions
    loss_functions = ["LS", "LS_MDCA", "LS_SLMDCA", "LS_SLMDCA_COSINE", "ECE_KDE"]
    comparison_df = combined_df[combined_df["Loss Function"].isin(loss_functions)]
    comparison_path = os.path.join(output_dir, "prompt_loss_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Loss function comparison saved to {comparison_path}")
else:
    print("No results found to combine")
EOF

echo "All done! Results are available in the output directory."