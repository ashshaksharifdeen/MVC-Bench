#!/bin/bash

# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Define GPUs to use
GPUS=(0 1 2)
gpu_count=${#GPUS[@]}
counter=0

# Create logs directory
mkdir -p logs

echo "Starting model and dataset experiments at $(date)"
echo "Using GPUs: ${GPUS[*]}"

# Loop over models and datasets and run experiments concurrently
for model in medclip biomedclip; do
  for dataset in covid rsna18; do
    # Pick a GPU from the list (round-robin)
    gpu=${GPUS[$(( counter % gpu_count ))]}
    echo "Starting experiment for model: $model, dataset: $dataset on GPU: $gpu at $(date)"
    
    # Launch the experiment in the background with output logged to a unique file.
    CUDA_VISIBLE_DEVICES=$gpu python cosine_experiments_medclip_biomedclip.py --model "$model" --dataset "$dataset" 2>&1 | tee -a "logs/${model}_${dataset}_cosine_experiments.log" &
    
    counter=$(( counter + 1 ))
  done
done

# Wait for all experiments to finish
wait
echo "All experiments completed at $(date)"

# Generate combined results report
python - <<'EOF'
import pandas as pd
import os
import numpy as np

# Directory for output
output_dir = "../../output"
models = ["medclip", "biomedclip"]
datasets = ["covid", "rsna18"]
loss_types = ["CE", "FL", "LS"]

# Create directory for combined results
combined_dir = os.path.join(output_dir, "final_cosine_experiments_combined")
os.makedirs(combined_dir, exist_ok=True)

# Combine results from all experiments
combined_results = []

for model in models:
    for dataset in datasets:
        # Assuming each experiment writes a CSV file for each (model, dataset)
        results_file = os.path.join(output_dir, f"final_cosine_experiments_{model}/{dataset}_{model}_results.csv")
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            df["model"] = model
            df["dataset"] = dataset
            combined_results.append(df)

if combined_results:
    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_path = os.path.join(combined_dir, "all_cosine_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to {combined_path}")
    
    # Create summary of best cosine weights
    summary_rows = []
    
    for model in models:
        for dataset in datasets:
            for loss_type in loss_types:
                # Filter for this model, dataset and loss type
                filtered = combined_df[(combined_df["model"] == model) & 
                                       (combined_df["dataset"] == dataset) & 
                                       (combined_df["losses"].str.startswith(loss_type))]
                
                if not filtered.empty:
                    # Group by loss (which includes cosine weight)
                    grouped = filtered.groupby("losses")
                    
                    # Get mean metrics for each loss configuration
                    means = grouped.agg({
                        "accuracy": "mean",
                        "ece": "mean",
                        "mce": "mean",
                        "ace": "mean"
                    }).reset_index()
                    
                    # Find best configurations
                    best_ece_row = means.loc[means["ece"].idxmin()]
                    best_acc_row = means.loc[means["accuracy"].idxmax()]
                    
                    # Extract cosine weight from loss name (assumes a pattern like 'CE_COSINE_0.5')
                    best_ece_cosine = float(best_ece_row["losses"].split("COSINE_")[1])
                    best_acc_cosine = float(best_acc_row["losses"].split("COSINE_")[1])
                    
                    summary_rows.append({
                        "model": model,
                        "dataset": dataset,
                        "loss_type": loss_type,
                        "best_ece_cosine": best_ece_cosine,
                        "best_ece": best_ece_row["ece"],
                        "accuracy_at_best_ece": best_ece_row["accuracy"],
                        "best_acc_cosine": best_acc_cosine,
                        "best_accuracy": best_acc_row["accuracy"],
                        "ece_at_best_acc": best_acc_row["ece"]
                    })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(combined_dir, "best_cosine_weights_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Best cosine weights summary saved to {summary_path}")
        
        # Create comparison table for use in papers/reports
        table_path = os.path.join(combined_dir, "cosine_weights_comparison_table.md")
        with open(table_path, "w") as f:
            f.write("# Cosine Weight Comparison\n\n")
            
            for model in models:
                f.write(f"## Model: {model}\n\n")
                
                for dataset in datasets:
                    f.write(f"### Dataset: {dataset}\n\n")
                    f.write("| Loss Type | Best Cosine (ECE) | ECE | Accuracy | Best Cosine (Acc) | Accuracy | ECE |\n")
                    f.write("|-----------|------------------|-----|----------|-------------------|----------|-----|\n")
                    
                    model_rows = [row for row in summary_rows if row["model"] == model and row["dataset"] == dataset]
                    for row in model_rows:
                        f.write(f"| {row['loss_type']} | {row['best_ece_cosine']} | {row['best_ece']:.2f} | {row['accuracy_at_best_ece']:.2f} | ")
                        f.write(f"{row['best_acc_cosine']} | {row['best_accuracy']:.2f} | {row['ece_at_best_acc']:.2f} |\n")
                    
                    f.write("\n")
        
        print(f"Comparison table saved to {table_path}")
else:
    print("No results found to combine")
EOF

echo "Results analysis completed"
echo "Check the output directory for detailed results and visualizations"
