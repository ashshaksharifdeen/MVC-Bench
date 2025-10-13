#!/bin/bash

# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Get GPU ID from command line or default to 0
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create logs directory
mkdir -p logs

echo "Starting alpha parameter experiments at $(date)"
echo "Using GPU: $GPU_ID"

# Run MedCLIP experiments
echo "=== Running MedCLIP alpha experiments ==="
echo "Starting at $(date)" > logs/medclip_alpha_experiments.log
python alpha_experiments_medclip_biomedclip.py --model medclip 2>&1 | tee -a logs/medclip_alpha_experiments.log
echo "Completed at $(date)" >> logs/medclip_alpha_experiments.log

# Run BioMedCLIP experiments
echo "=== Running BioMedCLIP alpha experiments ==="
echo "Starting at $(date)" > logs/biomedclip_alpha_experiments.log
python alpha_experiments_medclip_biomedclip.py --model biomedclip 2>&1 | tee -a logs/biomedclip_alpha_experiments.log
echo "Completed at $(date)" >> logs/biomedclip_alpha_experiments.log

echo "All experiments completed at $(date)"

# Generate combined results report
python - <<EOF
import pandas as pd
import os
import numpy as np

# Directory for output
output_dir = "../../output"
models = ["medclip", "biomedclip"]
datasets = ["covid", "rsna18"]
loss_configs = ["CE_SLMDCA", "FL_SLMDCA", "LS_SLMDCA"]

# Create directory for combined results
combined_dir = os.path.join(output_dir, "alpha_experiments_combined")
os.makedirs(combined_dir, exist_ok=True)

# Combine results from all experiments
combined_results = []

for model in models:
    results_file = os.path.join(output_dir, f"alpha_experiments_{model}/{model}_results.csv")
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        df["model"] = model
        combined_results.append(df)

if combined_results:
    combined_df = pd.concat(combined_results, ignore_index=True)
    combined_path = os.path.join(combined_dir, "all_alpha_results.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined results saved to {combined_path}")
    
    # Create summary of best alpha values
    summary_rows = []
    
    for model in models:
        model_results = combined_df[combined_df["model"] == model]
        
        for dataset in datasets:
            dataset_results = model_results[model_results["dataset"] == dataset]
            
            for loss_name in loss_configs:
                loss_results = dataset_results[dataset_results["losses"] == loss_name]
                
                if not loss_results.empty:
                    # Find best alpha for ECE
                    best_ece_idx = loss_results["ece"].idxmin()
                    best_ece_row = loss_results.loc[best_ece_idx]
                    
                    # Find best alpha for accuracy
                    best_acc_idx = loss_results["accuracy"].idxmax()
                    best_acc_row = loss_results.loc[best_acc_idx]
                    
                    summary_rows.append({
                        "model": model,
                        "dataset": dataset,
                        "loss": loss_name,
                        "best_ece_alpha": best_ece_row["alpha"],
                        "best_ece": best_ece_row["ece"],
                        "accuracy_at_best_ece": best_ece_row["accuracy"],
                        "best_acc_alpha": best_acc_row["alpha"],
                        "best_accuracy": best_acc_row["accuracy"],
                        "ece_at_best_acc": best_acc_row["ece"]
                    })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(combined_dir, "best_alpha_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Best alpha summary saved to {summary_path}")
    
    # Create comparison table for use in papers/reports
    table_path = os.path.join(combined_dir, "alpha_comparison_table.md")
    with open(table_path, "w") as f:
        f.write("# Alpha Parameter Comparison\n\n")
        
        for dataset in datasets:
            f.write(f"## Dataset: {dataset}\n\n")
            f.write("| Model | Loss Function | Best Alpha (ECE) | ECE | Accuracy | Best Alpha (Acc) | Accuracy | ECE |\n")
            f.write("|-------|--------------|-----------------|-----|----------|-----------------|----------|-----|\n")
            
            for model in models:
                model_rows = [row for row in summary_rows if row["model"] == model and row["dataset"] == dataset]
                for row in model_rows:
                    f.write(f"| {row['model']} | {row['loss']} | {row['best_ece_alpha']} | {row['best_ece']:.2f} | {row['accuracy_at_best_ece']:.2f} | {row['best_acc_alpha']} | {row['best_accuracy']:.2f} | {row['ece_at_best_acc']:.2f} |\n")
            
            f.write("\n")
    
    print(f"Comparison table saved to {table_path}")
else:
    print("No results found to combine")
EOF

echo "Results analysis completed"
echo "Check the output directory for detailed results and visualizations"