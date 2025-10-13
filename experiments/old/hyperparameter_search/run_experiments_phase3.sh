#!/bin/bash

# Check if GPU ID is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "Example: $0 0"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$1

# Create results directory if it doesn't exist
mkdir -p ../output/hyperparam_search_phase3

# Run the hyperparameter search
echo "Starting Phase 3 experiments..."
python hyperparam_search_phase3.py

# After completion, analyze results
echo "Analyzing results..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

try:
    # Read results
    results = pd.read_csv('../output/hyperparam_search_phase3/phase3_results.csv')
    
    # Create plots directory
    plots_dir = '../output/hyperparam_search_phase3/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # For each dataset
    for dataset in results['dataset'].unique():
        df_dataset = results[results['dataset'] == dataset]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Accuracy
        for loss in ['CE', 'CE_MDCA']:
            data = df_dataset[df_dataset['losses'] == loss]
            if not data.empty:
                ax1.plot(data['shots'], data['accuracy'], 'o-', label=loss)
        
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Number of Shots')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'{dataset}: Accuracy vs Shots')
        ax1.grid(True)
        ax1.legend()
        
        # Plot ECE
        for loss in ['CE', 'CE_MDCA']:
            data = df_dataset[df_dataset['losses'] == loss]
            if not data.empty:
                ax2.plot(data['shots'], data['ece'], 'o-', label=loss)
        
        ax2.set_xscale('log', base=2)
        ax2.set_xlabel('Number of Shots')
        ax2.set_ylabel('ECE (%)')
        ax2.set_title(f'{dataset}: ECE vs Shots')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{dataset}_metrics.png')
        plt.close()
        
        # Print summary table
        print(f'\nResults for {dataset}:')
        print('\nShot progression:')
        summary = []
        for shots in sorted(df_dataset['shots'].unique()):
            ce_data = df_dataset[(df_dataset['shots'] == shots) & (df_dataset['losses'] == 'CE')].iloc[0]
            mdca_data = df_dataset[(df_dataset['shots'] == shots) & (df_dataset['losses'] == 'CE_MDCA')].iloc[0]
            
            summary.append({
                'Shots': shots,
                'CE_Acc': ce_data['accuracy'],
                'MDCA_Acc': mdca_data['accuracy'],
                'Acc_Diff': mdca_data['accuracy'] - ce_data['accuracy'],  # Positive means MDCA is better
                'CE_ECE': ce_data['ece'],
                'MDCA_ECE': mdca_data['ece'],
                'ECE_Reduction': mdca_data['ece'] - ce_data['ece']  # Negative means MDCA reduces ECE
            })
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(float_format=lambda x: f'{x:.2f}'))
        
        # Calculate key statistics
        print(f'\nKey Statistics for {dataset}:')
        ece_reduction = -1 * summary_df['ECE_Reduction'].mean()  # Multiply by -1 so positive means improvement
        acc_change = summary_df['Acc_Diff'].mean()
        
        print(f'Average ECE Reduction with MDCA: {ece_reduction:.2f}% (positive means MDCA improves calibration)')
        print(f'Average Accuracy Change with MDCA: {acc_change:.2f}% (positive means MDCA improves accuracy)')
        
        # Find best shot count for each metric
        best_ce_ece = df_dataset[df_dataset['losses'] == 'CE']['ece'].min()
        best_mdca_ece = df_dataset[df_dataset['losses'] == 'CE_MDCA']['ece'].min()
        best_ce_acc = df_dataset[df_dataset['losses'] == 'CE']['accuracy'].max()
        best_mdca_acc = df_dataset[df_dataset['losses'] == 'CE_MDCA']['accuracy'].max()
        
        print(f'\nBest Results:')
        print(f'Best CE ECE: {best_ce_ece:.2f}%')
        print(f'Best MDCA ECE: {best_mdca_ece:.2f}%')
        print(f'Best CE Accuracy: {best_ce_acc:.2f}%')
        print(f'Best MDCA Accuracy: {best_mdca_acc:.2f}%')

except Exception as e:
    print(f'Error in analysis: {str(e)}')
"

echo "Results saved in output/hyperparam_search_phase3/"
echo "Plots saved in output/hyperparam_search_phase3/plots/"