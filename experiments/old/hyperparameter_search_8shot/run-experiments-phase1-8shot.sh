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
mkdir -p ../output/hyperparam_search_8shot

# Run the hyperparameter search
echo "Starting comparison experiments with 8 shots..."
python hyperparam_search_phase1_8shot.py

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
    results = pd.read_csv('../output/hyperparam_search_8shot/results.csv')
    
    # Create plots directory
    plots_dir = '../output/hyperparam_search_8shot/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # For each dataset
    for dataset in results['dataset'].unique():
        df_dataset = results[results['dataset'] == dataset]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot Accuracy vs Learning Rate for both methods
        for loss in ['CE', 'CE_MDCA']:
            data = df_dataset[df_dataset['losses'] == loss]
            ax1.semilogx(data['lr'], data['accuracy'], 'o-', label=loss)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'{dataset}: Accuracy vs Learning Rate')
        ax1.grid(True)
        ax1.legend()
        
        # Plot ECE vs Learning Rate for both methods
        for loss in ['CE', 'CE_MDCA']:
            data = df_dataset[df_dataset['losses'] == loss]
            ax2.semilogx(data['lr'], data['ece'], 'o-', label=loss)
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('ECE (%)')
        ax2.set_title(f'{dataset}: ECE vs Learning Rate')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{dataset}_comparison.png')
        plt.close()
        
        # Print detailed results
        print(f'\nResults for {dataset}:')
        print('\nCE Results:')
        ce_results = df_dataset[df_dataset['losses'] == 'CE'][['lr', 'accuracy', 'ece']].sort_values('lr')
        print(ce_results.to_string(float_format=lambda x: f'{x:.2f}'))
        
        print('\nCE+MDCA Results:')
        mdca_results = df_dataset[df_dataset['losses'] == 'CE_MDCA'][['lr', 'accuracy', 'ece']].sort_values('lr')
        print(mdca_results.to_string(float_format=lambda x: f'{x:.2f}'))
        
        # Find best configurations
        print('\nBest Configurations:')
        
        # For CE
        ce_data = df_dataset[df_dataset['losses'] == 'CE']
        best_ce_ece = ce_data.loc[ce_data['ece'].idxmin()]
        best_ce_acc = ce_data.loc[ce_data['accuracy'].idxmax()]
        
        print('\nCE Best ECE:')
        print(f'LR: {best_ce_ece[\"lr\"]:.3f}')
        print(f'ECE: {best_ce_ece[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_ce_ece[\"accuracy\"]:.2f}%')
        
        print('\nCE Best Accuracy:')
        print(f'LR: {best_ce_acc[\"lr\"]:.3f}')
        print(f'ECE: {best_ce_acc[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_ce_acc[\"accuracy\"]:.2f}%')
        
        # For CE+MDCA
        mdca_data = df_dataset[df_dataset['losses'] == 'CE_MDCA']
        best_mdca_ece = mdca_data.loc[mdca_data['ece'].idxmin()]
        best_mdca_acc = mdca_data.loc[mdca_data['accuracy'].idxmax()]
        
        print('\nCE+MDCA Best ECE:')
        print(f'LR: {best_mdca_ece[\"lr\"]:.3f}')
        print(f'ECE: {best_mdca_ece[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_mdca_ece[\"accuracy\"]:.2f}%')
        
        print('\nCE+MDCA Best Accuracy:')
        print(f'LR: {best_mdca_acc[\"lr\"]:.3f}')
        print(f'ECE: {best_mdca_acc[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_mdca_acc[\"accuracy\"]:.2f}%')

except Exception as e:
    print(f'Error in analysis: {str(e)}')
"

echo "Results saved in output/hyperparam_search_8shot/"
echo "Plots saved in output/hyperparam_search_8shot/plots/"