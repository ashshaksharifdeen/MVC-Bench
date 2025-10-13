#!/bin/bash

# Exit on error
set -e

# Check if GPU ID is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 GPU_ID"
    echo "Example: $0 0"
    exit 1
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$1

# Create results directory if it doesn't exist
mkdir -p ../../output/baseline_dca_weight
mkdir -p ../../output/baseline_dca_weight/plots

# Run the experiments
echo "Starting DCA weight experiments..."
python baseline_dca_weight_experiments.py

# After completion, analyze results
echo "Analyzing results..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    results_file = '../../output/baseline_dca_weight/dca_weight_results.csv'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f'Results file not found: {results_file}')
        
    results = pd.read_csv(results_file)
    
    if len(results) == 0:
        raise ValueError('No results found in the CSV file')
    
    plots_dir = '../../output/baseline_dca_weight/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # For each dataset
    for dataset in results['dataset'].unique():
        df_dataset = results[results['dataset'] == dataset].copy()
        print(f'\nProcessing results for {dataset}...')
        
        # Create line plots for trends
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Sort by DCA weight for line plots
        df_dataset = df_dataset.sort_values('dca_weight')
        
        # Plot Accuracy trend
        sns.lineplot(data=df_dataset, x='dca_weight', y='accuracy', ax=ax1, marker='o')
        ax1.set_title(f'{dataset}: Accuracy vs DCA Weight')
        ax1.set_xlabel('DCA Weight')
        ax1.set_ylabel('Accuracy (%)')
        
        # Plot ECE trend
        sns.lineplot(data=df_dataset, x='dca_weight', y='ece', ax=ax2, marker='o')
        ax2.set_title(f'{dataset}: ECE vs DCA Weight')
        ax2.set_xlabel('DCA Weight')
        ax2.set_ylabel('ECE (%)')
        
        # Plot Confidence trend
        sns.lineplot(data=df_dataset, x='dca_weight', y='confidence', ax=ax3, marker='o')
        ax3.set_title(f'{dataset}: Confidence vs DCA Weight')
        ax3.set_xlabel('DCA Weight')
        ax3.set_ylabel('Confidence (%)')
        
        # Plot Macro F1 trend
        sns.lineplot(data=df_dataset, x='dca_weight', y='macro_f1', ax=ax4, marker='o')
        ax4.set_title(f'{dataset}: Macro F1 vs DCA Weight')
        ax4.set_xlabel('DCA Weight')
        ax4.set_ylabel('Macro F1 (%)')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{dataset}_dca_weight_trends.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print detailed results
        print(f'\nDetailed metrics for {dataset}:')
        metrics = ['dca_weight', 'accuracy', 'confidence', 'ece', 'macro_f1']
        summary = df_dataset[metrics].sort_values(['ece'])
        print('\nSorted by ECE:')
        print(summary.to_string(float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else str(x), index=False))
        
        # Find best configurations
        best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
        print(f'\nBest by ECE:')
        print(f'DCA Weight: {best_ece[\"dca_weight\"]:.1f}')
        print(f'ECE: {best_ece[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_ece[\"accuracy\"]:.2f}%')
        print(f'Confidence: {best_ece[\"confidence\"]:.2f}%')
        print(f'Macro F1: {best_ece[\"macro_f1\"]:.2f}%')
        
        best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
        print(f'\nBest by Accuracy:')
        print(f'DCA Weight: {best_acc[\"dca_weight\"]:.1f}')
        print(f'ECE: {best_acc[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_acc[\"accuracy\"]:.2f}%')
        print(f'Confidence: {best_acc[\"confidence\"]:.2f}%')
        print(f'Macro F1: {best_acc[\"macro_f1\"]:.2f}%')
        
        # Save dataset-specific results to CSV
        summary.to_csv(f'{plots_dir}/{dataset}_dca_weight_summary.csv', index=False)

except Exception as e:
    print(f'Error in analysis: {str(e)}')
    raise
"

echo "Results saved in ../../output/baseline_dca_weight/"
echo "Plots saved in ../../output/baseline_dca_weight/plots/"