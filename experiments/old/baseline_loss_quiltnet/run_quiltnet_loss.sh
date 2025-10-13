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
mkdir -p ../../output/quiltnet_loss
mkdir -p ../../output/quiltnet_loss/plots

# Run the experiments
echo "Starting QuiltNet loss experiments..."
python quiltnet_loss_experiments.py

# After completion, analyze results
echo "Analyzing results..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    results_file = '../../output/quiltnet_loss/quiltnet_loss_results.csv'
    if not os.path.exists(results_file):
        raise FileNotFoundError(f'Results file not found: {results_file}')
        
    # Read results
    results = pd.read_csv(results_file)
    
    if len(results) == 0:
        raise ValueError('No results found in the CSV file')
    
    # Create plots directory
    plots_dir = '../../output/quiltnet_loss/plots'
    os.makedirs(plots_dir, exist_ok=True)

    # For each dataset
    for dataset in results['dataset'].unique():
        df_dataset = results[results['dataset'] == dataset].copy()
        print(f'\nProcessing results for {dataset}...')
        
        # Create plot with multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Sort for better visualization
        df_dataset = df_dataset.sort_values('losses')
        
        # Plot Accuracy
        sns.barplot(data=df_dataset, x='losses', y='accuracy', ax=axes[0,0])
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45, ha='right')
        axes[0,0].set_title(f'{dataset}: Accuracy by Loss Function')
        
        # Plot ECE
        sns.barplot(data=df_dataset, x='losses', y='ece', ax=axes[0,1])
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45, ha='right')
        axes[0,1].set_title(f'{dataset}: ECE by Loss Function')
        
        # Plot Confidence
        sns.barplot(data=df_dataset, x='losses', y='confidence', ax=axes[1,0])
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45, ha='right')
        axes[1,0].set_title(f'{dataset}: Confidence by Loss Function')
        
        # Plot Macro F1
        sns.barplot(data=df_dataset, x='losses', y='macro_f1', ax=axes[1,1])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45, ha='right')
        axes[1,1].set_title(f'{dataset}: Macro F1 by Loss Function')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/{dataset}_quiltnet_loss_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print detailed results
        print(f'\nDetailed metrics for {dataset}:')
        metrics = ['losses', 'accuracy', 'confidence', 'ece', 'macro_f1']
        summary = df_dataset[metrics].sort_values(['ece'])
        print('\nSorted by ECE:')
        print(summary.to_string(float_format=lambda x: '{:.2f}'.format(x) if isinstance(x, float) else str(x), index=False))
        
        # Print best configurations
        print('\nBest by ECE:')
        best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
        print(f'Loss: {best_ece[\"losses\"]}')
        print(f'ECE: {best_ece[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_ece[\"accuracy\"]:.2f}%')
        print(f'Confidence: {best_ece[\"confidence\"]:.2f}%')
        print(f'Macro F1: {best_ece[\"macro_f1\"]:.2f}%')
        
        print('\nBest by Accuracy:')
        best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
        print(f'Loss: {best_acc[\"losses\"]}')
        print(f'ECE: {best_acc[\"ece\"]:.2f}%')
        print(f'Accuracy: {best_acc[\"accuracy\"]:.2f}%')
        print(f'Confidence: {best_acc[\"confidence\"]:.2f}%')
        print(f'Macro F1: {best_acc[\"macro_f1\"]:.2f}%')
        
        # Save dataset-specific results to CSV
        summary.to_csv(f'{plots_dir}/{dataset}_metrics_summary.csv', index=False)

except Exception as e:
    print(f'Error in analysis: {str(e)}')
    raise
"

echo "Results saved in ../../output/quiltnet_loss/"
echo "Plots saved in ../../output/quiltnet_loss/plots/"