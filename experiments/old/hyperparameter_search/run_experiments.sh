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
mkdir -p ../output/hyperparam_search

# Run the hyperparameter search
echo "Starting hyperparameter search..."
python hyperparam_search.py

# After completion, analyze results
echo "Analyzing results..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read results
results = pd.read_csv('../output/hyperparam_search/phase1_results.csv')

# Create plots directory
os.makedirs('../output/hyperparam_search/plots', exist_ok=True)

# For each dataset
for dataset in results['dataset'].unique():
    df_dataset = results[results['dataset'] == dataset]
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_dataset, x='lr', y='ece', marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('ECE (%)')
    plt.title(f'{dataset}: ECE vs Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../output/hyperparam_search/plots/{dataset}_ece_vs_lr.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_dataset, x='lr', y='accuracy', marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset}: Accuracy vs Learning Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'../output/hyperparam_search/plots/{dataset}_accuracy_vs_lr.png')
    plt.close()
    
    # Print dataset summary
    print(f'\nResults for {dataset}:')
    print(df_dataset.sort_values('lr')[['lr', 'accuracy', 'ece']])
    
    # Print best configurations
    best_ece = df_dataset.loc[df_dataset['ece'].idxmin()]
    best_acc = df_dataset.loc[df_dataset['accuracy'].idxmax()]
    
    print(f'\nBest configuration for {dataset} by ECE:')
    print(f'Learning Rate: {best_ece[\"lr\"]}')
    print(f'ECE: {best_ece[\"ece\"]}%')
    print(f'Accuracy: {best_ece[\"accuracy\"]}%')
    
    print(f'\nBest configuration for {dataset} by Accuracy:')
    print(f'Learning Rate: {best_acc[\"lr\"]}')
    print(f'ECE: {best_acc[\"ece\"]}%')
    print(f'Accuracy: {best_acc[\"accuracy\"]}%')
"

echo "Results saved in output/hyperparam_search/"
echo "Plots saved in output/hyperparam_search/plots/"