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
mkdir -p ../output/hyperparam_search_phase2

# Run the hyperparameter search
echo "Starting Phase 2 experiments..."
python hyperparam_search_phase2.py

# After completion, analyze results
echo "Analyzing results..."
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read results
results = pd.read_csv('../output/hyperparam_search_phase2/phase2_results.csv')

# Create plots directory
os.makedirs('../output/hyperparam_search_phase2/plots', exist_ok=True)

# For each dataset
for dataset in results['dataset'].unique():
    df_dataset = results[results['dataset'] == dataset]
    
    # Plot ECE comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_dataset, x='losses', y='ece')
    plt.xticks(rotation=45)
    plt.xlabel('Loss Function')
    plt.ylabel('ECE (%)')
    plt.title(f'{dataset}: ECE vs Loss Function')
    plt.tight_layout()
    plt.savefig(f'../output/hyperparam_search_phase2/plots/{dataset}_ece_comparison.png')
    plt.close()
    
    # Plot Accuracy comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_dataset, x='losses', y='accuracy')
    plt.xticks(rotation=45)
    plt.xlabel('Loss Function')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset}: Accuracy vs Loss Function')
    plt.tight_layout()
    plt.savefig(f'../output/hyperparam_search_phase2/plots/{dataset}_accuracy_comparison.png')
    plt.close()
    
    # Print improvement over baseline
    print(f'\nResults for {dataset}:')
    baseline = df_dataset[df_dataset['losses'] == 'CE'].iloc[0]
    print('\nBaseline (CE only):')
    print(f'ECE: {baseline[\"ece\"]}%')
    print(f'Accuracy: {baseline[\"accuracy\"]}%')
    
    for loss in ['CE_DCA', 'CE_MDCA']:
        if loss in df_dataset['losses'].values:
            result = df_dataset[df_dataset['losses'] == loss].iloc[0]
            print(f'\n{loss}:')
            print(f'ECE: {result[\"ece\"]}% ({(result[\"ece\"]-baseline[\"ece\"]):.2f}% change)')
            print(f'Accuracy: {result[\"accuracy\"]}% ({(result[\"accuracy\"]-baseline[\"accuracy\"]):.2f}% change)')
"

echo "Results saved in output/hyperparam_search_phase2/"
echo "Plots saved in output/hyperparam_search_phase2/plots/"