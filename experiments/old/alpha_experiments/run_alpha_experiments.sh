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

# Create base directories
mkdir -p ../../output/alpha_experiments_plip
mkdir -p ../../output/alpha_experiments_quiltnet

# Run PLIP experiments
echo "Starting PLIP experiments..."
python alpha_experiments.py --model plip

# Run QuiltNet experiments
echo "Starting QuiltNet experiments..."
python alpha_experiments.py --model quiltnet

echo "All experiments completed!"
echo "Results saved in:"
echo "- ../../output/alpha_experiments_plip/plip_results.csv"
echo "- ../../output/alpha_experiments_quiltnet/quiltnet_results.csv"