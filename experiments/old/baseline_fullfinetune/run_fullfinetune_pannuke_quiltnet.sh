#!/bin/bash

# Simple script to run QuiltNet experiments on PanNuke dataset using GPU 0

echo "Starting QuiltNet full fine-tuning on PanNuke dataset"

# Run the command with GPU 0
CUDA_VISIBLE_DEVICES=0 python fullfinetune.py --model quiltnet --dataset pannuke

echo "Completed experiment"