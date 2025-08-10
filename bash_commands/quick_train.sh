#!/bin/bash

# Quick training script for histo-bench
# Simple and straightforward

# Set your paths here
METADATA_PATH="example_data/TCGA-LGG/training_metadata.csv"
FEATURE_DIR="data/features/uni/TCGA/TCGA-LGG"
EXPERIMENT_NAME="quick_test"

# Quick test command
echo "🚀 Running quick test..."
python train_mil.py \
    --metadata_path "$METADATA_PATH" \
    --feature_dir "$FEATURE_DIR" \
    --num_of_classes 3 \
    --feature_dim 1536 \
    --experiment_name "$EXPERIMENT_NAME" \
    --num_epochs 2 \
    --batch_size 1 \
    --learning_rate 0.0005 \
    --model_name "MeanPooling"
