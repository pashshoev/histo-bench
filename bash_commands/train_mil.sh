#!/bin/bash

# Train MIL script with grid search hyperparameters
# Edit the values below to change hyperparameters

# Change to project root directory
cd "$(dirname "$0")/.."

# Grid search hyperparameters
NUM_EPOCHS_VALUES=(1 2 3)  # Grid search values for num_epochs
LEARNING_RATE_VALUES=(0.0001 0.0005 0.001)  # Grid search values for learning_rate
HIDDEN_DIM=256
DROPOUT=0.3
BATCH_SIZE=8
VALIDATION_SIZE=0.2

# Model configuration
FEATURE_DIM=1536
MODEL_NAME="MeanPooling"
NUM_OF_CLASSES=3

# Training configuration
DEVICE="mps"
DISABLE_PROGRESS_BAR=false
RANDOM_SEED=42
VALIDATION_RATE=100
WEIGHT_DECAY=0.01

# Data configuration
FEATURE_DIR="data/features/uni/TCGA/TCGA-LGG"
METADATA_PATH="example_data/TCGA-LGG/training_metadata.csv"
NUM_WORKERS=0

# Logging configuration
EXPERIMENT_NAME="TCGA-LGG"
USE_WEIGHTED_SAMPLER=false

# Grid search loops
for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
    for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
        echo "Running training with NUM_EPOCHS=$NUM_EPOCHS, LEARNING_RATE=$LEARNING_RATE"
        
        # Build command with all parameters
        CMD="python train_mil.py \
            --num_epochs $NUM_EPOCHS \
            --learning_rate $LEARNING_RATE \
            --weight_decay $WEIGHT_DECAY \
            --hidden_dim $HIDDEN_DIM \
            --dropout $DROPOUT \
            --batch_size $BATCH_SIZE \
            --validation_size $VALIDATION_SIZE \
            --random_seed $RANDOM_SEED \
            --device $DEVICE \
            --model_name $MODEL_NAME \
            --num_of_classes $NUM_OF_CLASSES \
            --feature_dim $FEATURE_DIM \
            --metadata_path $METADATA_PATH \
            --feature_dir $FEATURE_DIR \
            --num_workers $NUM_WORKERS \
            --validation_rate $VALIDATION_RATE \
            --experiment_name $EXPERIMENT_NAME \
        
        # Add boolean flags if needed
        if [ "$DISABLE_PROGRESS_BAR" = true ]; then
            CMD="$CMD --disable_progress_bar"
        fi
        
        if [ "$USE_WEIGHTED_SAMPLER" = true ]; then
            CMD="$CMD --use_weighted_sampler"
        fi
        
        # Run the command
        eval $CMD
        
        echo "Completed training with NUM_EPOCHS=$NUM_EPOCHS, LEARNING_RATE=$LEARNING_RATE"
        echo "----------------------------------------"
    done
done

echo "Grid search completed for NUM_EPOCHS values: ${NUM_EPOCHS_VALUES[*]} and LEARNING_RATE values: ${LEARNING_RATE_VALUES[*]}"
