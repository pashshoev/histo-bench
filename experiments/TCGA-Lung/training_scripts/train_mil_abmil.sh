#!/bin/bash

# Train MIL script with grid search hyperparameters
# Edit the values below to change hyperparameters

# Change to project root directory
# cd "$(dirname "$0")/.."

# Hyperparameters
NUM_EPOCHS_VALUES=(25)
LEARNING_RATE_VALUES=(0.0001 0.001)
WEIGHT_DECAY_VALUES=(0 0.001)
HIDDEN_DIM=(512 1024) # Not used for MeanPooling
DROPOUT=(0 0.3) # Not used for MeanPooling
BATCH_SIZE=1
VALIDATION_SIZE=0.3

# Model configuration
FEATURE_DIM=2048
MODEL_NAME="ABMIL"
NUM_OF_CLASSES=2

# Training configuration
DEVICE="cuda"
RANDOM_SEED=42
VALIDATION_RATE=100

# Data configuration
FEATURE_DIR="data/TCGA-Lung/features/ResNet50"
METADATA_PATH="experiments/TCGA-Lung/training_labels.csv"
NUM_WORKERS=11

# Logging configuration
EXPERIMENT_NAME="TCGA-Lung-ResNet50-ABMIL"
USE_WEIGHTED_SAMPLER=false
DISABLE_PROGRESS_BAR=false

# Cross-validation configuration
N_FOLDS=1


# Grid search loops
for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
    for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
        for WEIGHT_DECAY in "${WEIGHT_DECAY_VALUES[@]}"; do
            echo "Running training with NUM_EPOCHS=$NUM_EPOCHS, LEARNING_RATE=$LEARNING_RATE, WEIGHT_DECAY=$WEIGHT_DECAY"
            
            # Build command with all parameters
            CMD="python train_mil.py \
                --num_epochs $NUM_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --weight_decay $WEIGHT_DECAY \
                --validation_size $VALIDATION_SIZE \
                --random_seed $RANDOM_SEED \
                --device $DEVICE \
                --model_name $MODEL_NAME \
                --num_of_classes $NUM_OF_CLASSES \
                --feature_dim $FEATURE_DIM \
                --hidden_dim $HIDDEN_DIM \
                --dropout $DROPOUT \
                --metadata_path $METADATA_PATH \
                --feature_dir $FEATURE_DIR \
                --num_workers $NUM_WORKERS \
                --validation_rate $VALIDATION_RATE \
                --experiment_name $EXPERIMENT_NAME \
                --n_folds $N_FOLDS"
            
            # Add boolean flags if needed
            if [ "$DISABLE_PROGRESS_BAR" = true ]; then
                CMD="$CMD --disable_progress_bar"
            fi
            
            if [ "$USE_WEIGHTED_SAMPLER" = true ]; then
                CMD="$CMD --use_weighted_sampler"
            fi
            
            # Run the command
            eval $CMD
            
            echo "Completed training with NUM_EPOCHS=$NUM_EPOCHS, LEARNING_RATE=$LEARNING_RATE, WEIGHT_DECAY=$WEIGHT_DECAY"
            echo "----------------------------------------"
        done
    done
done

echo "Grid search completed for NUM_EPOCHS values: ${NUM_EPOCHS_VALUES[*]} and LEARNING_RATE values: ${LEARNING_RATE_VALUES[*]}"
