#!/bin/bash

# Train MIL script with grid search hyperparameters
# Edit the values below to change hyperparameters

# Change to project root directory
cd "$(dirname "$0")/.."

# Hyperparameters
NUM_EPOCHS_VALUES=(1 2 3)
LEARNING_RATE_VALUES=(0.0001 0.0005 0.001) 
# HIDDEN_DIM=256 # Not used for MeanPooling
# DROPOUT=0.3 # Not used for MeanPooling
BATCH_SIZE=1
VALIDATION_SIZE=0.2

# Model configuration
FEATURE_DIM=1536
MODEL_NAME="MeanPooling"
NUM_OF_CLASSES=3

# Training configuration
DEVICE="mps"
RANDOM_SEED=42
VALIDATION_RATE=100
WEIGHT_DECAY=0.01

# Data configuration
FEATURE_DIR="data/features/uni/TCGA/TCGA-LGG"
METADATA_PATH="example_data/TCGA-LGG/training_metadata.csv"
NUM_WORKERS=0

# Logging configuration
COMET_API_KEY="9Uk3HLvmNE6oVdcML7nmU4HSX"
EXPERIMENT_NAME="TCGA-LGG"
USE_WEIGHTED_SAMPLER=false
DISABLE_PROGRESS_BAR=false


# Grid search loops
for NUM_EPOCHS in "${NUM_EPOCHS_VALUES[@]}"; do
    for LEARNING_RATE in "${LEARNING_RATE_VALUES[@]}"; do
        echo "Running training with NUM_EPOCHS=$NUM_EPOCHS, LEARNING_RATE=$LEARNING_RATE"
        
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
            --metadata_path $METADATA_PATH \
            --feature_dir $FEATURE_DIR \
            --num_workers $NUM_WORKERS \
            --validation_rate $VALIDATION_RATE \
            --experiment_name $EXPERIMENT_NAME \
            --comet_api_key $COMET_API_KEY"
        
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
