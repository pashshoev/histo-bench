#!/bin/bash

DATA_DIR="data/TCGA-NSCLC"

# Feature extraction
NUM_WORKERS=12
PATCH_BATCH_SIZE=32
DEVICE="cuda"
MODEL_NAME="ResNet50"

HF_TOKEN="hf-..." # Not needed for ResNet50

# Extract features
echo "[MAIN] EXTRACTING FEATURES..."
python extract_features.py \
    --wsi_dir "$DATA_DIR/slides" \
    --coordinates_dir "$DATA_DIR/coordinates/patches" \
    --wsi_meta_data_path "$DATA_DIR/coordinates/process_list_autogen.csv" \
    --features_dir "$DATA_DIR/features/$MODEL_NAME" \
    --device "$DEVICE" \
    --patch_batch_size $PATCH_BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --model_name "$MODEL_NAME" \
    --hf_token "$HF_TOKEN"
echo "[MAIN] Feature extraction completed."

# =================================================================================