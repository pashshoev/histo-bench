#!/bin/bash

# Configuration
MANIFEST_FILE="example_data/TCGA-Lung/manifest/manifest_LUAD.txt"
BATCH_SIZE=100
TOTAL_SLIDES=543
START_INDEX=0
END_INDEX=542
DATA_DIR="data/TCGA-LUAD"
GCS_BUCKET="gs://histo-bench/TCGA-Lung"
DEVICE="cuda"
MODEL_NAME="ResNet50"
HF_TOKEN="hf_..."

# Calculate number of batches for display
SLIDES_TO_PROCESS=$((END_INDEX - START_INDEX))
NUM_BATCHES=$(( (SLIDES_TO_PROCESS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "[MAIN] PROCESSING SLIDES [$START_INDEX-$END_INDEX) ($SLIDES_TO_PROCESS slides) in $NUM_BATCHES batches of $BATCH_SIZE each"

# Iterate over start indices with step size of BATCH_SIZE (matching Python range pattern)
for start_idx in $(seq $START_INDEX $BATCH_SIZE $((END_INDEX - 1))); do
    end_idx=$((start_idx + BATCH_SIZE))
    
    # Ensure end_idx doesn't exceed the specified end index
    if [ $end_idx -gt $END_INDEX ]; then
        end_idx=$END_INDEX
    fi
    
    # Calculate current batch number for display
    current_batch=$(( (start_idx - START_INDEX) / BATCH_SIZE + 1 ))
    echo "=== PROCESSING BATCH $current_batch/$NUM_BATCHES (slides $start_idx-$end_idx) ==="
    
    # 1. Download slides for this batch
    echo "[MAIN] DOWNLOADING SLIDES $start_idx-$end_idx..."
    python scripts/data_preparation/download_from_manifest.py \
        -m "$MANIFEST_FILE" \
        -s "$start_idx" \
        -e "$end_idx" \
        -D "$DATA_DIR/slides"
    echo "[MAIN] Download completed. Waiting 5 seconds..."
    sleep 5
    
    # 2. Create patches
    echo "[MAIN] CREATING PATCHES..."
    PYTHONPATH=. python -u CLAM/create_patches_fp.py \
        --source "$DATA_DIR/slides" \
        --save_dir "$DATA_DIR/coordinates" \
        --preset "CLAM/presets/tcga.csv" \
        --patch_size 512 \
        --step_size 512 \
        --patch_level 0 \
        --seg --patch --stitch \
        --log_level INFO
    echo "[MAIN] Patch creation completed. Waiting 5 seconds..."
    sleep 5
    
    # 3. Extract features
    echo "[MAIN] EXTRACTING FEATURES..."
    python extract_vision_features_local.py \
        --wsi_dir "$DATA_DIR/slides" \
        --coordinates_dir "$DATA_DIR/coordinates/patches" \
        --wsi_meta_data_path "$DATA_DIR/coordinates/process_list_autogen.csv" \
        --features_dir "$DATA_DIR/features/$MODEL_NAME" \
        --device "$DEVICE" \
        --patch_batch_size 32 \
        --num_workers 12 \
        --model_name "$MODEL_NAME" \
        --hf_token $HF_TOKEN
    echo "[MAIN] Feature extraction completed. Waiting 5 seconds..."
    sleep 5
    
    # 4. Upload features and coordinates to GCS
    echo "[MAIN] UPLOADING TO GOOGLE CLOUD STORAGE..."
    gcloud storage cp --recursive "$DATA_DIR/features/$MODEL_NAME/"* "$GCS_BUCKET/features/$MODEL_NAME/"
    gcloud storage cp --recursive "$DATA_DIR/coordinates/patches/"* "$GCS_BUCKET/patches/"
    echo "[MAIN] GCS upload completed. Waiting 5 seconds..."
    sleep 5
    
    # 5. Remove local slides, coordinates and features
    echo "[MAIN] CLEANING UP LOCAL FILES..."
    rm -rf "$DATA_DIR/slides"
    rm -rf "$DATA_DIR/coordinates"
    rm -rf "$DATA_DIR/features"
    echo "[MAIN] Cleanup completed. Waiting 5 seconds..."
    sleep 5
    
    echo "=== Batch $current_batch completed ==="
    echo ""
done

echo "All batches completed successfully!"