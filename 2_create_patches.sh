DATA_DIR="data/TCGA-NSCLC"

# Patching
PATCH_SIZE=512
STEP_SIZE=512
PATCH_LEVEL=0

echo "[MAIN] CREATING PATCHES..."
PYTHONPATH=. python -u CLAM/create_patches_fp.py \
    --source "$DATA_DIR/slides" \
    --save_dir "$DATA_DIR/coordinates" \
    --preset "CLAM/presets/tcga.csv" \
    --patch_size $PATCH_SIZE \
    --step_size $STEP_SIZE \
    --patch_level $PATCH_LEVEL \
    --seg --patch --stitch \
    --log_level INFO
echo "[MAIN] Patch creation completed."