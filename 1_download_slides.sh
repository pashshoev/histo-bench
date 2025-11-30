#!/bin/bash


# Downloading
MANIFEST_FILE="experiments/TCGA-NSCLC/manifest_LUAD_LUSC.txt"
START_INDEX=0
END_INDEX=1
DATA_DIR="data/TCGA-NSCLC"

# 1. Download slides for this batch
echo "[MAIN] DOWNLOADING SLIDES $start_idx-$end_idx..."
python scripts/data_preparation/download_from_manifest.py \
    -m "$MANIFEST_FILE" \
    -s "$START_INDEX" \
    -e "$END_INDEX" \
    -D "$DATA_DIR/slides"
echo "[MAIN] Download completed."