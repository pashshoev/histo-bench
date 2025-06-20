# histo-bench

Benchmarking Generalist Vision-Language Models vs. Domain-Specific Foundation Models in Digital Pathology

## Description

This project aims to evaluate and compare the performance of general-purpose multimodal models (such as CLIP, SAM, and GPT-4) with specialized models designed for histopathological image analysis. We focus on H&E-stained slides for classification and segmentation tasks, analyzing performance, scalability, and trade-offs.

## Goals

- Compare generalist and specialist models on histopathological datasets
- Benchmark classification and segmentation tasks
- Evaluate performance, data efficiency, and cost trade-offs
- Provide reproducible results and insights for the research community

## Project Structure

## Installation
1. `git clone https://github.com/pashshoev/histo-bench.git` to clone repo
2. `make install_deps` to install dependencies
3. `make setup_clam` to setup CLAM repo for patch coordinates extraction

## Usage

### Feature Extraction

The project provides two approaches for extracting vision features from Whole Slide Images (WSI):

#### Quick Start
```bash
# For local processing (requires pre-downloaded data)
python run_feature_extraction.py --mode local

# For GCS processing (downloads data on-the-fly with cache control)
python run_feature_extraction.py --mode gcs
```

#### Direct Script Usage
```bash
# Local processing
python extract_vision_features_local.py --config configs/vision_feature_extraction.yml

# GCS processing with concurrent downloading and cache control
python extract_vision_features_batch.py --config configs/vision_feature_extraction_gcs.yml
```

**GCS Processing Features:**
- Concurrent downloading and processing
- Cache control to prevent memory issues (configurable via `max_cache_size`)
- Automatic cleanup of temporary files
- No need to pre-download all data

For detailed information about the feature extraction scripts, see the configuration files in `configs/`.

### Data Preparation
1. Download patches from manifest:
   ```bash
   python scripts/data_preparation/download_from_manifest.py \
     -m MANIFEST_FILE \
     -s START_INDEX \
     -e END_INDEX \
     -d WHERE_TO_DOWNLOAD \
     -D WHERE_TO_MOVE \
     -u FILE_TO_LOG_FAILED_DOWNLOAD_IDS
   ```

2. Extract coordinates using CLAM
3. Extract features using the scripts above