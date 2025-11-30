# histo-bench

A universal framework for training and benchmarking Encoder × MIL × P combinations on The Cancer Genome Atlas Program (TCGA) benchmark datasets, where Encoder represents vision foundation models (ResNet50, PLIP, UNI, CONCH), MIL represents Multiple Instance Learning architectures (ABMIL, CLAM_MB, WiKG, TransMIL, MeanPooling), and P represents patch sampling ratios. The patch sampling ratio plays a major role in benchmarking, as it controls the fraction of patches used per slide and significantly impacts both computational efficiency and model performance. This framework provides an end-to-end pipeline from downloading whole slide images (WSI) to training MIL models with cross-validation and hyperparameter search.

While the primary usage is systematic evaluation across all Encoder × MIL × P combinations, the framework is flexible: any variable can be fixed to simplify training. For example, you can test a single encoder with a single MIL model using all patches (P=1.0), or configure any arbitrary combination of these variables to suit your specific research needs.

## Installation

### Prerequisites

**GDC Data Transfer Tool (gdc-client)**

The GDC client is required for downloading slides from TCGA in Step 1. Download and install it following the guidelines at the [official GDC Data Transfer Tool portal](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).

### Basic Installation

Install the required dependencies:

```bash
make install_deps
```

This will:
- Upgrade pip
- Install all packages from `requirements.txt`

### Optional: Install CONCH Encoder

To use the CONCH encoder model, install it separately:

```bash
make install_conch
```

**Note:** CONCH and other encoders (PLIP, UNI) are downloaded from Hugging Face and may require access via a Hugging Face account. Please refer to the official Hugging Face model pages (see [Step 3: Extract Features](#step-3-extract-features)) to request access and obtain authentication tokens.

## Quick Start

The framework includes ready-to-use scripts and metadata files to get you started quickly.

### Available Datasets

The `experiments/` folder contains pre-configured metadata files for two TCGA datasets:

- **TCGA-NSCLC**: Non-small cell lung cancer (LUAD/LUSC)
  - `experiments/TCGA-NSCLC/manifest_LUAD_LUSC.txt`
  - `experiments/TCGA-NSCLC/training_labels.csv`
- **TCGA-RCC**: Renal cell carcinoma (KICH/KIRC/KIRP)
  - `experiments/TCGA-RCC/manifest_KICH_KIRC_KIRP.txt`
  - `experiments/TCGA-RCC/training_labels.csv`

### Running the Pipeline

The framework provides four numbered scripts that can be run directly in sequence:

```bash
# Step 1: Download slides
bash 1_download_slides.sh

# Step 2: Create patches
bash 2_create_patches.sh

# Step 3: Extract features
bash 3_extract_features.sh

# Step 4: Train MIL models
bash 4_train_mil_models.sh
```

**Default Configuration:**
- **Dataset**: TCGA-NSCLC
- **Encoder**: ResNet50
- **Slides**: Downloads and processes 10 slides (indices [0-10)) by default
- **MIL Models**: ABMIL, CLAM_MB, WiKG, TransMIL
- **Patch Sampling Ratios**: 0.01, 0.1, 1.0

**Customization:**
You can modify variables inside each script to:
- Change the dataset (update manifest file paths)
- Select different encoders (ResNet50, PLIP, UNI, CONCH)
- Adjust the number of slides to download
- Modify patch extraction parameters
- Configure MIL training hyperparameters

Simply edit the variables at the top of each script before running.

## Framework Overview

This framework enables systematic evaluation of different encoder and MIL model combinations on histopathology datasets. The workflow consists of four main steps:

1. **Download slides** - Download WSI files from TCGA using manifest files
2. **Create patches** - Extract tissue patches from slides using CLAM preprocessing
3. **Extract features** - Generate feature embeddings using vision encoders
4. **Train MIL models** - Train and evaluate MIL models with cross-validation and grid search

Each step has configurable parameters to customize the pipeline for different experiments.

**Extensibility:** The framework is designed to be easily extended. You can add new encoders by implementing the standard interface defined in `scripts/models/encoder/base.py`, or add new MIL aggregators by creating new modules in `scripts/models/MIL/`. This modular design allows researchers to quickly integrate new models and architectures into the benchmarking pipeline.

## Workflow

### Step 1: Download Slides

Download whole slide images from TCGA using the GDC client and manifest files.

**Script:** `1_download_slides.sh`

**Parameters:**
- `MANIFEST_FILE`: Path to the manifest file (e.g., `experiments/TCGA-NSCLC/manifest_LUAD_LUSC.txt`)
- `START_INDEX`: Starting index in the manifest (0-based)
- `END_INDEX`: Ending index in the manifest (exclusive)
- `DATA_DIR`: Base directory for storing data

**Example:**
```bash
#!/bin/bash

MANIFEST_FILE="experiments/TCGA-NSCLC/manifest_LUAD_LUSC.txt"
START_INDEX=0
END_INDEX=1
DATA_DIR="data/TCGA-NSCLC"

python scripts/data_preparation/download_from_manifest.py \
    -m "$MANIFEST_FILE" \
    -s "$START_INDEX" \
    -e "$END_INDEX" \
    -D "$DATA_DIR/slides"
```

**Output Structure:**
```
data/TCGA-NSCLC/
└── slides/
    ├── SLIDE1.svs
    ├── SLIDE2.svs
    ├── SLIDE3.svs
    └── ...
```

The script uses `gdc-client` to download from the manifest, resolves nested directory structures, and places all slides in the destination folder.

### Step 2: Create Patches

Extract tissue patches from whole slide images using the CLAM preprocessing pipeline.

**Script:** `2_create_patches.sh`

**Parameters:**
- `PATCH_SIZE`: Size of each patch in pixels (e.g., 512)
- `STEP_SIZE`: Step size for patch extraction (e.g., 512)
- `PATCH_LEVEL`: Magnification level for patch extraction (e.g., 1)
- `DATA_DIR`: Base directory containing slides

**Example:**
```bash
DATA_DIR="data/TCGA-NSCLC"
PATCH_SIZE=512
STEP_SIZE=512
PATCH_LEVEL=1

PYTHONPATH=. python -u CLAM/create_patches_fp.py \
    --source "$DATA_DIR/slides" \
    --save_dir "$DATA_DIR/coordinates" \
    --preset "CLAM/presets/tcga.csv" \
    --patch_size $PATCH_SIZE \
    --step_size $STEP_SIZE \
    --patch_level $PATCH_LEVEL \
    --seg --patch --stitch \
    --log_level INFO
```

**Output Structure:**
```
data/TCGA-NSCLC/
├── slides/
│   ├── SLIDE1.svs
│   └── ...
└── coordinates/
    ├── patches/
    │   ├── SLIDE1.h5
    │   ├── SLIDE2.h5
    │   └── ...
    └── process_list_autogen.csv
```

**Output Format:**
Each `.h5` file contains:
- `coords`: Array of patch coordinates `(x, y)` for tissue patches
- Metadata attributes: `patch_level`, `patch_size`, `wsi_name`, etc.

The output is one `.h5` file per slide with the same base name (e.g., `SLIDE1.svs` → `SLIDE1.h5`).

### Step 3: Extract Features

Extract feature embeddings from patches using vision encoders.

**Script:** `3_extract_features.sh`

**Parameters:**
- `MODEL_NAME`: Encoder model to use (`ResNet50`, `PLIP`, `UNI`, or `CONCH`)
- `NUM_WORKERS`: Number of parallel workers for data loading
- `PATCH_BATCH_SIZE`: Batch size for patch processing
- `DEVICE`: Device to use (`cuda` or `cpu`)
- `HF_TOKEN`: Hugging Face token (required for PLIP, UNI, CONCH; not needed for ResNet50)

**Example:**
```bash
DATA_DIR="data/TCGA-NSCLC"
NUM_WORKERS=12
PATCH_BATCH_SIZE=32
DEVICE="cuda"
MODEL_NAME="ResNet50"
HF_TOKEN="hf-..."  # Not needed for ResNet50

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
```

**Available Encoders:**
- **ResNet50**: Standard CNN backbone (ImageNet pretrained), no token required
- **PLIP**: Lightweight CLIP model for pathology, requires HF token. Model: [`vinid/plip`](https://huggingface.co/vinid/plip)
- **UNI**: Transformer model optimized for histopathology, requires HF token. Model: [`MahmoodLab/UNI`](https://huggingface.co/MahmoodLab/UNI)
- **CONCH**: Contrastive learning-based model, requires HF token and separate installation. Model: [`MahmoodLab/conch`](https://huggingface.co/MahmoodLab/conch)

**Note:** Encoders are downloaded from Hugging Face, and model owners may require access via a Hugging Face account. Please refer to the official Hugging Face model pages linked above to request access and obtain the necessary authentication token.

**Output Structure:**
```
data/TCGA-NSCLC/
└── features/
    └── ResNet50/
        ├── SLIDE1.h5
        ├── SLIDE2.h5
        └── ...
```

Each feature `.h5` file contains:
- `features`: Extracted feature embeddings
- `coordinates`: Corresponding patch coordinates

### Step 4: Train MIL Models

Train Multiple Instance Learning models with cross-validation and grid search over hyperparameters.

**Script:** `4_train_mil_models.sh`

**Key Features:**
- Cross-validation with configurable number of folds
- Grid search over multiple hyperparameter combinations
- Support for multiple MIL architectures
- Early stopping and model checkpointing
- Experiment tracking (Comet ML)

**Grid Search Dimensions:**
- `PATCH_SAMPLING_RATIO_VALUES`: Fraction of patches to sample per slide (e.g., `0.01`, `0.1`, `1.0`)
- `MODEL_NAMES`: MIL model architectures (e.g., `ABMIL`, `CLAM_MB`, `WiKG`, `TransMIL`)
- Additional hyperparameters: learning rate, weight decay, hidden dimensions, dropout, etc.

**Example Configuration:**
```bash
# Hyperparameters
PATCH_SAMPLING_RATIO_VALUES=(0.01 0.1 1.0)
MODEL_NAMES=("ABMIL" "CLAM_MB" "WiKG" "TransMIL")
NUM_EPOCHS_VALUES=(50)
LEARNING_RATE_VALUES=(0.0001)

# Data configuration
FEATURE_DIR="data/TCGA-NSCLC/features/ResNet50"
METADATA_PATH="experiments/TCGA-NSCLC/training_labels.csv"

# Cross-validation
N_FOLDS=5
```

**Available MIL Models:**
- **MeanPooling**: Simple MIL approach that aggregates instance features via global average pooling
- **ABMIL**: Attention-based MIL that learns to focus on informative patches
- **CLAM_MB**: Multi-branch CLAM with instance-level supervision
- **WiKG**: Graph-based MIL that captures spatial relationships
- **TransMIL**: Transformer-based MIL for long-range dependencies

The script automatically runs training for all combinations of `PATCH_SAMPLING_RATIO_VALUES × MODEL_NAMES`, performing cross-validation for each configuration.

## Notes

- For ROCm GPUs, install the correct version of PyTorch
- PyTorch version must be >= 2.6 to load pretrained foundation models
- Set `COMET_API_KEY` environment variable for experiment tracking
- Ensure `gdc-client` is installed (see [Prerequisites](#prerequisites)) before running Step 1
