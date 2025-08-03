# Feature Extraction Scripts

This repository contains two scripts for extracting vision features from Whole Slide Images (WSI):

## 1. Local Feature Extraction (`extract_vision_features_local.py`)

**Purpose**: Extract features from WSIs stored locally on disk.

**Usage**:
```bash
python extract_vision_features_local.py --config configs/vision_feature_extraction.yml
```

**Configuration**: `configs/vision_feature_extraction.yml`
- Processes WSIs from local directories
- Uses a metadata CSV file to list WSIs to process
- Requires all WSI files and coordinate files to be pre-downloaded

**Key Features**:
- Sequential processing of WSIs
- Local file system based
- Simple configuration with local paths

## 2. GCS Feature Extraction (`extract_vision_features_batch.py`)

**Purpose**: Extract features from WSIs stored in Google Cloud Storage with on-the-fly downloading.

**Usage**:
```bash
python extract_vision_features_batch.py --config configs/vision_feature_extraction_gcs.yml
```

**Configuration**: `configs/vision_feature_extraction_gcs.yml`
- Downloads WSIs from GCS on-the-fly
- Concurrent downloading and processing
- Automatic cleanup of temporary files
- No need to pre-download all data

**Key Features**:
- Concurrent downloading and processing
- GCS integration with automatic authentication
- Memory-efficient (processes and deletes files immediately)
- Configurable batch sizes and worker counts
- Automatic coordinate file downloading

## Prerequisites

### For Local Processing:
- All WSI files must be downloaded to `wsi_dir`
- All coordinate files must be downloaded to `coordinates_dir`
- A metadata CSV file listing WSIs to process

### For GCS Processing:
- Google Cloud SDK installed and authenticated
- Google Cloud Storage access
- Sufficient local storage for temporary downloads

## Configuration Files

### Local Configuration (`vision_feature_extraction.yml`)
```yaml
# paths
wsi_dir: "data/TCGA-LGG/raw_data_part1/wsi"
coordinates_dir: "data/TCGA-LGG/processed_data_part1/wsi/patches"
wsi_meta_data_path: "data/TCGA-LGG/processed_data_part1/wsi/process_list_autogen.csv"
features_dir: "data/TCGA-LGG/processed_data_part1/wsi/features/resnet"

# processing
wsi_file_extension: ".svs"
device: "mps"
patch_batch_size: 64
num_workers: 8

# model configs
model_name: "resnet50"
```

### GCS Configuration (`vision_feature_extraction_gcs.yml`)
```yaml
# GCS Configuration
gcs_bucket_name: "histo-bench"
gcs_wsi_folder: "TCGA-LGG/wsi/"
gcs_coordinates_folder: "TCGA-LGG/coordinates/"

# Local paths
local_wsi_download_dir: "/tmp/wsi_temp_downloads/"
local_coordinates_dir: "/tmp/coordinates/"
features_dir: "data/TCGA-LGG/processed_data_part1/wsi/features/resnet"

# Processing settings
wsi_file_extension: ".svs"
device: "mps"
patch_batch_size: 64
num_workers: 8

# Concurrent processing settings
batch_size_wsi_listing: 5
max_download_workers: 3

# Model configuration
model_name: "resnet50"
```

## Output

Both scripts produce the same output format:
- HDF5 files containing extracted features and coordinates
- Files are saved to the specified `features_dir`
- Each WSI produces one HDF5 file with the same name as the coordinate file

## Performance Considerations

### Local Processing:
- Faster for small datasets or when all data is already local
- No network overhead
- Sequential processing may be slower for large datasets

### GCS Processing:
- Better for large datasets stored in the cloud
- Concurrent downloading and processing
- Memory efficient (streaming approach)
- Network bandwidth dependent

## Error Handling

Both scripts include error handling for:
- Missing WSI files
- Missing coordinate files
- Model loading errors
- File I/O errors

The GCS script additionally handles:
- Download failures
- Network timeouts
- GCS authentication errors 