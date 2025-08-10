# TCGA Processing
## Prerequisites
- installed gdc-client (use `bash_commands/install_gdc_ubuntu.sh`)
- downloaded manifest file (find samples in `example_data/` folder)
### Step 1: Download slides from TCGA using manifest
``` 
python scripts/data_preparation/download_from_manifest.py \
-m PATH_TO_MANIFEST_FILE \
-s START_INDEX \
-e END_INDEX \
-D DESTINATION_FOLDER
```

### Step 2: Extract valid patch coordinates

```
PYTHONPATH=. python -u CLAM/create_patches_fp.py \ 
--source FOLDER_WITH_SLIDES \
--save_dir DESTINATION_FOLDER \
--patch_size 512 \ 
--step_size 512 \ 
--patch_level 0 \
--seg --patch --stitch \ 
--disable_progress_bar \
--log_level INFO
```

### Step 3: Extract vision features

``` 
python extract_vision_features_local.py --config CONFIG_FILE_PATH
```

### Step 4: Train MIL model
```
python train_mil.py --config CONFIG_FILE_PATH
```
## Quick start example

1. Download the frist 2 slides from TCGA-LGG manifest
```
python scripts/data_preparation/download_from_manifest.py -m example_data/TCGA-LGG/manifest.txt -s 0 -e 2 -D example_data/TCGA-LGG/slides
```

2. Extract valid coordinates

```
PYTHONPATH=. python -u CLAM/create_patches_fp.py \
--source example_data/TCGA-LGG/slides \
--save_dir example_data/TCGA-LGG/coordinates \
--patch_size 512 \
--step_size 512 \
--patch_level 0 \
--seg --patch --stitch \
--disable_progress_bar \
--log_level INFO
```

3. Extract vision features

```
python extract_vision_features_local.py --config example_data/TCGA-LGG/configs/vision_feature_extraction.yml
```

# TODO update it later
4. Train ABMIL model
```
python train_mil.py 
`