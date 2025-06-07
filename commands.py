# download commands:
DOWNLOAD_WSI = """
python scripts/data_preparation/download_from_manifest.py \
-m data/TCGA-LGG/manifests/train.txt \
-s 0 \
-e 32 \
-d data/TCGA-LGG/tmp \
-D data/TCGA-LGG/raw_data_part1/wsi \
-u unsuccessfull_dowloads_part1.txt
"""

EXTRACT_COORDS = """
python scripts/data_preparation/extract_coordinates_clam.py \
--source_dir data/TCGA-LGG/raw_data_part1/wsi \
--save_dir data/TCGA-LGG/processed_data_part1/wsi \
--patch_size 224

python CLAM/create_patches_fp.py \
--source DATA_DIRECTORY \
--save_dir RESULTS_DIRECTORY \
--patch_size 224 
--preset tcga.csv
--seg --patch --stitch

"""