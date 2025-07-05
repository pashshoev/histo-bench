import os
from tqdm import tqdm

from extract_vision_features_local import run_feature_extraction
from scripts.data_preparation.split_gdc_manifest import split_manifest
from scripts.data_preparation.download_from_manifest import download_and_move
from scripts.data_preparation.extract_coordinates_clam import  run_clam_coordinate_extraction

data_folder = "./data"
dataset_path = os.path.join(data_folder, "TCGA-LGG")
manifest_dir = os.path.join(dataset_path, "manifests")
final_path = os.path.join(dataset_path, "patches_data", "wsi")

manifest_path = os.path.join(manifest_dir, "gdc_manifest.2025-05-31.101400.txt")
unsuccessful_downloads_path = "unsuccessful_downloads.txt"
config_for_feature_extraction = "configs/vision_feature_extraction.yml"

wsi_batch_size = 2
total_wsi_to_process = 6 # For development

# manifest = pd.read_csv(manifest_path).sample(frac=1, random_state=42)

# 1. Split manifest into train test
train_manifest_path, test_manifest_path = split_manifest(manifest_path, manifest_dir, test_size=0.2)


for start_idx in tqdm(range(0, total_wsi_to_process, wsi_batch_size), desc="Processing batches..."):
    end_idx = start_idx + wsi_batch_size
    end_idx = min(end_idx, total_wsi_to_process)
    batch_wsi_dir = os.path.join(dataset_path, f"raw_batch_{start_idx}", "wsi")

    # 2. Download batch samples from manifest to batch_wsi_dir
    print("Download samples for current batch...")
    try:
        unsuccessful_downloads = download_and_move(
            manifest_file_path=train_manifest_path,
            start_index=start_idx,
            end_index=end_idx,
            download_dir=os.path.join(dataset_path, "tmp"),
            destination_dir= batch_wsi_dir,
        )

        if unsuccessful_downloads:
            with open(unsuccessful_downloads_path, 'a') as f:
                for filename in unsuccessful_downloads:
                    f.write(f"{filename}\n")
            print(f"Unsuccessful downloads logged to: {unsuccessful_downloads_path}")
        else:
            print("All targeted slides were successfully downloaded and moved.")
            # if os.path.exists(unsuccessful_downloads_path):
            #     os.remove(unsuccessful_downloads_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # 3. Extract coordinates of valid patches
    # Call CLAM script using subprocess
    print("Extracting valid patch coordinates...")
    batch_wsi_dir = os.path.abspath(batch_wsi_dir)
    final_path = os.path.abspath(final_path)
    success = run_clam_coordinate_extraction(batch_wsi_dir, final_path, patch_size=224)
    if success:
        print("Successfully extracted coordinates.")

    # 4. Extract features using pre-trained models
    # Use extract_vision_features.py script
    run_feature_extraction(config_for_feature_extraction)
