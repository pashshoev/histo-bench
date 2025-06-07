import torch
from torch.utils.data import DataLoader
from scripts.models.vision.resnet import ResNetEncoder, ResNetTransforms
from scripts.data_preparation.patch_dataset import PatchDatasetFromWSI, WSIDataset
import os
from tqdm import tqdm
import yaml
import h5py
import argparse


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_encoder(config, device):
    """
    Initializes the feature encoder and patch transforms based on the configuration.

    Args:
        config (dict): Configuration dictionary containing model_name.
        device (torch.device): The device (CPU or CUDA) to load the model on.

    Returns:
        tuple: A tuple containing the encoder model and its associated transforms.
    """
    encoder, patch_transforms = None, None
    if config["model_name"].startswith("resnet"):
        patch_transforms = ResNetTransforms
        encoder = ResNetEncoder(model_name=config["model_name"], device=device)
    else:
        raise ValueError(f"Model '{config['model_name']}' not supported.")
    return encoder, patch_transforms


def run_feature_extraction(config_path: str, verbose: bool = False):
    """
    Executes the feature extraction pipeline for WSIs.

    Args:
        config_path (str): Path to the YAML configuration file.
        verbose (bool): If True, prints detailed information during processing.
    """
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, patch_transforms = get_encoder(config, device)

    print(f"Device: {device}")

    # Initialize WSI Dataset
    wsi_dataset = WSIDataset(config["wsi_meta_data_path"])
    print(f"Processing total {len(wsi_dataset)} slides")

    # Iterate through each WSI
    for slide_idx in tqdm(range(len(wsi_dataset)), desc="Processing WSIs"):
        wsi_filename = wsi_dataset[slide_idx]
        wsi_path = os.path.join(config["wsi_dir"], wsi_filename)

        # Construct path to coordinates file (from CLAM output)
        coord_filename = wsi_filename.replace(config["wsi_file_extension"], ".h5")
        coord_path = os.path.join(config["patches_dir"], coord_filename)

        # Check if WSI and coordinate file exist before processing
        if not os.path.exists(wsi_path):
            print(f"Warning: WSI file not found: {wsi_path}. Skipping.")
            continue
        if not os.path.exists(coord_path):
            print(f"Warning: Coordinate file not found: {coord_path}. Skipping.")
            continue
        os.makedirs(config["features_dir"], exist_ok=True)

        # Initialize Patch Dataset for the current WSI
        patch_dataset = PatchDatasetFromWSI(
            coordinates_file=coord_path,
            wsi_path=wsi_path,
            img_transforms=patch_transforms
        )

        # Determine number of workers for DataLoader
        num_workers_to_use = config.get("num_workers", 0)

        # Create DataLoader for patches
        patch_loader = DataLoader(
            patch_dataset,
            batch_size=config["patch_batch_size"],
            shuffle=False,
            num_workers=num_workers_to_use,
            pin_memory=True if torch.cuda.is_available() else False
        )
        # --- Data accumulation for the current WSI ---
        all_embeddings_for_wsi = []
        all_coordinates_for_wsi = []

        # Process patches in batches
        for batch in tqdm(patch_loader, desc=f"Extracting features for {wsi_filename}"):
            batch_patches = batch["patch"]
            batch_coordinates = batch["coordinates"]

            # Extract embeddings using the encoder
            embeddings = encoder.extract_embeddings(batch_patches)
            all_embeddings_for_wsi.append(embeddings.cpu())
            all_coordinates_for_wsi.append(batch_coordinates.cpu())

        # --- Saving the extracted features and coordinates for the current WSI ---
        final_embeddings = torch.cat(all_embeddings_for_wsi, dim=0).numpy()
        final_coordinates = torch.cat(all_coordinates_for_wsi, dim=0).numpy()

        # Construct the output HDF5 filename
        feature_h5_filename = wsi_filename.replace(config["wsi_file_extension"], ".h5")
        feature_h5_path = os.path.join(config["features_dir"], feature_h5_filename)

        print(f"Saving features and coordinates to: {feature_h5_path}")
        with h5py.File(feature_h5_path, 'w') as f:
            # Create datasets for embeddings and coordinates
            f.create_dataset('embeddings', data=final_embeddings, compression="gzip")
            f.create_dataset('coordinates', data=final_coordinates, compression="gzip")
        print(f"Successfully saved features and coordinates for {wsi_filename}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run WSI feature extraction.")
    parser.add_argument("--config", type=str, default="configs/vision_feature_extraction.yml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    run_feature_extraction(args.config, verbose=True)
    print("\nFeature extraction process completed (demo break applied).")

