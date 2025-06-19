import argparse
import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.data_preparation.patch_dataset import (PatchDatasetFromWSI,
                                                    WSIDataset)
from scripts.models.vision.resnet import ResNetEncoder, ResNetTransforms


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


def extract_features_for_wsi(patch_loader: DataLoader, encoder, device):
    """
    Iterates through the patch loader and extracts embeddings using the encoder.

    Args:
        patch_loader (DataLoader): DataLoader containing image patches and coordinates.
        encoder: Feature extractor model with `extract_embeddings` method.
        device (torch.device): Torch device used for inference.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Arrays of extracted embeddings and their corresponding coordinates.
    """
    all_embeddings = []
    all_coordinates = []

    for batch in tqdm(patch_loader, desc="Extracting features"):
        patches = batch["patch"].to(device)
        coords = batch["coordinates"]
        embeddings = encoder.extract_embeddings(patches)
        all_embeddings.append(embeddings.cpu())
        all_coordinates.append(coords.cpu())

    embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
    coordinates_np = torch.cat(all_coordinates, dim=0).numpy()

    return embeddings_np, coordinates_np


def save_features_to_h5(embeddings: np.ndarray, coordinates: np.ndarray, output_path: str):
    """
    Saves extracted features and coordinates to an HDF5 file with gzip compression.

    Args:
        embeddings (np.ndarray): Feature vectors extracted from patches.
        coordinates (np.ndarray): Corresponding patch coordinates.
        output_path (str): Destination path for the HDF5 file.
    """
    print(f"Saving features and coordinates to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings, compression="gzip")
        f.create_dataset('coordinates', data=coordinates, compression="gzip")


def process_single_wsi(wsi_filename: str, config: dict):
    """
    Full processing pipeline for a single WSI file:
    - Loads the WSI and its patch coordinates.
    - Applies transformations and loads patches into a DataLoader.
    - Extracts features from patches.
    - Saves the result to an HDF5 file.

    Args:
        wsi_filename (str): Filename of the whole slide image.
        config (dict): Dictionary containing paths, model settings, and processing parameters.
    """
    device = torch.device(config["device"])
    encoder, patch_transforms = get_encoder(config, device)

    wsi_path = os.path.join(config["wsi_dir"], wsi_filename)
    coord_filename = wsi_filename.replace(config["wsi_file_extension"], ".h5")
    coord_path = os.path.join(config["coordinates_dir"], coord_filename)

    if not os.path.exists(wsi_path):
        print(f"Warning: WSI file not found: {wsi_path}. Skipping.")
        return
    if not os.path.exists(coord_path):
        print(f"Warning: Coordinate file not found: {coord_path}. Skipping.")
        return

    os.makedirs(config["features_dir"], exist_ok=True)

    patch_dataset = PatchDatasetFromWSI(
        coordinates_file=coord_path,
        wsi_path=wsi_path,
        img_transforms=patch_transforms
    )

    patch_loader = DataLoader(
        patch_dataset,
        batch_size=config["patch_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=(device.type != "cpu")
    )

    print(f"Using device: {device}")
    embeddings, coordinates = extract_features_for_wsi(patch_loader, encoder, device)
    output_path = os.path.join(config["features_dir"], coord_filename)
    save_features_to_h5(embeddings, coordinates, output_path)


def run_feature_extraction(config_path: str):
    """
    Executes the full feature extraction pipeline for all WSIs listed in the metadata.

    Args:
        config_path (str): Path to the YAML configuration file containing paths, model, and processing settings.

    This function:
    - Loads configuration and initializes the encoder model.
    - Loads WSI metadata and iterates through all slides.
    - For each WSI, calls `process_single_wsi` to extract and save features.
    """

    config = load_config(config_path)


    wsi_dataset = WSIDataset(config["wsi_meta_data_path"])
    print(f"Processing total {len(wsi_dataset)} slides")

    for slide_idx in tqdm(range(len(wsi_dataset)), desc="Processing WSIs"):
        wsi_filename = wsi_dataset[slide_idx]
        process_single_wsi(wsi_filename, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run WSI feature extraction.")
    parser.add_argument("--config", type=str, default="configs/vision_feature_extraction.yml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()

    run_feature_extraction(args.config)
    print("\nFeature extraction process completed (demo break applied).")

