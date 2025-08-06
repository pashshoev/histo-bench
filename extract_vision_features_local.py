import argparse
import os
import sys
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import time
import torch.nn as nn
from torchvision import transforms
from scripts.data_preparation.patch_dataset import (PatchDatasetFromWSI,
                                                    WSIDataset)
from scripts.models.vision.resnet import ResNetEncoder
from scripts.models.vision.uni import UNIEncoder
from scripts.models.vision.conch import CONCHEncoder
from scripts.models.vision.plip import PLIPEncoder
from scripts.models.vision.base import ModelName


def load_config(config_path):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def get_encoder(config, device):
    """
    Initializes the feature encoder and its associated transforms based on the configuration.

    Args:
        config (dict): Configuration dictionary. Must include:
                       - "model_name": name of the model (e.g., "resnet50", "UNI")
                       - For UNI: "checkpoint_path": path to the .bin file
        device (torch.device): The device (CPU or CUDA) to load the model on.

    Returns:
        tuple: (encoder, patch_transforms)
    """
    logger.debug(f"Loading encoder: {config['model_name']}")
    try:
        if config["model_name"] == ModelName.RESNET50.value:
            encoder = ResNetEncoder(device=device)
        elif config["model_name"] == ModelName.UNI.value:
            encoder = UNIEncoder(hf_token=config["hf_token"], device=device)
        elif config["model_name"] == ModelName.CONCH.value:
            encoder = CONCHEncoder(hf_token=config["hf_token"], device=device)
        elif config["model_name"] == ModelName.PLIP.value:
            encoder = PLIPEncoder(hf_token=config["hf_token"], device=device)
        else:
            raise ValueError(f"Model '{config['model_name']}' not supported.")
    except Exception as e:
        logger.error(f"Error loading encoder: {e}")
        raise e
    logger.info(f"Encoder {encoder.model_name} loaded successfully")
    return encoder, encoder.transform


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
    batch_idx = 0
    for batch in tqdm(patch_loader, desc="Extracting features", disable=os.environ.get("DISABLE_PROGRESS_BAR", False)):
        batch_idx += 1
        if batch_idx % 1 == 0:
            logger.info(f"Extracting features for batch {batch_idx}/{len(patch_loader)}")

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
    logger.info(f"Saving features and coordinates to: {output_path}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('features', data=embeddings, compression="gzip")
        f.create_dataset('coordinates', data=coordinates, compression="gzip")


def process_single_wsi(wsi_filename: str, 
                       device: torch.device,
                       encoder: nn.Module,
                       patch_transforms: transforms.Compose,
                       config: dict):
    """
    Full processing pipeline for a single WSI file:
    - Loads the WSI and its patch coordinates.
    - Applies transformations and loads patches into a DataLoader.
    - Extracts features from patches.
    - Saves the result to an HDF5 file.

    Args:
        wsi_filename (str): Filename of the whole slide image.
        device (torch.device): Device to use for inference.
        encoder (nn.Module): Feature extractor model.
        patch_transforms (transforms.Compose): Transformations to apply to patches.
        config (dict): Dictionary containing paths, model settings, and processing parameters.
    """


    wsi_path = os.path.join(config["wsi_dir"], wsi_filename)
    file_extension = wsi_filename.split(".")[-1]
    coord_filename = wsi_filename.replace(file_extension, "h5")
    coord_path = os.path.join(config["coordinates_dir"], coord_filename)

    if not os.path.exists(wsi_path):
        logger.warning(f"WSI file not found: {wsi_path}. Skipping.")
        return
    if not os.path.exists(coord_path):
        logger.warning(f"Coordinate file not found: {coord_path}. Skipping.")
        return

    os.makedirs(config["features_dir"], exist_ok=True)

    patch_dataset = PatchDatasetFromWSI(
        coordinates_file=coord_path,
        wsi_path=wsi_path,
        transform=patch_transforms
    )

    patch_loader = DataLoader(
        patch_dataset,
        batch_size=config["patch_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=(device.type == "cuda")
    )

    logger.debug(f"Using device: {device}")
    embeddings, coordinates = extract_features_for_wsi(patch_loader, encoder, device)
    output_path = os.path.join(config["features_dir"], coord_filename)
    save_features_to_h5(embeddings, coordinates, output_path)

    del patch_loader
    del patch_dataset


def run_feature_extraction(config: dict):
    """
    Executes the full feature extraction pipeline for all WSIs listed in the metadata.

    Args:
        config (dict): Dictionary containing paths, model, and processing settings.

    This function:
    - Loads configuration and initializes the encoder model.
    - Loads WSI metadata and iterates through all slides.
    - For each WSI, calls `process_single_wsi` to extract and save features.
    """
    logger.remove()
    logger.add(sys.stderr, level=config["log_level"].upper(), format="{time:YYYY-MM-DD HH:mm:ss} | {message}") 

    os.environ["DISABLE_PROGRESS_BAR"] = str(config["disable_progress_bar"])


    wsi_dataset = WSIDataset(config["wsi_meta_data_path"])
    logger.info(f"Processing total of {len(wsi_dataset)} slides")
    processing_times = []
    device = torch.device(config["device"])
    encoder, patch_transforms = get_encoder(config, device)
    for slide_idx in tqdm(range(len(wsi_dataset)), desc="Processing WSIs", disable=os.environ.get("DISABLE_PROGRESS_BAR", False)):
        logger.info(f"Processing slide ({slide_idx}/{len(wsi_dataset)})")
        wsi_filename = wsi_dataset[slide_idx]
        start_time = time.time()
        process_single_wsi(wsi_filename, device, encoder, patch_transforms, config)
        end_time = time.time()
        processing_times.append(end_time - start_time)
        logger.info(f"Processing time: {end_time - start_time} seconds")
    
    logger.info(f"Average processing time per slide: {np.mean(processing_times).round(3)} seconds")


def build_config_from_args(args):
    """
    Builds a configuration dictionary from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        dict: Configuration dictionary
    """
    config = {
        "wsi_dir": args.wsi_dir,
        "coordinates_dir": args.coordinates_dir,
        "wsi_meta_data_path": args.wsi_meta_data_path,
        "features_dir": args.features_dir,
        "device": args.device,
        "patch_batch_size": args.patch_batch_size,
        "num_workers": args.num_workers,
        "model_name": args.model_name,
        "hf_token": args.hf_token,
        "log_level": args.log_level,
        "disable_progress_bar": args.disable_progress_bar
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run WSI feature extraction.")

    # Config file argument (optional)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the configuration YAML file. If not provided, individual arguments will be used.")

    # Path arguments
    parser.add_argument("--wsi_dir", type=str, default="example_data/TCGA-Lung/slides",
                        help="Directory containing WSI files")
    parser.add_argument("--coordinates_dir", type=str, default="example_data/TCGA-Lung/coordinates/patches",
                        help="Directory containing coordinate files")
    parser.add_argument("--wsi_meta_data_path", type=str, default="example_data/TCGA-Lung/coordinates/process_list_autogen.csv",
                        help="Path to WSI metadata CSV file")
    parser.add_argument("--features_dir", type=str, default="example_data/TCGA-Lung/features/resnet",
                        help="Directory to save extracted features")

    # Processing arguments
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for processing (cpu/cuda)")
    parser.add_argument("--patch_batch_size", type=int, default=128,
                        help="Batch size for patch processing")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="ResNet50",
                        help="Name of the model to use for feature extraction")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Hugging Face token for model access")

    # Logging arguments
    parser.add_argument("--log_level", type=str, default="DEBUG",
                        help="Logging level")
    parser.add_argument("--disable_progress_bar", action="store_true",
                        help="Disable progress bars")

    args = parser.parse_args()
    
    # Load config from file if provided, otherwise build from arguments
    if args.config is not None:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = build_config_from_args(args)
        logger.info("Built configuration from command-line arguments")
    
    run_feature_extraction(config)
    logger.info("Feature extraction process completed.")

