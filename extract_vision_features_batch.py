import argparse
import os
import queue
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import subprocess

from scripts.data_preparation.patch_dataset import PatchDatasetFromWSI
from scripts.models.vision.conch import CONCHEncoder
from scripts.models.vision.plip import PLIPEncoder
from scripts.models.vision.resnet import ResNetEncoder
from scripts.models.vision.uni import UNIEncoder


def load_config(config_path):
    """Loads configuration from a YAML file."""
    logger.info(f"Loading config from: {config_path}")
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
    logger.info(f"Loading encoder: {config['model_name']}")
    if config["model_name"].startswith("resnet"):
        encoder = ResNetEncoder(model_name=config["model_name"], device=device)
    elif config["model_name"] == "uni":
        encoder = UNIEncoder(hf_token=config["hf_token"], device=device)
    elif config["model_name"]=="conch":
        encoder = CONCHEncoder(hf_token=config["hf_token"], device=device)
    elif config["model_name"]=="plip":
        encoder = PLIPEncoder(hf_token=config["hf_token"], device=device)
    else:
        raise ValueError(f"Model '{config['model_name']}' not supported.")

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
    logger.info(f"Extracting features for {len(patch_loader)} patch batches")
    for batch in tqdm(patch_loader, desc="Extracting features"):
        patches = batch["patch"].to(device)
        coords = batch["coordinates"]
        embeddings = encoder.extract_embeddings(patches)
        all_embeddings.append(embeddings.cpu())
        all_coordinates.append(coords.cpu())

    embeddings_np = torch.cat(all_embeddings, dim=0).numpy()
    coordinates_np = torch.cat(all_coordinates, dim=0).numpy()
    logger.info(f"Extracted {len(embeddings_np)} features")
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
        f.create_dataset('embeddings', data=embeddings, compression="gzip")
        f.create_dataset('coordinates', data=coordinates, compression="gzip")


def process_single_wsi(wsi_filename: str, config: dict, encoder: torch.nn.Module, encoder_transform: torch.nn.Module):
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
    logger.info(f"Processing WSI: {wsi_filename}")
    device = torch.device(config["device"])

    wsi_path = os.path.join(config["wsi_dir"], wsi_filename)
    coord_filename = wsi_filename.replace(config["wsi_file_extension"], ".h5")
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
        transform=encoder_transform,
    )

    patch_loader = DataLoader(
        patch_dataset,
        batch_size=config["patch_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=(device.type != "cpu")
    )

    logger.info(f"Using device: {device}")
    embeddings, coordinates = extract_features_for_wsi(patch_loader, encoder, device)
    output_path = os.path.join(config["features_dir"], coord_filename)
    save_features_to_h5(embeddings, coordinates, output_path)


def get_wsi_batches_from_gcs(bucket_name: str, gcs_folder_path: str, batch_size: int = 5):
    """
    Lists .svs files from a specified GCS folder path and yields them in batches.
    """
    # Ensure gcs_folder_path ends with a slash if it's not empty, for proper prefix matching
    if gcs_folder_path and not gcs_folder_path.endswith('/'):
        gcs_folder_path += '/'

    logger.info(f"Listing files in bucket: {bucket_name}, folder: {gcs_folder_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    all_wsi_blobs = [
        blob for blob in bucket.list_blobs(prefix=gcs_folder_path)
        if blob.name.endswith('.svs')
    ]

    if not all_wsi_blobs:
        logger.warning(f"No .svs files found in gs://{bucket_name}/{gcs_folder_path}. Exiting.")
        return  # Exit if no files are found

    logger.info(f"Found {len(all_wsi_blobs)} WSI files.")
    # Sort blobs by name for consistent batching across runs
    all_wsi_blobs.sort(key=lambda blob: blob.name)

    current_batch = []
    for blob in all_wsi_blobs:
        full_gcs_path = f"gs://{bucket_name}/{blob.name}"
        current_batch.append(full_gcs_path)
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
    # Yield any remaining files in the last batch
    if current_batch:
        yield current_batch


def download_single_svs_file(bucket_name: str, gcs_file_path: str, local_download_dir: str, download_queue: queue.Queue):
    """
    Downloads a single .svs file from GCS and puts its local path into a queue.
    This function is designed to be run in a separate thread.
    """
    try:
        os.makedirs(local_download_dir, exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob_name = gcs_file_path.replace(f"gs://{bucket_name}/", "")
        blob = bucket.blob(blob_name)
        destination_file_name = os.path.join(local_download_dir, os.path.basename(blob.name))

        logger.info(f"[DOWNLOAD] Starting download of {blob.name} to {destination_file_name}")
        blob.download_to_filename(destination_file_name)
        logger.info(f"[DOWNLOAD] Finished download of {blob.name}")

        download_queue.put(destination_file_name)
    except Exception as e:
        logger.error(f"Error downloading {gcs_file_path}: {e}")
        download_queue.put(None)  # Signal failure for this specific download


def download_single_svs_file_with_cache_control(bucket_name: str, gcs_file_path: str, local_download_dir: str, download_queue: queue.Queue, cache_semaphore: threading.Semaphore):
    """
    Downloads a single .svs file from GCS with cache control.
    This function waits if the cache is full before downloading.
    The semaphore is NOT released here - it will be released by the processing thread
    after the file is removed from the queue.
    """
    try:
        # Acquire semaphore to control cache size
        # This will block if max_cache_size downloads are already in progress
        cache_semaphore.acquire()
        
        os.makedirs(local_download_dir, exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob_name = gcs_file_path.replace(f"gs://{bucket_name}/", "")
        blob = bucket.blob(blob_name)
        destination_file_name = os.path.join(local_download_dir, os.path.basename(blob.name))

        logger.info(f"Starting download of {blob.name} to {destination_file_name}")
        blob.download_to_filename(destination_file_name)
        logger.info(f"Finished download of {blob.name}")

        # Put the file path in the queue (this will block if queue is full)
        # The semaphore is NOT released here - it will be released after processing
        download_queue.put((destination_file_name, cache_semaphore))
        
    except Exception as e:
        logger.error(f"Error downloading {gcs_file_path}: {e}")
        download_queue.put((None, cache_semaphore))  # Signal failure for this specific download
        # Release semaphore on error since we won't be processing this file
        cache_semaphore.release()


def get_coordinate_file_path(wsi_filename: str, local_coordinates_dir: str):
    """
    Determines the expected local path of the .h5 coordinate file for a given WSI filename.
    Assumes coordinate files have the same base name as WSI but with .h5 extension.
    E.g., TCGA-XX-YYYY.svs -> TCGA-XX-YYYY.h5
    """
    base_name = os.path.splitext(os.path.basename(wsi_filename))[0]  # Get filename without .svs from full path
    coord_filename = f"{base_name}.h5"
    full_coord_path = os.path.join(local_coordinates_dir, coord_filename)
    return full_coord_path


def main_batch_processor(config: dict):
    """
    Main function to orchestrate concurrent downloading and processing of WSI files.
    """
    # Extract GCS and local paths from config
    gcs_bucket_name = config["gcs_bucket_name"]
    gcs_wsi_folder = config["gcs_wsi_folder"]
    local_wsi_download_dir = config["local_wsi_download_dir"]
    local_coordinates_dir = config["local_coordinates_dir"]
    batch_size_wsi_listing = config.get("batch_size_wsi_listing", 5)
    max_download_workers = config.get("max_download_workers", 3)
    max_cache_size = config.get("max_cache_size", 2)  # Maximum files in cache
    device = torch.device(config["device"])
    os.makedirs(local_wsi_download_dir, exist_ok=True)
    download_queue = queue.Queue(maxsize=max_cache_size)  # Queue with size limit
    cache_semaphore = threading.Semaphore(max_cache_size)  # Control cache size

    logger.info("\n--- Starting Concurrent WSI Processing ---")
    logger.info(f"Cache limit: {max_cache_size} files")
    logger.info(f"Download workers: {max_download_workers}")
    logger.info(f"Cache control: Downloads will pause when {max_cache_size} files are in local cache")

    # Get total number of files to process for completion check
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket_name)
    total_files_to_process = len([
        blob for blob in bucket.list_blobs(prefix=gcs_wsi_folder)
        if blob.name.endswith('.svs')
    ])
    logger.info(f"Total WSI files identified for processing: {total_files_to_process}")

    # Use a ThreadPoolExecutor for parallel downloads
    encoder, encoder_transform = get_encoder(config, device)
    with ThreadPoolExecutor(max_workers=max_download_workers) as download_executor:
        download_futures = []  # To keep track of submitted download tasks

        # Submit all download tasks concurrently in the background
        for batch_gcs_paths in get_wsi_batches_from_gcs(gcs_bucket_name, gcs_wsi_folder, batch_size_wsi_listing):
            for gcs_path in batch_gcs_paths:
                future = download_executor.submit(
                    download_single_svs_file_with_cache_control,
                    gcs_bucket_name,
                    gcs_path,
                    local_wsi_download_dir,
                    download_queue,
                    cache_semaphore
                )
                download_futures.append(future)

        processed_count = 0
        # Main thread loop: continuously try to get files from the queue and process them
        while processed_count < total_files_to_process:
            try:
                # Get a downloaded file path from the queue (with a timeout to prevent infinite blocking)
                # If queue is empty and downloads are still running, it will wait
                local_wsi_file_path, cache_semaphore = download_queue.get(timeout=100)  # Increased timeout
                if local_wsi_file_path is None:  # Handle potential download errors
                    logger.warning("[MAIN_LOOP] Skipping processing due to a previous download error.")
                    processed_count += 1
                    # Semaphore was already released in the download function for errors
                    continue

                logger.info(f"\n[MAIN_LOOP] Processing local file: {local_wsi_file_path}")

                wsi_filename = os.path.basename(local_wsi_file_path)
                
                # Update config with local paths for this processing
                local_config = config.copy()
                local_config["wsi_dir"] = local_wsi_download_dir
                local_config["coordinates_dir"] = local_coordinates_dir
                
                process_single_wsi(wsi_filename, local_config, encoder, encoder_transform)

                # Remove the local WSI data after processing
                if os.path.exists(local_wsi_file_path):
                    os.remove(local_wsi_file_path)
                    logger.info(f"[CLEANUP] Removed local WSI file: {local_wsi_file_path}")
                else:
                    logger.warning(f"[CLEANUP] WSI file not found for cleanup: {local_wsi_file_path}")

                # Release the semaphore AFTER processing is complete
                # This allows the next download to start, maintaining proper cache size control
                cache_semaphore.release()
                logger.info(f"[CACHE] Released semaphore, cache slot available for next download")

                processed_count += 1
            except queue.Empty:
                # Check if all downloads are truly finished and queue is empty, then break
                if all(f.done() for f in download_futures) and download_queue.empty():
                    logger.info("[MAIN_LOOP] All downloads complete and queue is empty. Exiting processing loop.")
                    break
                logger.info("[MAIN_LOOP] Queue is temporarily empty, waiting for more downloads or completion...")
                time.sleep(1)  # Wait a bit before retrying to avoid busy-waiting
            except Exception as e:
                logger.error(f"[MAIN_LOOP] An unexpected error occurred during processing: {e}")
                processed_count += 1  # Ensure loop progresses even on errors

    logger.info("\n--- All WSI files processed (or attempted). ---")


def run_gcs_feature_extraction(config_path: str):
    """
    Executes the full feature extraction pipeline for WSIs stored in Google Cloud Storage.

    Args:
        config_path (str): Path to the YAML configuration file containing GCS paths, model, and processing settings.
    """
    config = load_config(config_path)
    google_credentials_abs_path = os.path.expanduser(config["google_application_credentials"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_abs_path
    # Download coordinate files first
    logger.debug(f"\n--- Downloading Patch Coordinate Files from {config['gcs_bucket_name']}/{config['gcs_coordinates_folder']} ---")
    os.makedirs(config["local_coordinates_dir"], exist_ok=True)
    
    # Use gcloud command to download coordinate files
    try:
        subprocess.run([
            "gcloud", "storage", "cp", 
            f"gs://{config['gcs_bucket_name']}/{config['gcs_coordinates_folder']}*.h5", 
            config["local_coordinates_dir"]
        ], check=True)
        logger.debug(f"Coordinate files downloaded to: {config['local_coordinates_dir']}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading coordinate files: {e}")
        return

    main_batch_processor(config)


if __name__ == '__main__':
    """
    A script to extract vision features from WSIs stored in Google Cloud Storage.

    Args:
        --config (str): Path to the configuration YAML file containing GCS paths, model, and processing settings.

    Example:
        python extract_vision_features_batch.py --config configs/vision_feature_extraction_gcs.yml
    
    The script will:

    """
    parser = argparse.ArgumentParser(description="Run WSI feature extraction with GCS integration.")
    parser.add_argument("--config", type=str, default="configs/vision_feature_extraction_gcs.yml",
                        help="Path to the configuration YAML file.")
    args = parser.parse_args()
    logger.info(f"Running feature extraction with config: {args.config}")
    run_gcs_feature_extraction(args.config)
    logger.info("\nGCS feature extraction process completed.") 