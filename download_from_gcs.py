import os
import requests
from google.cloud import storage
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

load_dotenv()

# ======== CONFIG ========
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'histo-bench')
GCS_FOLDER_PREFIX = 'TCGA-RCC/features/ResNet50/'  # "Folder" path in GCS
LOCAL_DOWNLOAD_DIR = 'data/'  # Local base dir to save files
SIGNED_URL_EXPIRATION_MINUTES = 60 * 24  # URL expiration (max 7 days)

# ========================

def get_signed_urls_for_folder(bucket_name, folder_prefix, expiration_minutes=60):
    """
    List all blobs in a GCS folder and generate signed URLs for each.

    Args:
        bucket_name (str): Name of the GCS bucket.
        folder_prefix (str): Prefix representing the "folder" in GCS.
        expiration_minutes (int): Expiration time in minutes for signed URLs.

    Returns:
        dict: {blob_name: signed_url}
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=folder_prefix)
    signed_urls = {}

    for blob in blobs:
        # Skip "folders"
        if blob.name.endswith('/'):
            continue

        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(minutes=expiration_minutes),
                method="GET"
            )
            signed_urls[blob.name] = url
        except Exception as e:
            print(f"Error generating signed URL for {blob.name}: {e}")

    return signed_urls


def download_file(url, local_path, blob_name):
    """
    Download a single file from a signed URL.

    Args:
        url (str): The signed URL.
        local_path (str): Where to save the file locally.
        blob_name (str): GCS object name (for logging).
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Create local directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Successfully downloaded: {blob_name}")
        return blob_name, True

    except Exception as e:
        print(f"Error downloading {blob_name}: {e}")
        return blob_name, False


def download_files_in_parallel(urls, local_dir, max_workers=10):
    """
    Download multiple files from signed URLs in parallel.

    Args:
        urls (dict): {blob_name: signed_url}
        local_dir (str): Base directory to save files.
        max_workers (int): Number of threads.
    """
    if not urls:
        print("No URLs to download.")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                download_file,
                url,
                os.path.join(local_dir, blob_name),
                blob_name
            ): blob_name
            for blob_name, url in urls.items()
        }

        for future in as_completed(futures):
            blob_name, success = future.result()
            if not success:
                print(f"Download failed for {blob_name}")


if __name__ == '__main__':
    print("Generating signed URLs from GCS...")

    urls = get_signed_urls_for_folder(GCS_BUCKET_NAME, GCS_FOLDER_PREFIX, SIGNED_URL_EXPIRATION_MINUTES)

    if urls:
        print(f"Found {len(urls)} files to download.")
        print("Starting parallel download...")

        download_files_in_parallel(urls, LOCAL_DOWNLOAD_DIR, max_workers=10)

        print("\nAll downloads completed.")
    else:
        print("No files found to download.")
