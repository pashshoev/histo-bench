import boto3
import os
import requests
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Set your AWS credentials and S3 bucket information
# You should manage these securely, e.g., using environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# The "folder" is the S3 key prefix
S3_FOLDER_PREFIX = 'TCGA-Lung/features/ResNet50/'
LOCAL_DOWNLOAD_DIR = 'data/'

def get_presigned_urls_for_folder(bucket_name, folder_prefix, expiration_seconds=3600):
    """
    Lists all objects in a given S3 folder and generates a presigned URL for each.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_prefix (str): The prefix representing the "folder" in S3.
        expiration_seconds (int): The number of seconds the URLs are valid for.

    Returns:
        A dictionary where keys are object keys and values are presigned URLs.
    """
    s3_client = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    presigned_urls = {}
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)

    for page in pages:
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            object_key = obj['Key']
            
            # Skip the folder itself
            if object_key == folder_prefix:
                continue
                
            try:
                # Generate the presigned URL for the object
                url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': object_key},
                    ExpiresIn=expiration_seconds
                )
                presigned_urls[object_key] = url
            except Exception as e:
                print(f"Error generating URL for {object_key}: {e}")

    return presigned_urls

def download_file(url, local_path, s3_key):
    """
    Downloads a single file from a presigned URL.

    Args:
        url (str): The presigned URL.
        local_path (str): The local path to save the file.
        s3_key (str): The S3 key for logging purposes.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {s3_key}")
        return s3_key, True
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return s3_key, False

def download_files_in_parallel(urls, local_dir, max_workers=10):
    """
    Downloads files from a list of URLs in parallel using a thread pool.

    Args:
        urls (dict): A dictionary of S3 keys and their presigned URLs.
        local_dir (str): The local base directory to save files.
        max_workers (int): The maximum number of threads to use.
    """
    if not urls:
        print("No URLs to download.")
        return

    # Use ThreadPoolExecutor to manage parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, os.path.join(local_dir, key), key): key for key, url in urls.items()}
        
        for future in as_completed(futures):
            key, success = future.result()
            if not success:
                print(f"Download failed for {key}")

if __name__ == '__main__':
    # Step 1: Generate presigned URLs
    print("Generating presigned URLs...")
    urls = get_presigned_urls_for_folder(BUCKET_NAME, S3_FOLDER_PREFIX)

    if urls:
        print(f"Found {len(urls)} files to download.")
        print("\nStarting parallel download...")
        # Step 2: Download files in parallel
        download_files_in_parallel(urls, LOCAL_DOWNLOAD_DIR)
        print("\nAll download tasks completed.")
    else:
        print("No URLs to download. Exiting.")
