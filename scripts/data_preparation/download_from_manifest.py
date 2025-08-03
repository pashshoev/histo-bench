import argparse
import os
import shutil
import subprocess

import pandas as pd

from loguru import logger

def drop_already_downloaded_slides(
    manifest_df: pd.DataFrame, 
    destination_dir: str, 
    filename_col: str = "filename"
) -> pd.DataFrame:
    """
    Drops slides from manifest DataFrame that already exist in destination_dir.

    Args:
        manifest_df (pd.DataFrame): The manifest dataframe to drop slides from.
        destination_dir (str): The directory to check for existing slides.
        filename_col (str): The column name containing the filenames.

    Returns:
        pd.DataFrame: The manifest dataframe with slides that do not exist in destination_dir dropped.
    """
    if not os.path.exists(destination_dir):
        logger.warning(
            f"Destination '{destination_dir}' not found. Returning original manifest."
        )
        return manifest_df
    expected_file_paths = manifest_df[filename_col].apply(
        lambda fn: os.path.join(destination_dir, fn)
    )
    file_exists_series = expected_file_paths.apply(os.path.exists)

    filtered_manifest_df = manifest_df[~file_exists_series].copy()

    num_removed = len(manifest_df) - len(filtered_manifest_df)
    if num_removed > 0:
        logger.info(
            f"Removed {num_removed} entries from manifest (already exist in {destination_dir})."
        )

    return filtered_manifest_df


def download_slides_from_manifest(
    manifest_df: pd.DataFrame, 
    download_dir: str
) -> list[str]:
    """
    Downloads slides from a manifest dataframe using gdc-client.

    Args:
        manifest_df (pd.DataFrame): The manifest dataframe to download slides from.
        download_dir (str): The directory to download the slides to.

    Returns:
        list[str]: A list of the ids of the slides that were downloaded.
    """
    manifest_path = "temp_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)

    os.makedirs(download_dir, exist_ok=True)

    try:
        subprocess.run(
            ["gdc-client", "download", "-m", manifest_path, "-d", download_dir],
            check=True,
        )
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    return manifest_df["id"].tolist()


def verify_and_move_downloads(
    targeted_file_ids: list[str],
    manifest_df: pd.DataFrame,
    download_dir: str,
    destination_dir: str,
) -> list[str]:
    """
    Verifies and moves downloaded slides to the destination directory.

    Args:
        targeted_file_ids (list[str]): A list of the ids of the slides that were downloaded.
        manifest_df (pd.DataFrame): The manifest dataframe to verify and move slides from.
        download_dir (str): The directory to download the slides to.
        destination_dir (str): The directory to move the slides to.

    Returns:
        list[str]: A list of the ids of the slides that were not successfully downloaded or moved.
    """
    os.makedirs(destination_dir, exist_ok=True)
    unsuccessful_downloads = []

    for file_id in targeted_file_ids:
        row = manifest_df[manifest_df["id"] == file_id]
        if row.empty:
            unsuccessful_downloads.append(f"ID_NOT_FOUND_IN_MANIFEST: {file_id}")
            continue

        original_filename = row.iloc[0]["filename"]

        expected_download_folder = os.path.join(download_dir, file_id)
        expected_svs_path = os.path.join(expected_download_folder, original_filename)
        expected_partial_path = expected_svs_path + ".partial"

        if os.path.exists(expected_svs_path):
            try:
                final_destination_path = os.path.join(
                    destination_dir, original_filename
                )
                shutil.move(expected_svs_path, final_destination_path)
                shutil.rmtree(expected_download_folder)
            except Exception as e:
                logger.exception(f"Exception while moving partial file to destination folder: {e}")
                unsuccessful_downloads.append(original_filename)
        elif os.path.exists(expected_partial_path):
            unsuccessful_downloads.append(original_filename)
        else:
            unsuccessful_downloads.append(original_filename)

    return unsuccessful_downloads


def download_and_move(
    manifest_file_path: str,
    start_index: int,
    end_index: int,
    download_dir: str,
    destination_dir: str,
) -> list[str]:
    """
    Orchestrates the download and moving of SVS files.

    Args:
        manifest_file_path (str): Path to the input GDC manifest file.
        start_index (int): Index of the first SVS file to download.
        end_index (int): Index of the last SVS file to download.
        download_dir (str): Temporary directory for gdc-client downloads.
        destination_dir (str): Final directory for successfully moved .svs files.

    Returns:
        list[str]: A list of original filenames that were not successfully downloaded or moved.
    """
    if not os.path.exists(manifest_file_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_file_path}")

    # Read manifest subset and drop already downloaded slides
    manifest = pd.read_csv(manifest_file_path, sep="\t")
    slides_to_download = manifest.iloc[start_index:end_index]
    slides_to_download = drop_already_downloaded_slides(
        slides_to_download, destination_dir, filename_col="filename"
    )

    if slides_to_download.empty:
        logger.info("No slides selected to download.")
        return []
    logger.info(f"Downloading {len(slides_to_download)} slides...")

    # Download slides
    targeted_file_ids = download_slides_from_manifest(
        manifest_df=slides_to_download, download_dir=download_dir
    )

    # Verify and move downloads
    unsuccessful_filenames = verify_and_move_downloads(
        targeted_file_ids=targeted_file_ids,
        manifest_df=manifest,
        download_dir=download_dir,
        destination_dir=destination_dir,
    )
    return unsuccessful_filenames


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download slides from a GDC manifest, verify, "
        "move successful downloads, and log unsuccessful ones."
    )
    parser.add_argument(
        "-m",
        "--manifest_file",
        type=str,
        required=True,
        help="Path to the input GDC manifest file (e.g., gdc_manifest.txt).",
    )
    parser.add_argument(
        "-s",
        "--start_index",
        type=int,
        default=0,
        help="Index of the first SVS file to download.",
    )
    parser.add_argument(
        "-e",
        "--end_index",
        type=int,
        default=-1,
        help="Index of the last SVS file to download.",
    )
    parser.add_argument(
        "-d",
        "--download_dir",
        type=str,
        default="gdc_downloads",
        help="Temporary directory where gdc-client downloads files (default: gdc_downloads).",
    )
    parser.add_argument(
        "-D",
        "--destination_dir",
        type=str,
        default="final_svs_slides",
        help="Final directory where successfully downloaded .svs files will be moved (default: final_svs_slides).",
    )
    parser.add_argument(
        "-u",
        "--unsuccessful_log",
        type=str,
        default="unsuccessful_downloads.txt",
        help="Path to a text file to log unsuccessful downloads (default: unsuccessful_downloads.txt).",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    try:
        unsuccessful_downloads = download_and_move(
            manifest_file_path=args.manifest_file,
            start_index=args.start_index,
            end_index=args.end_index,
            download_dir=args.download_dir,
            destination_dir=args.destination_dir,
        )

        if unsuccessful_downloads:
            with open(args.unsuccessful_log, "w") as f:
                for filename in unsuccessful_downloads:
                    f.write(f"{filename}\n")
            logger.info(f"Unsuccessful downloads logged to: {args.unsuccessful_log}")
        else:
            logger.info("All targeted slides were successfully downloaded and moved.")
            if os.path.exists(args.unsuccessful_log):
                os.remove(args.unsuccessful_log)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during gdc-client execution: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":  
    main()
