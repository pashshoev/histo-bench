import pandas as pd
import subprocess
import os
import shutil
import argparse


def filter_manifest_by_downloaded(manifest_df: pd.DataFrame, destination_dir: str,
                                  filename_col: str = 'filename') -> pd.DataFrame:
    """
    Filters manifest DataFrame for files already downloaded to destination_dir.
    """
    if not os.path.exists(destination_dir):
        print(f"Warning: Destination '{destination_dir}' not found. Returning original manifest.")
        return manifest_df
    expected_file_paths = manifest_df[filename_col].apply(lambda fn: os.path.join(destination_dir, fn))
    file_exists_series = expected_file_paths.apply(os.path.exists)

    filtered_manifest_df = manifest_df[~file_exists_series].copy()

    num_removed = len(manifest_df) - len(filtered_manifest_df)
    if num_removed > 0:
        print(f"Removed {num_removed} entries from manifest (already exist in {destination_dir}).")

    return filtered_manifest_df

def download_slides_from_manifest_subset(
        manifest_subset_df: pd.DataFrame,
        download_dir: str
) -> list[str]:

    mini_manifest_path = "temp_mini_manifest.tsv"
    manifest_subset_df.to_csv(mini_manifest_path, sep="\t", index=False)

    os.makedirs(download_dir, exist_ok=True)

    try:
        subprocess.run([
            "gdc-client", "download",
            "-m", mini_manifest_path,
            "-d", download_dir
        ], check=True)
    finally:
        if os.path.exists(mini_manifest_path):
            os.remove(mini_manifest_path)

    return manifest_subset_df['id'].tolist()


def verify_and_move_downloads(
        targeted_file_ids: list[str],
        manifest_df: pd.DataFrame,
        download_dir: str,
        destination_dir: str
) -> list[str]:
    os.makedirs(destination_dir, exist_ok=True)
    unsuccessful_downloads = []

    for file_id in targeted_file_ids:
        row = manifest_df[manifest_df['id'] == file_id]
        if row.empty:
            unsuccessful_downloads.append(f"ID_NOT_FOUND_IN_MANIFEST: {file_id}")
            continue

        original_filename = row.iloc[0]['filename']

        expected_download_folder = os.path.join(download_dir, file_id)
        expected_svs_path = os.path.join(expected_download_folder, original_filename)
        expected_partial_path = expected_svs_path + ".partial"

        if os.path.exists(expected_svs_path):
            try:
                final_destination_path = os.path.join(destination_dir, original_filename)
                shutil.move(expected_svs_path, final_destination_path)
                shutil.rmtree(expected_download_folder)
            except Exception as e:
                print(f"Exception while moving partial file to destination folder: {e}")
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
    destination_dir: str
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

    manifest = pd.read_csv(manifest_file_path, sep="\t")
    slides_to_download = manifest.iloc[start_index:end_index]
    slides_to_download = filter_manifest_by_downloaded(slides_to_download, destination_dir, filename_col='filename')

    if slides_to_download.empty:
        print("No slides selected for download.")
        return []
    print(f"Downloading {len(slides_to_download)} slides...")

    targeted_file_ids = download_slides_from_manifest_subset(
        manifest_subset_df=slides_to_download,
        download_dir=download_dir
    )

    unsuccessful_filenames = verify_and_move_downloads(
        targeted_file_ids=targeted_file_ids,
        manifest_df=manifest,
        download_dir=download_dir,
        destination_dir=destination_dir
    )
    return unsuccessful_filenames


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download slides from a GDC manifest, verify, "
                    "move successful downloads, and log unsuccessful ones."
    )
    parser.add_argument(
        "-m", "--manifest_file",
        type=str,
        required=True,
        help="Path to the input GDC manifest file (e.g., gdc_manifest.txt)."
    )
    parser.add_argument(
        "-s", "--start_index",
        type=int,
        default=0,
        help="Index of the first SVS file to download."
    )
    parser.add_argument(
        "-e", "--end_index",
        type=int,
        default=-1,
        help="Index of the last SVS file to download."
    )
    parser.add_argument(
        "-d", "--download_dir",
        type=str,
        default="gdc_downloads",
        help="Temporary directory where gdc-client downloads files (default: gdc_downloads)."
    )
    parser.add_argument(
        "-D", "--destination_dir",
        type=str,
        default="final_svs_slides",
        help="Final directory where successfully downloaded .svs files will be moved (default: final_svs_slides)."
    )
    parser.add_argument(
        "-u", "--unsuccessful_log",
        type=str,
        default="unsuccessful_downloads.txt",
        help="Path to a text file to log unsuccessful downloads (default: unsuccessful_downloads.txt)."
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
            destination_dir=args.destination_dir
        )

        if unsuccessful_downloads:
            with open(args.unsuccessful_log, 'w') as f:
                for filename in unsuccessful_downloads:
                    f.write(f"{filename}\n")
            print(f"Unsuccessful downloads logged to: {args.unsuccessful_log}")
        else:
            print("All targeted slides were successfully downloaded and moved.")
            if os.path.exists(args.unsuccessful_log):
                os.remove(args.unsuccessful_log)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error during gdc-client execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
