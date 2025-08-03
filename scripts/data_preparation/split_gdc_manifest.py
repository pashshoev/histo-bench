import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def get_case_id(filename: str):
    """
    Extracts the case ID from a filename.
    Assumes case ID is the first three components separated by hyphens.
    """
    id_components = filename.split("-")[:3]
    case_id = "-".join(id_components)
    return case_id

def split_manifest(
    manifest_file_path: str,
    output_dir: str = "manifest_results",
    test_size: float = 0.2
):
    """
    Reads a GDC manifest file, processes it to keep only the largest
    entry per case_id, and splits the data into train and test sets.
    The resulting sets are saved as tab-separated CSV files in the specified
    output directory.

    Args:
        manifest_file_path (str): Path to the input GDC manifest file.
        output_dir (str): Directory to save the split manifest files.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    if not os.path.exists(manifest_file_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_file_path}")

    manifest = pd.read_csv(manifest_file_path, sep="\t")

    manifest["case_id"] = manifest["filename"].apply(get_case_id)

    unique_manifest = manifest.loc[manifest.groupby('case_id')['size'].idxmax()]

    train, test = train_test_split(unique_manifest, test_size=test_size, random_state=42)

    print(f"Size of train set: {len(train)}")
    print(f"Size of test set: {len(test)}")

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.txt")
    test_path = os.path.join(output_dir, "test.txt")

    print(f"Saving train set to: {train_path}")
    train.to_csv(train_path, sep="\t", index=False)
    print(f"Saving test set to: {test_path}")
    test.to_csv(test_path, sep="\t", index=False)

    print("\nManifest splitting complete.")
    return train_path, test_path

def parse_arguments():
    """
    Parses command-line arguments for the manifest splitter script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process a GDC manifest file, filter by highest size per case_id, "
                    "and split into train and test sets."
    )
    parser.add_argument(
        "-m", "--manifest_file",
        type=str,
        required=True,
        help="Path to the input GDC manifest file (e.g., gdc_manifest.txt)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="manifest_results",
        help="Directory to save the split manifest files (default: manifest_results)."
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)."
    )

    args = parser.parse_args()

    # Basic validation for sizes
    if not (0 <= args.test_size < 1):
        parser.error("test_size must be between 0 and less than 1.")

    return args

def main():
    """
    Main function to run the manifest splitting process.
    Parses arguments and calls the split_manifest function.
    """
    args = parse_arguments()

    try:
        split_manifest(
            manifest_file_path=args.manifest_file,
            output_dir=args.output_dir,
            test_size=args.test_size
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
