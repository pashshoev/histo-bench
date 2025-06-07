import argparse
import os
import subprocess
import sys  # Import sys to get the current Python executable

# You might need to adjust "CLAM" if the folder name is different.
CLAM_SCRIPT_PATH = os.path.join("/CLAM", "create_patches_fp.py")

# Absolute path to the TCGA preset file within the cloned CLAM repo.
TCGA_PRESET_PATH = os.path.join("tcga.csv")


def get_args():
    parser = argparse.ArgumentParser(
        description="Run CLAM coordinate extraction via command line."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to the directory containing WSI files for CLAM input."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to the directory where CLAM will save extracted coordinates."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=224,
        help="The size of the patches to extract (e.g., 224)."
    )

    return parser.parse_args()


def run_clam_coordinate_extraction(
        source_dir: str, save_dir: str, patch_size: int = 224
) -> bool:  # Added return type hint for clarity
    """
    Calls the CLAM script to extract patch coordinates from WSIs.

    Args:
        source_dir (str): Path to the directory containing WSI files (DATA_DIRECTORY).
        save_dir (str): Path to the directory where extracted coordinates will be saved (RESULTS_DIRECTORY).
        patch_size (int): The size of the patches to extract. Defaults to 224.
    Returns:
        True if successful, False otherwise.
    """
    print(f"\n--- Initiating CLAM Coordinate Extraction ---")
    print(f"CLAM Script Path: {CLAM_SCRIPT_PATH}")
    print(f"Using Python executable: {sys.executable}")  # Show which python is being used
    print(f"Source Directory (WSI Input): {source_dir}")
    print(f"Save Directory (Coordinate Output): {save_dir}")
    print(f"Patch Size: {patch_size}")
    print(f"Preset Path: {TCGA_PRESET_PATH}")

    # Ensure output directory exists before CLAM tries to write to it
    os.makedirs(save_dir, exist_ok=True)
    print(f"Ensured output directory exists: {save_dir}")

    try:
        # Construct the command for subprocess.run
        command = [
            sys.executable,  # Use the current Python interpreter
            CLAM_SCRIPT_PATH,
            "--source",
            source_dir,
            "--save_dir",
            save_dir,
            "--patch_size",
            str(patch_size),
            "--preset",
            "tcga.csv",
            "--seg",
            "--patch",
            "--stitch",
        ]

        print(f"Executing command: {' '.join(command)}")

        # Run the subprocess
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        print("\n--- CLAM Script Output (STDOUT) ---")
        print(result.stdout)

        if result.stderr:
            print("\n--- CLAM Script Errors (STDERR) ---")
            print(result.stderr)

        print(f"--- CLAM Coordinate Extraction Successful for {source_dir} ---")
        return True  # Indicate success

    except subprocess.CalledProcessError as e:
        print(f"\n--- Error: CLAM Script Failed! ---")
        print(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}.")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"\n--- An unexpected error occurred during CLAM call: {e} ---")
        return False


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory '{args.source_dir}' does not exist.")
        sys.exit(1)

    # Convert paths to absolute paths for clarity and robustness,
    source_dir_abs = os.path.abspath(args.source_dir)
    save_dir_abs = os.path.abspath(args.save_dir)

    print("\nAttempting to call CLAM with provided arguments...")
    success = run_clam_coordinate_extraction(
        source_dir=source_dir_abs,
        save_dir=save_dir_abs,
        patch_size=args.patch_size,
    )

    if success:
        print("\nCLAM call completed successfully.")
    else:
        print("\nCLAM call failed. Check logs for details.")

