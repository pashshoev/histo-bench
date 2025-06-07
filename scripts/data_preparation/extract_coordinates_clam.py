import argparse
import os
import subprocess

# --- Configuration for CLAM Integration ---
# Absolute path to the CLAM virtual environment
CLAM_VENV_PATH = os.path.expanduser("~/PycharmProjects/CLAM/.venv")

# Absolute path to the actual CLAM script for patch creation
CLAM_SCRIPT_PATH = os.path.expanduser("~/PycharmProjects/CLAM/create_patches_fp.py")

# Path to the Python executable within CLAM's venv
CLAM_PYTHON_EXEC = os.path.join(CLAM_VENV_PATH, "bin", "python")

TCGA_PRESET_PATH = os.path.expanduser("~/PycharmProjects/CLAM/presets/tcga.csv")

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
):
    """
    Calls the CLAM script to extract patch coordinates from WSIs.

    Args:
        source_dir (str): Path to the directory containing WSI files (DATA_DIRECTORY).
        save_dir (str): Path to the directory where extracted coordinates will be saved (RESULTS_DIRECTORY).
        patch_size (int): The size of the patches to extract. Defaults to 256.
    Returns:
        True if successful, False otherwise.
    """
    print(f"\n--- Initiating CLAM Coordinate Extraction ---")

    # Ensure output directory exists before CLAM tries to write to it
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Construct the command for subprocess.run
        command = [
            CLAM_PYTHON_EXEC,
            CLAM_SCRIPT_PATH,
            "--source",
            source_dir,
            "--save_dir",
            save_dir,
            "--patch_size",
            str(patch_size),
            "--preset",
            TCGA_PRESET_PATH,
            "--seg",  # Flag, no value
            "--patch",  # Flag, no value
            "--stitch",  # Flag, no value
        ]

        print(f"Executing command: {' '.join(command)}")

        # Run the subprocess
        result = subprocess.run(
            command,
            check=True,  # Raises CalledProcessError if return code is non-zero
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode stdout/stderr as text
            encoding="utf-8",  # Explicitly set encoding for consistent output
        )

        print("\n--- CLAM Script Output (STDOUT) ---")
        print(result.stdout)

        if result.stderr:
            print("\n--- CLAM Script Errors (STDERR) ---")
            print(result.stderr)

        print(f"--- CLAM Coordinate Extraction Successful for {source_dir} ---")
    except Exception as e:
        print(f"\n--- An unexpected error occurred during CLAM call: {e} ---")
        return False
    return True

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    source_dir = os.path.abspath(args.source_dir)
    save_dir = os.path.abspath(args.save_dir)
    print("\nAttempting to call CLAM with provided arguments...")
    try:
        run_clam_coordinate_extraction(
            source_dir=source_dir,
            save_dir=save_dir,
            patch_size=args.patch_size,
        )
        print("\nCLAM call completed successfully.")
    except Exception as e:
        print(f"\nCLAM call failed: {e}")
    finally:
        pass # Add cleanup logic here if needed (e.g., shutil.rmtree)
