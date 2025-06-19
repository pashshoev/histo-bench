#!/usr/bin/env python3
"""
Helper script to run feature extraction with either local or GCS-based processing.
This script provides a simple interface to choose between the two approaches.
"""

import argparse
import subprocess
import sys
import os


def run_local_extraction(config_path=None):
    """Run local feature extraction."""
    if config_path is None:
        config_path = "configs/vision_feature_extraction.yml"
    
    print("Running local feature extraction...")
    print(f"Using config: {config_path}")
    
    cmd = [sys.executable, "extract_vision_features_local.py", "--config", config_path]
    return subprocess.run(cmd)


def run_gcs_extraction(config_path=None):
    """Run GCS-based feature extraction."""
    if config_path is None:
        config_path = "configs/vision_feature_extraction_gcs.yml"
    
    print("Running GCS-based feature extraction...")
    print(f"Using config: {config_path}")
    
    cmd = [sys.executable, "extract_vision_features_batch.py", "--config", config_path]
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Run feature extraction with either local or GCS-based processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run local extraction with default config
  python run_feature_extraction.py --mode local
  
  # Run GCS extraction with default config
  python run_feature_extraction.py --mode gcs
  
  # Run with custom config
  python run_feature_extraction.py --mode local --config my_config.yml
  python run_feature_extraction.py --mode gcs --config my_gcs_config.yml
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["local", "gcs"], 
        required=True,
        help="Processing mode: 'local' for local files, 'gcs' for Google Cloud Storage"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to custom configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if required files exist
    if args.mode == "local":
        script_path = "extract_vision_features_local.py"
        default_config = "configs/vision_feature_extraction.yml"
    else:  # gcs
        script_path = "extract_vision_features_batch.py"
        default_config = "configs/vision_feature_extraction_gcs.yml"
    
    if not os.path.exists(script_path):
        print(f"Error: Script {script_path} not found!")
        sys.exit(1)
    
    config_path = args.config or default_config
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found!")
        sys.exit(1)
    
    # Run the appropriate extraction
    if args.mode == "local":
        result = run_local_extraction(config_path)
    else:
        result = run_gcs_extraction(config_path)
    
    if result.returncode == 0:
        print("\n✅ Feature extraction completed successfully!")
    else:
        print(f"\n❌ Feature extraction failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main() 