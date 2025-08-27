#!/usr/bin/env python3
"""
Script to find files that are in the manifest but not uploaded to Google Cloud Storage.
This script handles both GCS listing and missing file detection in one go.
"""

import subprocess
import csv
import argparse
import os
from pathlib import Path

def get_gcs_files(bucket_path):
    """Get all files from GCS bucket path."""
    try:
        # Run gcloud command to list files
        result = subprocess.run(
            ['gcloud', 'storage', 'ls', bucket_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Split the output into lines and filter out empty lines
        files = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error running gcloud command: {e}")
        return []

def extract_filename_from_gcs_path(gcs_path, bucket_prefix):
    """Extract just the filename from the full GCS path."""
    if gcs_path.startswith(bucket_prefix):
        return gcs_path[len(bucket_prefix):]
    return gcs_path

def read_manifest(manifest_path):
    """Read manifest file and return list of filenames."""
    filenames = []
    try:
        with open(manifest_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            # Skip header if it exists
            header = next(reader)
            if header[0] == 'id':
                # This is a proper manifest with header
                for row in reader:
                    if len(row) >= 2:
                        filenames.append(row[1])  # filename is in second column
            else:
                # No header, first row is data
                filenames.append(header[1])  # filename is in second column
                for row in reader:
                    if len(row) >= 2:
                        filenames.append(row[1])
    except Exception as e:
        print(f"Error reading manifest file: {e}")
        return []
    
    return filenames

def find_missing_files(manifest_filenames, gcs_files, bucket_prefix):
    """Find files that are in manifest but not in GCS."""
    # Extract just filenames from GCS paths
    gcs_filenames = set()
    for gcs_path in gcs_files:
        filename = extract_filename_from_gcs_path(gcs_path, bucket_prefix)
        # Remove .h5 extension if present (since GCS files are .h5 but manifest has .svs)
        if filename.endswith('.h5'):
            filename = filename[:-3] + '.svs'
        gcs_filenames.add(filename)
    
    # Find missing files
    missing_files = []
    for manifest_filename in manifest_filenames:
        if manifest_filename not in gcs_filenames:
            missing_files.append(manifest_filename)
    
    return missing_files

def save_missing_manifest(manifest_path, missing_filenames, output_path):
    """Save missing files to a new manifest file."""
    try:
        # Read original manifest to get full rows
        original_rows = []
        with open(manifest_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            original_rows = list(reader)
        
        # Find rows with missing filenames
        missing_rows = []
        header = None
        
        for i, row in enumerate(original_rows):
            if i == 0 and row[0] == 'id':
                # This is the header
                header = row
                missing_rows.append(header)
            elif len(row) >= 2 and row[1] in missing_filenames:
                missing_rows.append(row)
        
        # Write missing files to new manifest
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(missing_rows)
        
        print(f"Saved {len(missing_rows) - (1 if header else 0)} missing files to {output_path}")
        
    except Exception as e:
        print(f"Error saving missing manifest: {e}")

def main():
    parser = argparse.ArgumentParser(description='Find files missing from GCS bucket')
    parser.add_argument('--gcs_bucket_path', help='GCS bucket path (e.g., gs://histo-bench/TCGA-LUSC/features/ResNet50)')
    parser.add_argument('--manifest_path', help='Path to the manifest file')
    parser.add_argument('--output', '-o', default='missing_files_manifest.txt', 
                       help='Output path for missing files manifest (default: missing_files_manifest.txt)')
    
    args = parser.parse_args()
    
    # Extract bucket prefix for filename extraction
    bucket_prefix = args.gcs_bucket_path.rstrip('/') + '/'
    
    print("=== FINDING MISSING FILES ===")
    print(f"GCS Bucket Path: {args.gcs_bucket_path}")
    print(f"Manifest Path: {args.manifest_path}")
    print(f"Output File: {args.output}")
    print()
    
    # Step 1: Get files from GCS bucket
    print("[STEP 1] Fetching files from GCS bucket...")
    gcs_files = get_gcs_files(args.gcs_bucket_path)
    if not gcs_files:
        print("Error: No files found in GCS bucket or failed to access bucket")
        return
    
    print(f"Found {len(gcs_files)} files in GCS bucket")
    
    # Step 2: Read manifest file
    print("[STEP 2] Reading manifest file...")
    manifest_filenames = read_manifest(args.manifest_path)
    if not manifest_filenames:
        print("Error: No files found in manifest or failed to read manifest")
        return
    
    print(f"Found {len(manifest_filenames)} files in manifest")
    
    # Step 3: Find missing files
    print("[STEP 3] Finding missing files...")
    missing_files = find_missing_files(manifest_filenames, gcs_files, bucket_prefix)
    
    print(f"Found {len(missing_files)} missing files")
    
    # Step 4: Save missing files to manifest
    print("[STEP 4] Creating missing files manifest...")
    if missing_files:
        save_missing_manifest(args.manifest_path, missing_files, args.output)
        
        # Print first few missing files
        print("\nFirst 10 missing files:")
        for filename in missing_files[:10]:
            print(f"  {filename}")
        
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print("No missing files found! All files are already uploaded to GCS.")
        # Create empty file with header
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['id', 'filename', 'md5', 'size', 'state'])
    
    # Step 5: Summary
    print()
    print("=== COMPLETED ===")
    print("Summary:")
    print(f"  - GCS files: {len(gcs_files)}")
    print(f"  - Manifest files: {len(manifest_filenames)}")
    print(f"  - Missing files: {len(missing_files)}")
    print(f"  - Output: {args.output}")
    
    if missing_files:
        print()
        print(f"Missing files manifest created at: {args.output}")
        print("You can use this file to download and process the missing files.")
    else:
        print()
        print("All files are present in GCS! 🎉")

if __name__ == "__main__":
    main()
