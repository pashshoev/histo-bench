import os

import h5py
import pandas as pd
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset


def filter_data(metadata_df: pd.DataFrame, feature_dir: str) -> pd.DataFrame:
    data_info = []
    for index, row in metadata_df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        h5_path = os.path.join(feature_dir, f"{slide_id}.h5")
        if os.path.exists(h5_path):
            data_info.append({'slide_id': slide_id, 'feature_path': h5_path, 'label': label, 'case_id': row['case_id']})
        else:
            pass
            logger.warning(f"Feature file not found for slide_id: {slide_id} at {h5_path}. Skipping.")
    
    filtered_df = pd.DataFrame(data_info)
    return filtered_df


class MILDataset(Dataset):
    def __init__(self, feature_dir: str, metadata_df: pd.DataFrame, patch_sampling_ratio: float = 1.0):
        """
        Args:
            feature_dir (str): Path to the directory containing SLIDE_ID.h5 files.
            metadata_df (pd.DataFrame): DataFrame with 'slide_id' and 'label' columns.
                                        Make sure 'slide_id' matches the H5 file names (without .h5 extension).
            patch_sampling_ratio (float): Fraction of patches to randomly sample from each slide (default: 1.0, use all patches).
        """
        self.feature_dir = feature_dir
        self.metadata_df = metadata_df
        self.patch_sampling_ratio = patch_sampling_ratio
        
        # Create a mapping from index to slide_id and label for efficient lookup
        self.data_info = []
        for index, row in self.metadata_df.iterrows():
            slide_id = row['slide_id']
            label = row['label']
            h5_path = os.path.join(feature_dir, f"{slide_id}.h5")
            if os.path.exists(h5_path):
                self.data_info.append({'slide_id': slide_id, 'feature_path': h5_path, 'label': label})
            else:
                pass
                logger.warning(f"Feature file not found for slide_id: {slide_id} at {h5_path}. Skipping.")
        df = pd.DataFrame(self.data_info)
        logger.info(f"Labels distribution:\n{df['label'].value_counts().sort_index()}")

    def __len__(self):
        # Returns the total number of bags (slides) in the dataset
        return len(self.data_info)

    def __getitem__(self, idx):
            """
            Retrieves a single bag (slide) and its label.

            Args:
                idx (int): Index of the bag to retrieve.

            Returns:
                tuple: A tuple containing:
                    - features (torch.Tensor): Tensor of features for all patches in the bag.
                                            Shape: (num_patches, feature_dim)
                    - label (torch.Tensor): Tensor representing the bag's label.
                                            Shape: (1,) (or appropriate for your task, e.g., for multi-class)
            """
            info = self.data_info[idx]
            feature_path = info['feature_path']
            label = info['label']
            slide_id = info['slide_id'] # Useful for debugging or logging

            features = None
            try:
                with h5py.File(feature_path, 'r') as f:
                    features = torch.from_numpy(f['features'][()]).float()

                label_tensor = torch.tensor(label, dtype=torch.long) 
                
                if features.dim() == 3 and features.shape[0] == 1:
                    # Drop the batch dimension if it's 1
                    features = features.squeeze(0)
                elif features.dim() == 1:
                    # Ensure features are always 2D (num_patches, feature_dim)
                    features = features.unsqueeze(0)
                if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
                    # Drop the batch dimension if it's 1
                    label_tensor = label_tensor.squeeze(0)
                
                # Random patch sampling if ratio < 1.0
                if self.patch_sampling_ratio < 1.0:
                    num_patches = features.shape[0]
                    num_samples = max(10, int(num_patches * self.patch_sampling_ratio))
                    
                    # Randomly sample patch indices
                    patch_indices = np.random.choice(num_patches, size=num_samples, replace=False)
                    features = features[patch_indices]
                
                logger.info(f"Feature shape: {features.shape}, Label shape: {label_tensor.shape}")
                return features, label_tensor

            except Exception as e:
                logger.error(f"Error loading H5 file {feature_path} for slide {slide_id}: {e}")
                # Depending on your needs, you might:
                # 1. Raise the error to stop execution.
                # 2. Return None and filter in DataLoader's collate_fn (more complex).
                # 3. For now, we'll re-raise, as it indicates a data integrity issue.
                raise

