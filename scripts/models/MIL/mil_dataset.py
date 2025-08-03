import os

import h5py
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset


def filter_data(metadata_df: pd.DataFrame, feature_dir: str) -> pd.DataFrame:
    data_info = []
    for index, row in metadata_df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        h5_path = os.path.join(feature_dir, f"{slide_id}.h5")
        if os.path.exists(h5_path):
            data_info.append({'slide_id': slide_id, 'feature_path': h5_path, 'label': label})
        else:
            pass
            logger.warning(f"Feature file not found for slide_id: {slide_id} at {h5_path}. Skipping.")
    
    filtered_df = pd.DataFrame(data_info)
    return filtered_df


class MILDataset(Dataset):
    def __init__(self, feature_dir: str, metadata_df: pd.DataFrame):
        """
        Args:
            feature_dir (str): Path to the directory containing SLIDE_ID.h5 files.
            metadata_df (pd.DataFrame): DataFrame with 'slide_id' and 'label' columns.
                                        Make sure 'slide_id' matches the H5 file names (without .h5 extension).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.feature_dir = feature_dir
        self.metadata_df = metadata_df
        
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
        logger.info(f"Labels distribution:\n{df['label'].value_counts()}")

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
                if label_tensor.dim() == 3 and label_tensor.shape[0] == 1:
                    # Drop the batch dimension if it's 1
                    label_tensor = label_tensor.squeeze(0)
                # logger.info(f"Feature shape: {features.shape}, Label shape: {label_tensor.shape}")
                return features, label_tensor

            except Exception as e:
                logger.error(f"Error loading H5 file {feature_path} for slide {slide_id}: {e}")
                # Depending on your needs, you might:
                # 1. Raise the error to stop execution.
                # 2. Return None and filter in DataLoader's collate_fn (more complex).
                # 3. For now, we'll re-raise, as it indicates a data integrity issue.
                raise


def mil_collate_fn(batch):
    """
    Custom collate function for MIL datasets with variable-length bags.
    
    Args:
        batch: List of tuples (features, label) where features have variable shapes
        
    Returns:
        tuple: (padded_features, labels, attention_masks)
            - padded_features: Tensor of shape (batch_size, max_patches, feature_dim)
            - labels: Tensor of shape (batch_size,)
            - attention_masks: Tensor of shape (batch_size, max_patches) with 1s for real patches, 0s for padding
    """
    features_list, labels_list = zip(*batch)
    
    # Get the maximum number of patches in this batch
    max_patches = max(feat.shape[0] for feat in features_list)
    feature_dim = features_list[0].shape[1]
    batch_size = len(features_list)
    
    # Initialize padded tensors
    padded_features = torch.zeros(batch_size, max_patches, feature_dim)
    attention_masks = torch.zeros(batch_size, max_patches, dtype=torch.bool)
    labels = torch.stack(labels_list)
    
    # Fill the tensors
    for i, (features, _) in enumerate(batch):
        num_patches = features.shape[0]
        padded_features[i, :num_patches, :] = features
        attention_masks[i, :num_patches] = True  # True for real patches, False for padding
    
    return padded_features, labels, attention_masks