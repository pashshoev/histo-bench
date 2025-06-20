import pandas as pd
import os
import h5py
import numpy as np
import openslide
from torch.utils.data import Dataset

class PatchDatasetFromWSI(Dataset):
    def __init__(self,
                 coordinates_file,
                 wsi_path,
                 transform,
        ):
        """
        Returns patch and respective coordinates for the given wsi
        Args:
            coordinates_file (string): Path to the .h5 file containing patched data.
            wsi_path (string): Path to the WSI file.
            transform (callable): Optional transform to be applied on a sample.
        """
        self.coordinates_path = coordinates_file
        self.wsi_path = wsi_path
        self.wsi_obj = None # Will be opened later in __getitem__
        self.transform = transform

        with h5py.File(self.coordinates_path, 'r') as hdf5_file:
            self.coordinates = np.array(hdf5_file['coords'])
            self.length = len(self.coordinates)
            self.patch_level = hdf5_file['coords'].attrs['patch_level']
            self.patch_size = hdf5_file['coords'].attrs['patch_size']

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        print(f"\t----- DATASET INFO -----\t")
        print(f"Patch level: {self.patch_level}")
        print(f"Total patches: {self.length}")
        print(f"Patch size: {self.patch_size} x {self.patch_size}")

    def __getitem__(self, idx):
        # Open the WSI object if it hasn't been opened yet in this process
        if self.wsi_obj is None:
            self.wsi_obj = openslide.OpenSlide(self.wsi_path)

        coordinates = self.coordinates[idx]
        patch = self.wsi_obj.read_region(coordinates, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        patch = self.transform(patch)

        return {'patch': patch, 'coordinates': coordinates}


class WSIDataset(Dataset):
    def __init__(self, csv_path):
        """Stores available slide ids"""
        self.df = pd.read_csv(csv_path)[["slide_id"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]





