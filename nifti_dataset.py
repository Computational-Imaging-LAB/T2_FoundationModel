"""
NIfTI Dataset Handler Module.

This module provides functionality for loading and preprocessing NIfTI medical image files
for use with the MAE-VAE model. It includes support for 3D volume slicing, normalization,
and data augmentation.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage import transform, exposure
from torchvision import transforms

class NIfTIDataset(Dataset):
    """Dataset class for handling NIfTI medical image files.

    Args:
        nifti_dir (str): Directory containing NIfTI files.
        slice_axis (int): Axis along which to extract slices (0: sagittal, 1: coronal, 2: axial).
        transform (callable, optional): Optional transform to be applied on a sample.
        image_size (int): Size to resize images to.
        min_val (float): Minimum value for normalization.
        max_val (float): Maximum value for normalization.

    Attributes:
        nifti_files (list): List of NIfTI file paths.
        slice_mapping (list): Mapping of global indices to (file_idx, slice_idx) pairs.
    """

    def __init__(self, nifti_dir, slice_axis=2, transform=None, image_size=224,
                 min_val=-1000, max_val=1000):
        self.nifti_dir = nifti_dir
        self.slice_axis = slice_axis
        self.transform = transform
        self.image_size = image_size
        self.min_val = min_val
        self.max_val = max_val

        self.nifti_files = []
        self.slice_mapping = []

        # Get all NIfTI files
        for filename in os.listdir(nifti_dir):
            if filename.endswith(('.nii', '.nii.gz')):
                filepath = os.path.join(nifti_dir, filename)
                img = nib.load(filepath)
                n_slices = img.shape[slice_axis]
                
                file_idx = len(self.nifti_files)
                self.nifti_files.append(filepath)
                
                # Create mapping for each slice in this file
                for slice_idx in range(n_slices):
                    self.slice_mapping.append((file_idx, slice_idx))

    def __len__(self):
        """Get dataset length.

        Returns:
            int: Total number of slices across all volumes.
        """
        return len(self.slice_mapping)

    def __getitem__(self, idx):
        """Get a specific slice from the dataset.

        Args:
            idx (int): Global index of the slice.

        Returns:
            torch.Tensor: Preprocessed slice tensor.
        """
        file_idx, slice_idx = self.slice_mapping[idx]
        nifti_path = self.nifti_files[file_idx]
        
        # Load NIfTI file
        img = nib.load(nifti_path)
        volume = img.get_fdata()
        
        # Extract slice
        if self.slice_axis == 0:
            slice_data = volume[slice_idx, :, :]
        elif self.slice_axis == 1:
            slice_data = volume[:, slice_idx, :]
        else:  # self.slice_axis == 2
            slice_data = volume[:, :, slice_idx]
        
        # Preprocess slice
        slice_tensor = self.preprocess_slice(slice_data)
        
        if self.transform is not None:
            slice_tensor = self.transform(slice_tensor)
        
        return slice_tensor

    def preprocess_slice(self, slice_data):
        """Preprocess a single slice.

        Args:
            slice_data (numpy.ndarray): Raw slice data.

        Returns:
            torch.Tensor: Preprocessed slice tensor.
        """
        # Clip values
        slice_data = np.clip(slice_data, self.min_val, self.max_val)
        
        # Normalize to [0, 1]
        slice_data = (slice_data - self.min_val) / (self.max_val - self.min_val)
        
        # Enhance contrast
        p2, p98 = np.percentile(slice_data, (2, 98))
        slice_data = exposure.rescale_intensity(slice_data, in_range=(p2, p98))
        
        # Resize
        if slice_data.shape != (self.image_size, self.image_size):
            slice_data = transform.resize(slice_data, (self.image_size, self.image_size),
                                       mode='constant', anti_aliasing=True)
        
        # Convert to RGB by repeating the channel
        slice_data = np.stack([slice_data] * 3, axis=0)
        
        return torch.FloatTensor(slice_data)

def get_nifti_dataloader(nifti_dir, batch_size=32, image_size=224, slice_axis=2,
                        transform=None, shuffle=True, num_workers=4):
    """Create a DataLoader for NIfTI files.

    Args:
        nifti_dir (str): Directory containing NIfTI files.
        batch_size (int): Batch size.
        image_size (int): Size to resize images to.
        slice_axis (int): Axis along which to extract slices.
        transform (callable, optional): Optional transform to be applied on a sample.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the NIfTI dataset.
    """
    dataset = NIfTIDataset(
        nifti_dir=nifti_dir,
        slice_axis=slice_axis,
        transform=transform,
        image_size=image_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
