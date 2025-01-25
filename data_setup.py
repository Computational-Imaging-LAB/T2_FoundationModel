"""
Data Setup Module for MAE-VAE Model.

This module handles the creation of symbolic links and preprocessing of 3D NIfTI data,
specifically focusing on files containing 'org' in their names.
"""

import os
import glob
import shutil
import nibabel as nib
import numpy as np
from tqdm import tqdm

def create_symlinks(source_dir, target_dir, pattern='*org*.nii.gz'):
    """Create symbolic links for NIfTI files matching the pattern.

    Args:
        source_dir (str): Source directory containing the original data
        target_dir (str): Target directory for symbolic links
        pattern (str): Pattern to match files (default: '*org*.nii.gz')

    Returns:
        list: List of created symbolic link paths
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all matching files in source directory
    source_files = glob.glob(os.path.join(source_dir, pattern))
    symlinks = []
    
    print(f"Creating symbolic links for {len(source_files)} files...")
    for source_file in tqdm(source_files):
        # Get the base filename
        basename = os.path.basename(source_file)
        # Create the target path
        target_path = os.path.join(target_dir, basename)
        
        # Remove existing symlink if it exists
        if os.path.exists(target_path):
            os.remove(target_path)
        
        # Create symbolic link
        os.symlink(source_file, target_path)
        symlinks.append(target_path)
    
    return symlinks

def extract_slices(nifti_path, output_dir, axis=2):
    """Extract 2D slices from a 3D NIfTI file.

    Args:
        nifti_path (str): Path to the NIfTI file
        output_dir (str): Directory to save the extracted slices
        axis (int): Axis along which to extract slices (0: sagittal, 1: coronal, 2: axial)

    Returns:
        int: Number of slices extracted
    """
    # Load NIfTI file
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]
    
    # Extract slices
    n_slices = data.shape[axis]
    for i in range(n_slices):
        if axis == 0:
            slice_data = data[i, :, :]
        elif axis == 1:
            slice_data = data[:, i, :]
        else:  # axis == 2
            slice_data = data[:, :, i]
        
        # Save slice as .npy file
        output_path = os.path.join(output_dir, f"{base_name}_slice_{i:04d}.npy")
        np.save(output_path, slice_data)
    
    return n_slices

def process_dataset(source_dir, target_base_dir, axis=2):
    """Process the entire dataset by creating symlinks and extracting slices.

    Args:
        source_dir (str): Source directory containing the original data
        target_base_dir (str): Base directory for processed data
        axis (int): Axis along which to extract slices (0: sagittal, 1: coronal, 2: axial)

    Returns:
        tuple: Number of processed files and total number of slices
    """
    # Create directories
    symlink_dir = os.path.join(target_base_dir, 'symlinks')
    slices_dir = os.path.join(target_base_dir, 'slices')
    
    # Create symlinks
    symlinks = create_symlinks(source_dir, symlink_dir)
    
    # Process each file
    total_slices = 0
    print("\nExtracting slices from NIfTI files...")
    for symlink in tqdm(symlinks):
        # Create subject-specific directory for slices
        subject_name = os.path.splitext(os.path.splitext(os.path.basename(symlink))[0])[0]
        subject_slice_dir = os.path.join(slices_dir, subject_name)
        
        # Extract slices
        n_slices = extract_slices(symlink, subject_slice_dir, axis)
        total_slices += n_slices
    
    return len(symlinks), total_slices

def main():
    """Main function to set up the dataset."""
    # Define directories
    source_dir = "/path/to/your/data"  # Replace with your data directory
    target_base_dir = "processed_data"
    
    # Process the dataset
    n_files, n_slices = process_dataset(source_dir, target_base_dir)
    
    print(f"\nProcessing complete!")
    print(f"Processed {n_files} files")
    print(f"Extracted {n_slices} total slices")
    print(f"\nProcessed data is stored in: {os.path.abspath(target_base_dir)}")

if __name__ == '__main__':
    main()
