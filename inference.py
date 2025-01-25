"""
Inference Module for MAE-VAE Model.

This module provides functionality for running inference on NIfTI medical images using
the trained MAE-VAE model. It supports reconstruction of individual slices and entire
volumes, with visualization capabilities.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from mae_vae import MAEVAEViT
import numpy as np
from einops import rearrange
import nibabel as nib
from skimage.transform import resize
from skimage import exposure
import os

class MAEVAEInference:
    """Inference class for MAE-VAE model.

    Args:
        checkpoint_path (str): Path to model checkpoint.
        device (torch.device, optional): Device to run inference on.

    Attributes:
        model (MAEVAEViT): The loaded model.
        device (torch.device): Device being used.
        normalize (transforms.Normalize): Normalization transform.
        inverse_normalize (transforms.Normalize): Inverse normalization transform.
    """

    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model = MAEVAEViT(
            image_size=224,
            patch_size=16,
            embed_dim=512,
            depth=6,
            num_heads=8,
            decoder_dim=384,
            decoder_depth=4,
            mask_ratio=0.75,
            latent_dim=256
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.inverse_normalize = transforms.Normalize(
            mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
            std=[1/0.5, 1/0.5, 1/0.5]
        )

    def preprocess_slice(self, slice_data, image_size=224):
        """Preprocess a single slice from a NIfTI file.

        Args:
            slice_data (numpy.ndarray): Input slice data.
            image_size (int, optional): Target image size.

        Returns:
            torch.Tensor: Preprocessed slice tensor.
        """
        slice_data = resize(slice_data, (image_size, image_size),
                          mode='constant', anti_aliasing=True)
        
        slice_data = exposure.rescale_intensity(slice_data, out_range=(0, 1))
        
        p2, p98 = np.percentile(slice_data, (2, 98))
        slice_data = exposure.rescale_intensity(slice_data, in_range=(p2, p98))
        
        slice_data = np.stack([slice_data] * 3, axis=0)
        slice_tensor = torch.FloatTensor(slice_data)
        slice_tensor = self.normalize(slice_tensor)
        
        return slice_tensor

    def get_slice(self, nifti_data, slice_idx, axis=2):
        """Extract a 2D slice from 3D volume.

        Args:
            nifti_data (numpy.ndarray): Input 3D volume.
            slice_idx (int): Index of slice to extract.
            axis (int, optional): Axis along which to extract slice.

        Returns:
            numpy.ndarray: Extracted 2D slice.
        """
        if axis == 0:
            slice_data = nifti_data[slice_idx, :, :]
        elif axis == 1:
            slice_data = nifti_data[:, slice_idx, :]
        else:  # axis == 2
            slice_data = nifti_data[:, :, slice_idx]
        
        return slice_data

    def reconstruct_slice(self, nifti_path, slice_idx, axis=2, with_mask=True):
        """Reconstruct a specific slice from a NIfTI file.

        Args:
            nifti_path (str): Path to NIfTI file.
            slice_idx (int): Index of slice to reconstruct.
            axis (int, optional): Axis along which to extract slice.
            with_mask (bool, optional): Whether to use masking.

        Returns:
            tuple: Original slice, reconstructed slice, and mask (if with_mask=True).
        """
        img = nib.load(nifti_path)
        nifti_data = img.get_fdata()
        
        slice_data = self.get_slice(nifti_data, slice_idx, axis)
        slice_tensor = self.preprocess_slice(slice_data)
        slice_tensor = slice_tensor.unsqueeze(0)
        
        with torch.no_grad():
            if with_mask:
                pred, mask, mu, logvar = self.model(slice_tensor.to(self.device))
            else:
                pred, mask, mu, logvar = self.model(slice_tensor.to(self.device), mask_ratio=0.0)
            
            pred = pred[:, 1:, :]
            patch_size = self.model.patch_size
            h = w = int(np.sqrt(pred.shape[1]))
            c = 3
            
            reconstructed = rearrange(
                pred,
                'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=h, w=w, p1=patch_size, p2=patch_size, c=c
            )
            
            reconstructed = self.inverse_normalize(reconstructed)
            original = self.inverse_normalize(slice_tensor)
            
            return original.cpu(), reconstructed.cpu(), mask.cpu() if with_mask else None

    def reconstruct_volume(self, nifti_path, axis=2, with_mask=True, save_dir=None):
        """Reconstruct all slices in a volume.

        Args:
            nifti_path (str): Path to NIfTI file.
            axis (int, optional): Axis along which to extract slices.
            with_mask (bool, optional): Whether to use masking.
            save_dir (str, optional): Directory to save visualizations.

        Yields:
            tuple: Original slice, reconstructed slice, and mask (if with_mask=True).
        """
        img = nib.load(nifti_path)
        nifti_data = img.get_fdata()
        n_slices = nifti_data.shape[axis]
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for slice_idx in range(n_slices):
            original, recon, mask = self.reconstruct_slice(
                nifti_path, slice_idx, axis, with_mask
            )
            
            if save_dir:
                fig, axes = plt.subplots(1, 3 if with_mask else 2, figsize=(15, 5))
                
                axes[0].imshow(original.squeeze().permute(1, 2, 0).clamp(0, 1), cmap='gray')
                axes[0].set_title(f'Original - Slice {slice_idx}')
                axes[0].axis('off')
                
                axes[1].imshow(recon.squeeze().permute(1, 2, 0).clamp(0, 1), cmap='gray')
                axes[1].set_title(f'Reconstruction - Slice {slice_idx}')
                axes[1].axis('off')
                
                if with_mask:
                    axes[2].imshow(mask.squeeze(), cmap='gray')
                    axes[2].set_title('Mask')
                    axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'slice_{slice_idx:04d}.png'))
                plt.close()
            
            yield original, recon, mask

def main():
    """Main function for running inference."""
    inference = MAEVAEInference('best_model.pth')
    os.makedirs('outputs', exist_ok=True)
    
    nifti_path = '/path/to/your/nifti/file.nii.gz'
    slice_idx = 50
    
    original, recon, mask = inference.reconstruct_slice(
        nifti_path,
        slice_idx=slice_idx,
        axis=2
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original.squeeze().permute(1, 2, 0).clamp(0, 1), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon.squeeze().permute(1, 2, 0).clamp(0, 1), cmap='gray')
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    axes[2].imshow(mask.squeeze(), cmap='gray')
    axes[2].set_title('Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/single_slice_reconstruction.png')
    plt.close()
    
    volume_output_dir = os.path.join('outputs', 'volume_reconstruction')
    for original, recon, mask in inference.reconstruct_volume(
        nifti_path,
        axis=2,
        save_dir=volume_output_dir
    ):
        pass

if __name__ == '__main__':
    main()
