"""
Training Module for MAE-VAE Model.

This module handles the training of the MAE-VAE model on NIfTI medical image data.
It includes functionality for data loading, model initialization, training loop execution,
and checkpoint management.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mae_vae import MAEVAEViT, MAEVAETrainer
from nifti_dataset import get_nifti_dataloader
from tqdm import tqdm
import math
import os
import argparse

def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train MAE-VAE model on NIfTI data')
    parser.add_argument('--train_dir', type=str, required=True,
                      help='Directory containing training NIfTI files')
    parser.add_argument('--val_dir', type=str, required=True,
                      help='Directory containing validation NIfTI files')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Input image size')
    parser.add_argument('--slice_axis', type=int, default=2,
                      help='Axis along which to extract slices (0: sagittal, 1: coronal, 2: axial)')
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create data loaders
    train_loader = get_nifti_dataloader(
        nifti_dir=args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        slice_axis=args.slice_axis,
        transform=transform,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = get_nifti_dataloader(
        nifti_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        slice_axis=args.slice_axis,
        transform=transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        shuffle=False,
        num_workers=4
    )

    # Model initialization
    model = MAEVAEViT(
        image_size=args.image_size,
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=8,
        decoder_dim=384,
        decoder_depth=4,
        mask_ratio=0.75,
        latent_dim=256
    ).to(device)

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )

    # Learning rate scheduler
    warmup_epochs = 5
    def get_lr_factor(epoch):
        """Calculate learning rate factor.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Learning rate factor.
        """
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.num_epochs - warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_factor)

    # Initialize trainer
    trainer = MAEVAETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        kld_weight=0.05
    )

    # Create directory for saving models and logs
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        epoch_train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Train]')
        for batch in pbar:
            metrics = trainer.train_step(batch)
            epoch_train_losses.append(metrics['loss'])
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'recon': f"{metrics['recon_loss']:.4f}",
                'kld': f"{metrics['kld_loss']:.4f}"
            })

        # Validation phase
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} [Val]'):
                metrics = trainer.validate_step(batch)
                epoch_val_losses.append(metrics['loss'])

        # Calculate average losses
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}/{args.num_epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # Save checkpoint
        checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, 'best_model.pth')
            print(f'Saved new best model with validation loss: {best_val_loss:.4f}')

if __name__ == '__main__':
    main()
