"""
MAE-VAE Vision Transformer Implementation.

This module implements a Masked Autoencoder Variational Autoencoder using Vision Transformer
architecture. It combines the benefits of masked prediction tasks with variational inference
for robust representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np

class PatchEmbedding(nn.Module):
    """2D Image to Patch Embedding.

    Args:
        image_size (int): Input image size.
        patch_size (int): Patch size for tokenization.
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.

    Attributes:
        num_patches (int): Total number of patches.
        proj (nn.Conv2d): Projection layer for patch embedding.
    """

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Patch embeddings of shape (B, N, D).
        """
        b = x.shape[0]
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head Self Attention mechanism.

    Args:
        embed_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by number of heads'
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, D).

        Returns:
            torch.Tensor: Attended tensor of shape (B, N, D).
        """
        B, N, C = x.shape
        assert C == self.embed_dim, f'Input embedding dim {C} does not match layer embedding dim {self.embed_dim}'
        
        # (B, N, C) -> (B, N, 3C) -> (B, N, 3, H, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        
        # (B, N, 3, H, D) -> (3, B, H, N, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Split into q, k, v
        q, k, v = qkv.unbind(0)  # Each has shape (B, H, N, D)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        
        # Combine heads
        x = (attn @ v)  # (B, H, N, D)
        x = x.transpose(1, 2)  # (B, N, H, D)
        x = x.reshape(B, N, C)  # (B, N, C)
        
        # Output projection
        x = self.proj(x)
        return x

class TransformerEncoder(nn.Module):
    """Transformer Block.

    Args:
        embed_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden dimension ratio.
        dropout (float): Dropout rate.
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MAEVAEViT(nn.Module):
    """Masked Autoencoder Variational Autoencoder with Vision Transformer.

    Args:
        image_size (int): Input image size.
        patch_size (int): Patch size for tokenization.
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        decoder_dim (int): Decoder embedding dimension.
        decoder_depth (int): Number of decoder transformer blocks.
        mask_ratio (float): Ratio of patches to mask.
        latent_dim (int): Dimension of latent space.
    """

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        mask_ratio=0.75,
        latent_dim=256
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # VAE components
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_var = nn.Linear(embed_dim, latent_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(latent_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_dim))
        
        self.decoder_layers = nn.ModuleList([
            TransformerEncoder(decoder_dim, num_heads)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_pred = nn.Linear(decoder_dim, patch_size * patch_size * in_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize patch embeddings and position embeddings
        pos_embed = self.patch_embed.pos_embedding
        decoder_pos_embed = self.decoder_pos_embed
        
        torch.nn.init.normal_(pos_embed, std=.02)
        torch.nn.init.normal_(decoder_pos_embed, std=.02)
        
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Apply layer-wise scaling to initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize layer weights.

        Args:
            m (nn.Module): Layer to initialize.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """Perform random masking by per-sample shuffling.

        Args:
            x (torch.Tensor): Input tensor.
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            tuple: Masked tensor, mask, and ids_restore.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep tokens
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def reparameterize(self, mu, logvar):
        """Reparameterization trick.

        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder.

        Args:
            x (torch.Tensor): Input tensor.
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            tuple: Latent representation and mask.
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply Transformer encoder
        for layer in self.encoder_layers:
            x_masked = layer(x_masked)
            
        # Get latent representations
        x_masked_mean = x_masked.mean(dim=1)  # Global average pooling
        mu = self.fc_mu(x_masked_mean)
        logvar = self.fc_var(x_masked_mean)
        z = self.reparameterize(mu, logvar)
        
        return z, mask, ids_restore, mu, logvar

    def forward_decoder(self, z, ids_restore):
        """Forward pass through decoder.

        Args:
            z (torch.Tensor): Latent vector.
            ids_restore (torch.Tensor): Indices for restoring masked patches.

        Returns:
            torch.Tensor: Reconstructed patches.
        """
        # Embed latent vector
        x = self.decoder_embed(z)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - 1, 1)
        x = torch.cat([x, mask_tokens], dim=1)  # [B, N+1, D]
        
        # Restore position information
        x = x + self.decoder_pos_embed
        
        # Apply Transformer decoder
        for layer in self.decoder_layers:
            x = layer(x)
            
        # Predictor projection
        x = self.decoder_pred(x)
        
        return x

    def forward(self, x, mask_ratio=None):
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            mask_ratio (float, optional): Ratio of patches to mask.

        Returns:
            tuple: Predictions, mask, mean, and log variance.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        # Encode
        z, mask, ids_restore, mu, logvar = self.forward_encoder(x, mask_ratio)
        
        # Decode
        pred = self.forward_decoder(z, ids_restore)
        
        return pred, mask, mu, logvar

    def compute_loss(self, x, pred, mask, mu, logvar, kld_weight=0.01):
        """Compute VAE loss.

        Args:
            x (torch.Tensor): Input tensor.
            pred (torch.Tensor): Predicted tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.
            kld_weight (float): Weight for KL divergence term.

        Returns:
            tuple: Total loss, reconstruction loss, and KL divergence loss.
        """
        # Convert image to patches
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                          p1=self.patch_size, p2=self.patch_size)
        
        # Handle masking without CLS token
        mask = mask[:, 1:]  # Remove mask for CLS token
        pred = pred[:, 1:]  # Remove CLS token predictions
        
        # Compute reconstruction loss (MSE)
        loss_recon = F.mse_loss(pred, patches, reduction='none')
        loss_recon = (loss_recon * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)
        
        # KL divergence loss
        loss_kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss_kld = loss_kld.mean()
        
        # Total loss
        loss = loss_recon + kld_weight * loss_kld
        
        return loss, loss_recon, loss_kld

class MAEVAETrainer:
    """MAE-VAE Trainer.

    Args:
        model (nn.Module): MAE-VAE model.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to use.
        kld_weight (float): Weight for KL divergence term.
    """

    def __init__(
        self,
        model,
        optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        kld_weight=0.05
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.kld_weight = kld_weight
        self.scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    def train_step(self, batch):
        """Train step.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            dict: Losses.
        """
        self.model.train()
        x = batch.to(self.device)
        
        # Use mixed precision training
        with torch.cuda.amp.autocast():
            # Forward pass
            pred, mask, mu, logvar = self.model(x)
            
            # Compute loss
            loss, recon_loss, kld_loss = self.model.compute_loss(
                x, pred, mask, mu, logvar, self.kld_weight
            )
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
        }

    def validate_step(self, batch):
        """Validate step.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            dict: Losses.
        """
        self.model.eval()
        with torch.no_grad():
            x = batch.to(self.device)
            
            # Forward pass
            pred, mask, mu, logvar = self.model(x)
            
            # Compute loss
            loss, recon_loss, kld_loss = self.model.compute_loss(
                x, pred, mask, mu, logvar, self.kld_weight
            )
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kld_loss': kld_loss.item()
        }
