"""
Vision Transformer (ViT) backbone implementation for DETR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings.
    Think of this like CNN but instead of sliding window, we take fixed patches.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # This is like a Conv2d but with stride=kernel_size (no overlap)
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - batch of images
        Returns:
            (B, num_patches, embed_dim) - flattened patch embeddings
        """
        B, C, H, W = x.shape
        
        # Check input size
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}, {W}) doesn't match expected ({self.img_size}, {self.img_size})"
        
        # Convert to patches: (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to (B, num_patches, embed_dim) - standard transformer format
        x = x.transpose(1, 2)
        
        return x


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for image patches.
    Since images are 2D, we need to encode both x and y positions.
    """
    
    def __init__(self, embed_dim: int, height: int, width: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        
        # Create learnable positional embeddings
        # Each patch gets a unique position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width, embed_dim))
        
        # Initialize with small random values
        self._init_weights()
    
    def _init_weights(self):
        """Initialize positional embeddings with truncated normal."""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to patch embeddings."""
        return x + self.pos_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    This is the core of the transformer - allows patches to "talk" to each other.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim) where N is number of patches
        Returns:
            (B, N, embed_dim) - attended features
        """
        B, N, C = x.shape
        
        # Generate Q, K, V: (B, N, 3*embed_dim) -> 3 x (B, N, embed_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Feed-forward network in transformer block.
    Simple 2-layer MLP with GELU activation.
    """
    
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer encoder block.
    Standard architecture: LayerNorm -> Attention -> LayerNorm -> MLP
    With residual connections.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm design (like in original ViT)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for feature extraction.
    This will be used as the CNN replacement in DETR.
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional encoding
        patch_grid = img_size // patch_size
        self.pos_encoding = PositionalEncoding2D(embed_dim, patch_grid, patch_grid)
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT backbone.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Feature embeddings (B, num_patches, embed_dim)
        """
        # Convert to patches
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def get_feature_map_size(self) -> Tuple[int, int]:
        """Get the spatial dimensions of the feature map."""
        patches_per_dim = self.img_size // self.patch_size
        return patches_per_dim, patches_per_dim


def create_vit_small(img_size: int = 224) -> ViTBackbone:
    """Create ViT-Small configuration."""
    return ViTBackbone(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    )


def create_vit_base(img_size: int = 224) -> ViTBackbone:
    """Create ViT-Base configuration."""
    return ViTBackbone(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    )


# Test function to verify implementation
def test_vit_backbone():
    """Test the ViT backbone with dummy data."""
    print("Testing ViT Backbone...")
    
    # Create model
    model = create_vit_small(img_size=224)
    
    # Test with dummy input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        features = model(x)
    
    print(f"Output shape: {features.shape}")
    print(f"Expected: (2, 196, 384)")  # 14x14 patches, 384 dim
    
    # Check if shapes match
    expected_patches = (224 // 16) ** 2  # 14*14 = 196
    assert features.shape == (batch_size, expected_patches, 384)
    
    print("✓ ViT Backbone test passed!")


if __name__ == "__main__":
    test_vit_backbone()