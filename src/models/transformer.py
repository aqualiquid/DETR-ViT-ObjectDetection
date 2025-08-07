"""
DETR Transformer implementation.
Encoder: Process ViT features with self-attention
Decoder: Use object queries to detect objects with cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing ViT features.
    Applies self-attention to let different image patches communicate.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        # Single encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Stack multiple layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.d_model = d_model
    
    def forward(self, src: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (B, N, d_model) - features from ViT backbone
            pos_embed: (B, N, d_model) - positional embeddings
        Returns:
            (B, N, d_model) - encoded features
        """
        # Add positional encoding if provided
        if pos_embed is not None:
            src = src + pos_embed
        
        # Apply transformer encoder
        output = self.transformer_encoder(src)
        
        return output


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for object detection.
    Uses learnable object queries to attend to encoder features.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        # Single decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        # Stack multiple layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.d_model = d_model
    
    def forward(self, 
                tgt: torch.Tensor,
                memory: torch.Tensor,
                query_pos: Optional[torch.Tensor] = None,
                pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: (B, num_queries, d_model) - object queries
            memory: (B, N, d_model) - encoder output
            query_pos: (B, num_queries, d_model) - query positional embeddings
            pos_embed: (B, N, d_model) - encoder positional embeddings
        Returns:
            (B, num_queries, d_model) - decoded features for each query
        """
        # Add positional encoding to queries
        if query_pos is not None:
            tgt = tgt + query_pos
        
        # Add positional encoding to memory
        if pos_embed is not None:
            memory = memory + pos_embed
        
        # Apply transformer decoder
        output = self.transformer_decoder(tgt, memory)
        
        return output


class PositionalEncoding2D(nn.Module):
    """
    2D positional encoding for spatial features.
    Combines x and y position encodings.
    """
    
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Create learnable 2D position embeddings
        self.row_embed = nn.Embedding(max_len, d_model // 2)
        self.col_embed = nn.Embedding(max_len, d_model // 2)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize embeddings."""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, d_model) - flattened spatial features
            height: spatial height of feature map
            width: spatial width of feature map
        Returns:
            (B, H*W, d_model) - positional embeddings
        """
        batch_size = x.size(0)
        
        # Create position indices
        i = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)
        j = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        
        # Get embeddings
        x_emb = self.col_embed(i)  # (H, W, d_model//2)
        y_emb = self.row_embed(j)  # (H, W, d_model//2)
        
        # Concatenate x and y embeddings
        pos = torch.cat([x_emb, y_emb], dim=-1)  # (H, W, d_model)
        
        # Flatten and expand for batch
        pos = pos.flatten(0, 1).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, H*W, d_model)
        
        return pos


class DETRTransformer(nn.Module):
    """
    Complete DETR transformer combining encoder and decoder.
    """
    
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_queries: int = 100):
        super().__init__()
        
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Positional encoding for spatial features
        self.pos_encoding = PositionalEncoding2D(d_model)
        
        # Projection layer to match ViT output to transformer input
        # ViT outputs different dimensions (384 for small, 768 for base)
        # We need to project to d_model (256)
        self.input_proj = nn.Conv2d(768, d_model, kernel_size=1)  # Will adjust based on ViT
        
    def set_vit_output_dim(self, vit_dim: int):
        """Set the input projection based on ViT output dimension."""
        self.input_proj = nn.Conv2d(vit_dim, self.d_model, kernel_size=1)
    
    def forward(self, vit_features: torch.Tensor, feature_height: int, feature_width: int) -> torch.Tensor:
        """
        Args:
            vit_features: (B, H*W, vit_dim) - features from ViT backbone
            feature_height: spatial height of feature map
            feature_width: spatial width of feature map
        Returns:
            (B, num_queries, d_model) - object embeddings
        """
        batch_size = vit_features.size(0)
        
        # Reshape to spatial format for projection
        # (B, H*W, vit_dim) -> (B, vit_dim, H, W)
        vit_features = vit_features.transpose(1, 2).view(
            batch_size, -1, feature_height, feature_width
        )
        
        # Project to transformer dimension
        # (B, vit_dim, H, W) -> (B, d_model, H, W)
        src = self.input_proj(vit_features)
        
        # Flatten back to sequence format
        # (B, d_model, H, W) -> (B, H*W, d_model)
        src = src.flatten(2).transpose(1, 2)
        
        # Generate positional encoding
        pos_embed = self.pos_encoding(src, feature_height, feature_width)
        
        # Encoder: process image features
        memory = self.encoder(src, pos_embed)
        
        # Get object queries
        # (num_queries, d_model) -> (B, num_queries, d_model)
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Initialize target queries (learnable)
        tgt = torch.zeros_like(query_embed)
        
        # Decoder: generate object embeddings
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            query_pos=query_embed,
            pos_embed=pos_embed
        )
        
        return output


def create_detr_transformer(num_queries: int = 100) -> DETRTransformer:
    """Create DETR transformer with standard configuration."""
    return DETRTransformer(
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_queries=num_queries
    )


def test_detr_transformer():
    """Test the DETR transformer with dummy data."""
    print("Testing DETR Transformer...")
    
    # Create transformer
    transformer = create_detr_transformer(num_queries=100)
    
    # Dummy ViT features (from ViT-Base: 768 dim)
    # For 224x224 image with patch_size=16: 14x14 = 196 patches
    batch_size = 2
    num_patches = 196  # 14 * 14
    vit_dim = 768
    
    vit_features = torch.randn(batch_size, num_patches, vit_dim)
    
    print(f"ViT features shape: {vit_features.shape}")
    
    # Forward pass
    with torch.no_grad():
        # Set input projection for ViT-Base
        transformer.set_vit_output_dim(vit_dim)
        
        object_embeddings = transformer(vit_features, feature_height=14, feature_width=14)
    
    print(f"Object embeddings shape: {object_embeddings.shape}")
    print(f"Expected: ({batch_size}, 100, 256)")
    
    # Check output shape
    assert object_embeddings.shape == (batch_size, 100, 256)
    
    print("✓ DETR Transformer test passed!")


if __name__ == "__main__":
    test_detr_transformer()