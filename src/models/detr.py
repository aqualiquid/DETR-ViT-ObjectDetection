"""
Complete DETR model combining ViT backbone + Transformer + Detection Head.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

try:
    from .vit_backbone import ViTBackbone, create_vit_small, create_vit_base
    from .transformer import DETRTransformer, create_detr_transformer
    from .detection_head import DetectionHead
except ImportError:
    from vit_backbone import ViTBackbone, create_vit_small, create_vit_base
    from transformer import DETRTransformer, create_detr_transformer
    from detection_head import DetectionHead


class DETR(nn.Module):
    """
    Complete DETR model for object detection.
    
    Architecture:
    Input Image → ViT Backbone → DETR Transformer → Detection Head → Predictions
    """
    
    def __init__(self,
                 backbone: ViTBackbone,
                 transformer: DETRTransformer,
                 num_classes: int = 80,
                 num_queries: int = 100):
        super().__init__()
        
        self.backbone = backbone
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Detection head
        self.detection_head = DetectionHead(
            d_model=transformer.d_model,
            num_classes=num_classes,
            num_queries=num_queries
        )
        
        # Set transformer input projection based on backbone output dim
        self.transformer.set_vit_output_dim(backbone.embed_dim)
        
        # Get feature map dimensions from backbone
        self.feature_height, self.feature_width = backbone.get_feature_map_size()
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete DETR model.
        
        Args:
            images: (B, C, H, W) - batch of input images
            
        Returns:
            Dict with:
                - 'pred_logits': (B, num_queries, num_classes + 1)
                - 'pred_boxes': (B, num_queries, 4)
        """
        # Step 1: Extract features using ViT backbone
        vit_features = self.backbone(images)  # (B, num_patches, embed_dim)
        
        # Step 2: Process features through DETR transformer
        object_embeddings = self.transformer(
            vit_features, 
            self.feature_height, 
            self.feature_width
        )  # (B, num_queries, d_model)
        
        # Step 3: Generate predictions using detection head
        predictions = self.detection_head(object_embeddings)
        
        return predictions
    
    def get_model_info(self) -> Dict[str, int]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'backbone_dim': self.backbone.embed_dim,
            'transformer_dim': self.transformer.d_model,
            'num_classes': self.num_classes,
            'num_queries': self.num_queries,
            'image_size': self.backbone.img_size,
            'patch_size': self.backbone.patch_size
        }


def create_detr_small(num_classes: int = 80, num_queries: int = 100, img_size: int = 224) -> DETR:
    """Create DETR with ViT-Small backbone."""
    backbone = create_vit_small(img_size=img_size)
    transformer = create_detr_transformer(num_queries=num_queries)
    
    return DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=num_queries
    )


def create_detr_base(num_classes: int = 80, num_queries: int = 100, img_size: int = 224) -> DETR:
    """Create DETR with ViT-Base backbone."""
    backbone = create_vit_base(img_size=img_size)
    transformer = create_detr_transformer(num_queries=num_queries)
    
    return DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_queries=num_queries
    )


def test_complete_detr():
    """Test the complete DETR model."""
    print("Testing Complete DETR Model...")
    print("=" * 50)
    
    # Create DETR model
    model = create_detr_small(num_classes=80, num_queries=100, img_size=224)
    
    # Print model info
    info = model.get_model_info()
    print("Model Configuration:")
    for key, value in info.items():
        if 'parameters' in key:
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTesting forward pass...")
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    print(f"Input shape: {images.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    
    # Check outputs
    pred_logits = predictions['pred_logits']
    pred_boxes = predictions['pred_boxes']
    
    print(f"\nOutput shapes:")
    print(f"  Class logits: {pred_logits.shape}")
    print(f"  Bounding boxes: {pred_boxes.shape}")
    
    # Verify shapes
    expected_logits_shape = (batch_size, 100, 81)  # 80 classes + 1 no-object
    expected_boxes_shape = (batch_size, 100, 4)    # [cx, cy, w, h]
    
    assert pred_logits.shape == expected_logits_shape, f"Expected {expected_logits_shape}, got {pred_logits.shape}"
    assert pred_boxes.shape == expected_boxes_shape, f"Expected {expected_boxes_shape}, got {pred_boxes.shape}"
    
    # Check box coordinates are normalized
    assert torch.all(pred_boxes >= 0) and torch.all(pred_boxes <= 1), "Box coordinates should be in [0, 1]"
    
    print("\n✓ Complete DETR model test passed!")
    print(f"✓ Model has {info['total_parameters']:,} parameters")
    
    return model


if __name__ == "__main__":
    model = test_complete_detr()