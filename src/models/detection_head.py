"""
Detection Head for DETR.
Takes object embeddings and predicts class + bounding box for each query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-layer perceptron for prediction heads."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DetectionHead(nn.Module):
    """
    Detection head that predicts class and bounding box for each object query.
    """
    
    def __init__(self, 
                 d_model: int = 256,
                 num_classes: int = 91,  # COCO: 80 classes + 1 background + 10 extra
                 num_queries: int = 100):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        # Class prediction head
        # Predicts probability for each class (including "no object")
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        
        # Bounding box prediction head  
        # Predicts [cx, cy, w, h] in normalized coordinates [0, 1]
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
    
    def forward(self, object_embeddings: torch.Tensor) -> dict:
        """
        Args:
            object_embeddings: (B, num_queries, d_model) - from transformer decoder
        
        Returns:
            dict with:
                - 'pred_logits': (B, num_queries, num_classes + 1)
                - 'pred_boxes': (B, num_queries, 4)
        """
        # Class predictions
        class_logits = self.class_embed(object_embeddings)
        
        # Box predictions (apply sigmoid to get [0, 1] range)
        box_coords = self.bbox_embed(object_embeddings).sigmoid()
        
        return {
            'pred_logits': class_logits,
            'pred_boxes': box_coords
        }


def test_detection_head():
    """Test detection head with dummy object embeddings."""
    print("Testing Detection Head...")
    
    # Create detection head
    head = DetectionHead(d_model=256, num_classes=80, num_queries=100)
    
    # Dummy object embeddings from transformer
    batch_size = 2
    object_embeddings = torch.randn(batch_size, 100, 256)
    
    print(f"Input embeddings shape: {object_embeddings.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions = head(object_embeddings)
    
    pred_logits = predictions['pred_logits']
    pred_boxes = predictions['pred_boxes']
    
    print(f"Class logits shape: {pred_logits.shape}")
    print(f"Box predictions shape: {pred_boxes.shape}")
    print(f"Expected logits: ({batch_size}, 100, 81)")  # 80 classes + 1 no-object
    print(f"Expected boxes: ({batch_size}, 100, 4)")   # [cx, cy, w, h]
    
    # Check shapes
    assert pred_logits.shape == (batch_size, 100, 81)
    assert pred_boxes.shape == (batch_size, 100, 4)
    
    # Check box coordinates are in [0, 1] range
    assert torch.all(pred_boxes >= 0) and torch.all(pred_boxes <= 1)
    
    print("✓ Detection Head test passed!")


if __name__ == "__main__":
    test_detection_head()