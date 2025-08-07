"""
DETR Loss function with bipartite matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian algorithm to find optimal assignment between predictions and ground truth.
    This is the core of DETR - it solves the set prediction problem.
    """
    
    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> List[Tuple]:
        """
        Compute optimal assignment between predictions and targets.
        
        Args:
            outputs: Dict with:
                - 'pred_logits': (B, num_queries, num_classes)
                - 'pred_boxes': (B, num_queries, 4)
            targets: List of B dicts, each with:
                - 'labels': (num_objects,) - class labels
                - 'boxes': (num_objects, 4) - boxes in [cx, cy, w, h] format
                
        Returns:
            List of B tuples (pred_indices, target_indices)
        """
        batch_size, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions for easier processing
        # (B, num_queries, C) -> (B * num_queries, C)
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)
        
        # Concatenate all targets
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])
        
        # Compute classification cost
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute L1 cost between predicted and target boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute GIoU cost
        cost_giou = -self.generalized_box_iou(out_bbox, tgt_bbox)
        
        # Combine costs
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).detach().cpu()
        
        # Apply Hungarian algorithm for each sample in batch
        indices = []
        for i, c in enumerate(C.split([len(v['boxes']) for v in targets], -1)):
            # Convert to numpy for Hungarian algorithm
            indices.append(linear_sum_assignment(c[i].numpy()))
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]
    
    def generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Generalized IoU between two sets of boxes.
        Boxes are in [cx, cy, w, h] format.
        """
        # Convert to [x0, y0, x1, y1] format
        boxes1 = self.box_cxcywh_to_xyxy(boxes1)
        boxes2 = self.box_cxcywh_to_xyxy(boxes2)
        
        return self.box_iou(boxes1, boxes2)[0]
    
    def box_cxcywh_to_xyxy(self, x: torch.Tensor) -> torch.Tensor:
        """Convert boxes from [cx, cy, w, h] to [x0, y0, x1, y1] format."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    
    def box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom
        
        wh = (rb - lt).clamp(min=0)  # intersection
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou, union


class DETRLoss(nn.Module):
    """
    Complete DETR loss function.
    Combines classification loss + bounding box regression loss + GIoU loss.
    """
    
    def __init__(self, num_classes: int, weight_dict: Dict[str, float] = None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher()
        
        if weight_dict is None:
            weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
        self.weight_dict = weight_dict
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute DETR loss.
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dict with individual and total losses
        """
        # Get optimal matching between predictions and targets
        indices = self.matcher(outputs, targets)
        
        # Compute individual losses
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        loss_giou = self.loss_giou(outputs, targets, indices)
        
        # Combine losses
        losses = {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        
        # Weighted sum
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        losses['loss_total'] = total_loss
        
        return losses
    
    def loss_labels(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                   indices: List[Tuple]) -> torch.Tensor:
        """Classification loss."""
        src_logits = outputs['pred_logits']  # (B, num_queries, num_classes)
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes)
    
    def loss_boxes(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                  indices: List[Tuple]) -> torch.Tensor:
        """L1 bounding box loss."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        return loss_bbox.sum() / len(target_boxes) if len(target_boxes) > 0 else torch.tensor(0.0)
    
    def loss_giou(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]], 
                 indices: List[Tuple]) -> torch.Tensor:
        """GIoU bounding box loss."""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_giou = 1 - torch.diag(self.matcher.generalized_box_iou(src_boxes, target_boxes))
        return loss_giou.sum() / len(target_boxes) if len(target_boxes) > 0 else torch.tensor(0.0)
    
    def _get_src_permutation_idx(self, indices: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation indices for source (predictions)."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def create_dummy_targets(batch_size: int = 2, max_objects: int = 5) -> List[Dict[str, torch.Tensor]]:
    """Create dummy targets for testing."""
    targets = []
    for _ in range(batch_size):
        num_objects = torch.randint(1, max_objects + 1, (1,)).item()
        
        # Random boxes in [cx, cy, w, h] format, normalized to [0, 1]
        boxes = torch.rand(num_objects, 4)
        boxes[:, 2:] *= 0.5  # Make width/height smaller
        
        # Random class labels (excluding background class)
        labels = torch.randint(0, 80, (num_objects,))
        
        targets.append({
            'boxes': boxes,
            'labels': labels
        })
    
    return targets


def test_detr_loss():
    """Test DETR loss function."""
    print("Testing DETR Loss Function...")
    
    # Create dummy predictions
    batch_size = 2
    num_queries = 100
    num_classes = 80
    
    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
        'pred_boxes': torch.rand(batch_size, num_queries, 4)  # normalized boxes
    }
    
    # Create dummy targets
    targets = create_dummy_targets(batch_size, max_objects=3)
    
    print(f"Batch size: {batch_size}")
    print(f"Predictions shape: {outputs['pred_logits'].shape}, {outputs['pred_boxes'].shape}")
    print(f"Targets: {[len(t['boxes']) for t in targets]} objects per image")
    
    # Create loss function
    criterion = DETRLoss(num_classes=num_classes)
    
    # Compute loss
    losses = criterion(outputs, targets)
    
    print("\nLoss values:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✓ DETR Loss test passed!")
    
    return criterion


if __name__ == "__main__":
    criterion = test_detr_loss()