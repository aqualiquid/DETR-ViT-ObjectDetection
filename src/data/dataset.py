"""
COCO dataset loader for DETR training.
"""

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
import os


class COCODataset(Dataset):
    """COCO dataset for object detection."""
    
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
    
    def __getitem__(self, idx: int):
        """Get a single sample."""
        # TODO: Implement COCO data loading
        # Load image, annotations, apply transforms
        pass
    
    def __len__(self) -> int:
        return len(self.ids)


# TODO: Implement data transforms and collate function
