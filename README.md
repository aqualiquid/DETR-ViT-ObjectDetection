# DETR with Vision Transformer for Object Detection

Implementation of DETR (Detection Transformer) with Vision Transformer backbone from scratch.

## Project Overview

This project implements **DETR (Detection Transformer)** with a **Vision Transformer (ViT)** backbone for object detection, built completely from scratch. Unlike traditional CNN-based detectors, DETR treats object detection as a direct set prediction problem using transformers.

### Key Features
- From Scratch Implementation: Complete DETR architecture built from ground up
- Vision Transformer Backbone: ViT for feature extraction instead of traditional CNN
- End-to-End Training: No NMS or anchor generation required
- COCO Dataset Support: Training and evaluation on MS COCO
- Optimized Inference: Performance optimizations for real-world deployment

## Architecture

The model processes images through a pipeline of ViT backbone, transformer encoder-decoder, and detection head for end-to-end object detection without requiring NMS or anchor generation.

### Model Components
1. **Vision Transformer (ViT)**: Patch-based image encoder
2. **Transformer Encoder**: Self-attention for global context
3. **Transformer Decoder**: Cross-attention with learnable object queries  
4. **Detection Head**: Classification and bounding box regression
5. **Bipartite Matching**: Hungarian algorithm for loss computation

## Results

| Model | Backbone | mAP | mAP@50 | FPS | Parameters |
|-------|----------|-----|--------|-----|------------|
| DETR-ViT-Small | ViT-S/16 | Coming Soon | - | - | ~40M |
| DETR-ViT-Base  | ViT-B/16 | Coming Soon | - | - | ~90M |

*Results will be updated as training progresses*

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/DETR-ViT-ObjectDetection.git
cd DETR-ViT-ObjectDetection

# Create conda environment
conda create -n detr-vit python=3.9 -y
conda activate detr-vit

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation
```bash
# Download COCO dataset
python src/data/download_coco.py --data_dir data/raw

# Prepare data
python src/data/prepare_data.py
```

### Training
```bash
# Train with default config
python train.py --config configs/detr_vit_small.yaml

# Monitor training
tensorboard --logdir results/logs
```

### Inference
```bash
# Run inference on single image
python infer.py --image path/to/image.jpg --model checkpoints/best_model.pth

# Batch inference
python infer.py --input_dir path/to/images --output_dir results/predictions
```

## Project Structure

```
DETR-ViT-ObjectDetection/
├── README.md
├── requirements.txt
├── train.py                    # Training script
├── infer.py                    # Inference script
├── configs/
│   ├── detr_vit_small.yaml
│   └── detr_vit_base.yaml
├── src/
│   ├── models/
│   │   ├── vit_backbone.py     # Vision Transformer implementation
│   │   ├── transformer.py     # Encoder/Decoder blocks
│   │   ├── detr.py            # Main DETR model
│   │   └── losses.py          # Bipartite matching loss
│   ├── data/
│   │   ├── dataset.py         # COCO dataset loader
│   │   ├── transforms.py      # Data augmentation
│   │   └── download_coco.py   # Dataset download utility
│   └── utils/
│       ├── training.py        # Training utilities
│       ├── evaluation.py     # mAP calculation
│       └── visualization.py  # Result visualization
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   └── 03_results_analysis.ipynb
└── results/
    ├── checkpoints/
    ├── logs/
    └── visualizations/
```

## Implementation Details

### Vision Transformer Backbone
- Patch size: 16x16
- Position embeddings: Learnable 2D embeddings
- Multi-head self-attention with 12/16 layers

### Transformer Architecture
- **Encoder**: 6 layers, 8 attention heads, 256 hidden dim
- **Decoder**: 6 layers with cross-attention to encoder features
- **Object Queries**: 100 learnable embeddings

### Training Strategy
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with cosine scheduling
- **Augmentation**: RandomResizedCrop, ColorJitter, Normalize
- **Loss**: Classification + L1 + GIoU with bipartite matching

## Training Progress

Training logs and metrics will be tracked using:
- **TensorBoard**: Real-time loss/mAP monitoring
- **Wandb**: Experiment tracking and comparison

## Key Insights & Learnings

1. **Attention Patterns**: Visualizing what the model focuses on
2. **Object Queries**: How learnable queries specialize for different objects
3. **Training Dynamics**: Convergence patterns compared to CNN detectors
4. **Performance Trade-offs**: Speed vs accuracy analysis

## TODO

- [x] Project setup and repository structure
- [ ] Vision Transformer backbone implementation
- [ ] Transformer encoder/decoder blocks
- [ ] Bipartite matching loss function
- [ ] COCO dataset integration
- [ ] Training pipeline
- [ ] Evaluation metrics
- [ ] Inference optimization
- [ ] Results analysis and visualization

## Contributing

This is a portfolio project, but feedback and suggestions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

Made for learning modern computer vision