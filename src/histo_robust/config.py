from __future__ import annotations

from typing import Dict, List

# Nine histological classes in colorectal cancer tissue classification
CLASSES: List[str] = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)

# Standard ImageNet normalization parameters
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# 13-Cell Ablation Study Matrix
EXPERIMENTS: List[Dict[str, str]] = [
  # Stage 0: Anchor Baseline
  {"id": "EXP-01", "stage": "Stage 0", "backbone": "resnet50",      "norm": "none",     "aug": "none",        "notes": "Raw baseline"},
  # Stage 1: Color Normalization
  {"id": "EXP-02", "stage": "Stage 1", "backbone": "resnet50",      "norm": "reinhard", "aug": "none",        "notes": "Statistical LAB transfer"},
  {"id": "EXP-03", "stage": "Stage 1", "backbone": "resnet50",      "norm": "macenko",  "aug": "none",        "notes": "Optical density deconvolution"},
  # Stage 2: Augmentation Policy
  {"id": "EXP-04", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_geo",     "notes": "Spatial invariance (flips/rot)"},
  {"id": "EXP-05", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_stain",   "notes": "HED stain jitter"},
  {"id": "EXP-06", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_combined","notes": "Spatial + HED jitter"},
  # Stage 3: Normalization x Augmentation Interaction
  {"id": "EXP-07", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_geo",     "notes": "Macenko + spatial"},
  {"id": "EXP-08", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_stain",   "notes": "Macenko + stain jitter"},
  {"id": "EXP-09", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_combined","notes": "Full defense ResNet"},
  # Stage 4: Pretrained Vision Backbones & Foundation Models
  {"id": "EXP-10", "stage": "Stage 4", "backbone": "convnext_tiny", "norm": "none",     "aug": "none",        "notes": "ConvNeXt-Tiny raw"},
  {"id": "EXP-11", "stage": "Stage 4", "backbone": "convnext_tiny", "norm": "macenko",  "aug": "aug_combined","notes": "ConvNeXt-Tiny defended"},
  {"id": "EXP-12", "stage": "Stage 4", "backbone": "phikon",        "norm": "none",     "aug": "none",        "notes": "Phikon (TCGA SSL ViT) raw"},
  {"id": "EXP-13", "stage": "Stage 4", "backbone": "phikon",        "norm": "macenko",  "aug": "aug_combined","notes": "Phikon defended"},
]
