#!/usr/bin/env python
"""Inference & Clinical Demonstration Script for Digital Histopathology Classification.

Fulfills the course exam requirement:
'complete source code for data preparation, training, evaluation, and inference/demo'
and supports Slide 4 (Conclusion / Demo).

Usage examples:
  python demo_inference.py
  python demo_inference.py --image results/splits/reference_stain.png --backbone resnet50 --norm macenko
  python demo_inference.py --image path/to/patch.png --backbone convnext_tiny --norm none
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# Add src/ to path so histo_robust can be loaded
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from histo_robust.config import (
  CLASSES,
  CLASS_TO_IDX,
  IMAGENET_MEAN,
  IMAGENET_STD,
  NUM_CLASSES,
)
from histo_robust.models.backbones import build_model

# Clinical morphological descriptions for 9 colorectal tissue classes
CLASS_DESCRIPTIONS: Dict[str, str] = {
  "ADI": "Adipose tissue (large lipid vacuoles with eccentric nuclei)",
  "BACK": "Background clear glass slide area (no cellular tissue)",
  "DEB": "Necrotic debris, hemorrhage, or cautery burn artifact",
  "LYM": "Dense clusters of lymphocytes / immune infiltrates",
  "MUC": "Amorphous extracellular mucin pools (light basophilic staining)",
  "MUS": "Smooth muscle bundles (spindle-shaped fibers with blunt nuclei)",
  "NORM": "Normal non-neoplastic colorectal mucosa (healthy glandular crypts)",
  "STR": "Cancer-associated desmoplastic stroma (fibrous collagen matrix)",
  "TUM": "Colorectal Carcinoma (malignant dysplastic glandular epithelium)",
}


def load_image(image_path: Path) -> Image.Image:
  if not image_path.exists():
    raise FileNotFoundError(f"Image not found at {image_path}")
  return Image.open(image_path).convert("RGB")


def preprocess_image(
  img: Image.Image,
  norm_mode: str = "none",
  ref_tile_path: Path | None = None,
) -> torch.Tensor:
  """Apply optional color normalization and ImageNet standardization."""
  arr = np.array(img)

  if norm_mode in ("macenko", "reinhard") and ref_tile_path and ref_tile_path.exists():
    ref_rgb = np.array(Image.open(ref_tile_path).convert("RGB"))
    if norm_mode == "reinhard":
      from run_experiments import reinhard_apply, reinhard_fit
      ref_stats = reinhard_fit(ref_rgb)
      arr = reinhard_apply(arr, ref_stats)
    elif norm_mode == "macenko":
      from run_experiments import macenko_apply, macenko_fit
      ref_params = macenko_fit(ref_rgb)
      arr = macenko_apply(arr, ref_params)

  tensor = TF.to_tensor(arr)
  # Resize to 224x224 if necessary
  if tensor.shape[1:] != (224, 224):
    tensor = TF.resize(tensor, [224, 224])
  return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def run_inference(
  image_path: Path,
  backbone_name: str = "resnet50",
  norm_mode: str = "none",
  checkpoint_path: Path | None = None,
  ref_tile_path: Path | None = None,
  device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[str, float, Dict[str, float]]:
  """Run single-patch inference and return top prediction, confidence, and class distribution."""
  device = torch.device(device_str)
  img = load_image(image_path)

  # Preprocess
  tensor = preprocess_image(img, norm_mode=norm_mode, ref_tile_path=ref_tile_path)
  batch = tensor.unsqueeze(0).to(device)

  # Model build & load
  model = build_model(backbone_name=backbone_name, num_classes=NUM_CLASSES, pretrained=True)
  if checkpoint_path and checkpoint_path.exists():
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    print(f"[Model] Successfully loaded weights from {checkpoint_path}")
  else:
    print(f"[Model] Initialized {backbone_name} backbone with default weights.")

  model = model.to(device)
  model.eval()

  with torch.no_grad():
    logits = model(batch)
    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]

  top_idx = int(np.argmax(probs))
  top_class = CLASSES[top_idx]
  confidence = float(probs[top_idx])

  prob_dict = {CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
  return top_class, confidence, prob_dict


def render_report(
  image_path: Path,
  backbone_name: str,
  norm_mode: str,
  top_class: str,
  confidence: float,
  prob_dict: Dict[str, float],
):
  """Format and print an interactive terminal diagnosis report."""
  width = 72
  print("=" * width)
  print("   DIGITAL HISTOPATHOLOGY TISSUE CLASSIFICATION REPORT (DEMO)")
  print("=" * width)
  print(f"Target Patch      : {image_path.resolve()}")
  print(f"Architecture      : {backbone_name.upper()}")
  print(f"Stain Normalization: {norm_mode.upper()}")
  print("-" * width)
  print(f"DIAGNOSTIC CALL   : [{top_class}] - {CLASS_DESCRIPTIONS.get(top_class, '')}")
  print(f"CONFIDENCE SCORE  : {confidence * 100:.2f}%")
  print("-" * width)
  print("PREDICTED CLASS PROBABILITY DISTRIBUTION:")
  print(f"{'Class':<6} | {'Probability':<12} | {'Bar Visualization'}")
  print("-" * width)
  for c_name, p in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
    bar_len = int(p * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    flag = " <--" if c_name == top_class else ""
    print(f"{c_name:<6} | {p * 100:6.2f}%     | [{bar}]{flag}")
  print("=" * width)


def main():
  default_ref = REPO_ROOT / "results" / "splits" / "reference_stain.png"

  parser = argparse.ArgumentParser(description="Histopathology Image Classification Demo")
  parser.add_argument("--image", default=str(default_ref), help="Path to input histopathology image tile")
  parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "convnext_tiny", "phikon"], help="Model backbone")
  parser.add_argument("--norm", default="none", choices=["none", "reinhard", "macenko"], help="Stain normalization")
  parser.add_argument("--checkpoint", default=None, help="Optional trained model checkpoint (.pt)")
  parser.add_argument("--ref-stain", default=str(default_ref), help="Reference stain tile for normalization")
  args = parser.parse_args()

  img_path = Path(args.image)
  ckpt_path = Path(args.checkpoint) if args.checkpoint else None
  ref_path = Path(args.ref_stain) if args.ref_stain else None

  top_class, conf, probs = run_inference(
    image_path=img_path,
    backbone_name=args.backbone,
    norm_mode=args.norm,
    checkpoint_path=ckpt_path,
    ref_tile_path=ref_path,
  )

  render_report(
    image_path=img_path,
    backbone_name=args.backbone,
    norm_mode=args.norm,
    top_class=top_class,
    confidence=conf,
    prob_dict=probs,
  )


if __name__ == "__main__":
  main()
