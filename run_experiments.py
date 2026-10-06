from __future__ import annotations

import argparse
import copy
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406] # https://github.com/pytorch/vision/issues/1439
IMAGENET_STD = [0.229, 0.224, 0.225]

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


# Stain Normalization (Reinhard & Macenko)
# Ruderman et al. (1998) matrices for Reinhard l-alpha-beta color transfer
_LMS_MAT = np.array([
  [0.3811, 0.5783, 0.0402],
  [0.1967, 0.7244, 0.0782],
  [0.0241, 0.1288, 0.8444],
], dtype=np.float64)  #https://home.cis.rit.edu/~cnspci/references/dip/color_transfer/reinhard2001.pdf

_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
  [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
  [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],
], dtype=np.float64)

_INV_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0),  1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0), -2.0 / np.sqrt(6.0),  0.0],
], dtype=np.float64)

_INV_LMS_MAT = np.linalg.inv(_LMS_MAT)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
  norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
  lms = norm_rgb @ _LMS_MAT.T
  log_lms = np.log10(np.clip(lms, 1e-4, None))
  return log_lms @ _LAB_MAT.T


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
  log_lms = lab @ _INV_LAB_MAT.T
  lms = 10.0 ** log_lms
  rgb = lms @ _INV_LMS_MAT.T
  return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
  """Extract mean and standard deviation per channel in Reinhard LAB space."""
  lab = _rgb_to_lab(reference_rgb)
  return {
    "mean": np.mean(lab, axis=(0, 1)),
    "std": np.std(lab, axis=(0, 1)) + 1e-6,
  }


def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
  """Apply Reinhard statistical color transfer (pure NumPy)."""
  lab = _rgb_to_lab(image_rgb)
  mean = np.mean(lab, axis=(0, 1))
  std = np.std(lab, axis=(0, 1)) + 1e-6

  norm_lab = np.zeros_like(lab)
  for c in range(3):
    norm_lab[:, :, c] = ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]

  return _lab_to_rgb(norm_lab)


def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
  """Estimate H&E stain vectors and 99th percentile concentrations from reference tile."""
  od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
  flat_od = od.reshape(-1, 3)
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold
  flat_od = flat_od[mask]

  _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
  proj = flat_od @ vh[:2].T
  phi = np.arctan2(proj[:, 1], proj[:, 0])

  min_phi = np.percentile(phi, 1.0)
  max_phi = np.percentile(phi, 99.0)

  v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
  v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

  # Ensure Hematoxylin is column 0 (stronger in red absorption)
  if v1[0] < v2[0]:
    v1, v2 = v2, v1

  stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
  concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
  q99 = np.percentile(concentrations, 99.0, axis=0)

  return {"stain_matrix": stain_matrix, "q99": q99}


def macenko_apply(image_rgb: np.ndarray, ref_params: Dict[str, np.ndarray], od_threshold: float = 0.15) -> np.ndarray:
  """Normalize an H&E tile to canonical reference stain vectors and concentrations."""
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  h, w, _ = od.shape
  flat_od = od.reshape(-1, 3)
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold

  if mask.sum() < 0.20 * h * w:
    return image_rgb

  valid_od = flat_od[mask]
  try:
    _, _, vh = np.linalg.svd(valid_od, full_matrices=False)
    proj = valid_od @ vh[:2].T
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, 1.0)
    max_phi = np.percentile(phi, 99.0)

    v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
    if v1[0] < v2[0]:
      v1, v2 = v2, v1
    tile_stain = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])

    tile_conc = flat_od @ np.linalg.pinv(tile_stain).T
    q99 = np.percentile(tile_conc[mask], 99.0, axis=0) + 1e-6

    # Match reference concentration scaling
    norm_conc = tile_conc * (ref_params["q99"] / q99)
    norm_od = norm_conc @ ref_params["stain_matrix"].T
    norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0
    return np.clip(norm_rgb.reshape(h, w, 3), 0, 255).astype(np.uint8)
  except Exception:
    return image_rgb


# Data Augmentations (Geometric & HED Stain Jitter)
HE_STAIN_MATRIX = np.array([
  [0.650, 0.072],
  [0.704, 0.990],
  [0.286, 0.105],
], dtype=np.float64)
HE_STAIN_MATRIX /= np.linalg.norm(HE_STAIN_MATRIX, axis=0, keepdims=True)


def hed_stain_jitter(image_rgb: np.ndarray, sigma: float = 0.2, bias: float = 0.05) -> np.ndarray:
  """Apply biologically-grounded H&E stain concentration scaling and shifting."""
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  h, w, _ = od.shape
  flat_od = od.reshape(-1, 3)

  c = flat_od @ np.linalg.pinv(HE_STAIN_MATRIX).T
  # Stochastic independent shift and scale per stain channel
  alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
  beta = np.random.uniform(-bias, bias, size=2)

  c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
  c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)

  od_jittered = c @ HE_STAIN_MATRIX.T
  rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
  return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)


class PatchTransform:
  """Unified transform executing optional normalization, augmentation, and tensor conversion."""

  def __init__(self, policy: str = "none", norm_fn: Any = None, is_train: bool = True):
    self.policy = policy
    self.norm_fn = norm_fn
    self.is_train = is_train

  def __call__(self, img_pil: Image.Image) -> Any:
    import torchvision.transforms.functional as TF

    arr = np.array(img_pil.convert("RGB"))
    if self.norm_fn is not None:
      arr = self.norm_fn(arr)

    if self.is_train and self.policy in ("aug_stain", "aug_combined"):
      if random.random() > 0.2:
        arr = hed_stain_jitter(arr)

    tensor = TF.to_tensor(arr)  # scales [0, 255] -> [0.0, 1.0]

    if self.is_train and self.policy in ("aug_geo", "aug_combined"):
      if random.random() > 0.5:
        tensor = TF.hflip(tensor)
      if random.random() > 0.5:
        tensor = TF.vflip(tensor)
      rot = random.choice([0, 90, 180, 270])
      if rot > 0:
        tensor = TF.rotate(tensor, rot)

    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


# Dataset & Splits Preparation
class HistologyDataset:
  def __init__(self, df: pd.DataFrame, transform: PatchTransform):
    self.df = df.reset_index(drop=True)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    img = Image.open(row["image_path"])
    return self.transform(img), int(row["label_idx"])


SUPPORTED_IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def find_class_images(base_dir: Path, class_name: str) -> List[Path]:
  """Locate images for a histological class, handling case variations and nesting."""
  if not base_dir.exists() or not base_dir.is_dir():
    return []

  c_folder = None
  # 1. Direct child match (exact or case-insensitive)
  if (base_dir / class_name).is_dir():
    c_folder = base_dir / class_name
  else:
    for child in base_dir.iterdir():
      if child.is_dir() and child.name.upper() == class_name.upper():
        c_folder = child
        break

  # 2. Search recursively across subdirectories if not found directly
  if c_folder is None:
    for child in base_dir.rglob("*"):
      if child.is_dir() and child.name.upper() == class_name.upper():
        c_folder = child
        break

  if c_folder is None or not c_folder.is_dir():
    return []

  # Collect all image files with supported extensions (case-insensitive)
  images = [
    p for p in c_folder.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
  ]
  # Fallback to recursive in case patches are further nested
  if not images:
    images = [
      p for p in c_folder.rglob("*")
      if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
  return sorted(images)


def prepare_dataset_splits(
  source_dir: Path, target_dir: Path, out_dir: Path, subset_size: int = 25000, seed: int = 100
) -> Tuple[Dict[str, pd.DataFrame], Path]:
  """Generate 70/15/15 stratified source splits and pick canonical reference tile."""
  out_dir.mkdir(parents=True, exist_ok=True)
  np.random.seed(seed)
  random.seed(seed)

  # Collect source domain patches
  source_records = []
  for c_idx, c_name in enumerate(CLASSES):
    c_images = find_class_images(source_dir, c_name)
    for p in c_images:
      source_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "source"})

  source_df = pd.DataFrame(source_records)
  if len(source_df) == 0:
    items = [p.name for p in list(source_dir.iterdir())[:15]] if source_dir.is_dir() else "directory does not exist"
    raise RuntimeError(
      f"No source images found under {source_dir}.\n"
      f"Contents found at {source_dir}: {items}.\n"
      f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
    )

  # Stratified subset if requested
  if 0 < subset_size < len(source_df):
    per_class = subset_size // NUM_CLASSES
    source_df = source_df.groupby("class_name", group_keys=False).apply(
      lambda g: g.sample(min(len(g), per_class), random_state=seed)
    ).reset_index(drop=True)

  # Stratified 70 / 15 / 15 split
  from sklearn.model_selection import train_test_split
  train_df, rest_df = train_test_split(source_df, test_size=0.30, stratify=source_df["class_name"], random_state=seed)
  val_df, test_id_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["class_name"], random_state=seed)

  # Collect target domain (100% out-of-domain)
  target_records = []
  for c_idx, c_name in enumerate(CLASSES):
    c_images = find_class_images(target_dir, c_name)
    for p in c_images:
      target_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "target"})

  test_ood_df = pd.DataFrame(target_records)
  if len(test_ood_df) == 0:
    items = [p.name for p in list(target_dir.iterdir())[:15]] if target_dir.is_dir() else "directory does not exist"
    raise RuntimeError(
      f"No target images found under {target_dir}.\n"
      f"Contents found at {target_dir}: {items}.\n"
      f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
    )

  # Save CSVs
  splits = {"train": train_df, "val_id": val_df, "test_id": test_id_df, "test_ood": test_ood_df}
  for name, df in splits.items():
    df.to_csv(out_dir / f"{name}.csv", index=False)

  # Pick reference tile strictly from training split (dense colorectal tumor tile)
  tum_train = train_df[train_df["class_name"] == "TUM"]
  ref_path = Path(tum_train.iloc[0]["image_path"])
  ref_img = Image.open(ref_path).convert("RGB")
  ref_save_path = out_dir / "reference_stain.png"
  ref_img.save(ref_save_path)

  print(f"[splits] Train={len(train_df)} | Val={len(val_df)} | Test-ID={len(test_id_df)} | Test-OOD={len(test_ood_df)}")
  print(f"[reference] Canonical stain reference: {ref_path.name}")
  return splits, ref_save_path


# Model Architecture (ResNet-50, ConvNeXt-Tiny, Phikon)
def build_model(backbone_name: str, num_classes: int = NUM_CLASSES):
  import timm
  import torch.nn as nn

  if backbone_name == "phikon":
    from transformers import AutoModel
    class PhikonClassifier(nn.Module):
      def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("owkin/phikon")
        # Linear probing: freeze ViT backbone
        for p in self.encoder.parameters():
          p.requires_grad = False
        self.fc = nn.Linear(768, num_classes)

      def forward(self, x):
        feat = self.encoder(x).last_hidden_state[:, 0]
        return self.fc(feat)

    return PhikonClassifier()

  if backbone_name == "convnext_tiny":
    return timm.create_model("convnext_tiny.fb_in1k", pretrained=True, num_classes=num_classes)

  # Default: ResNet-50
  return timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)


# Training & Evaluation Functions
def evaluate_model(model, loader, device) -> Dict[str, float]:
  import torch
  from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

  model.eval()
  all_preds, all_probs, all_targets = [], [], []

  with torch.no_grad():
    for images, targets in loader:
      images = images.to(device, non_blocking=True)
      with torch.amp.autocast("cuda"):
        logits = model(images)
      probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
      all_probs.append(probs)
      all_preds.append(np.argmax(probs, axis=1))
      all_targets.append(targets.numpy())

  y_true = np.concatenate(all_targets)
  y_pred = np.concatenate(all_preds)
  y_prob = np.concatenate(all_probs)

  macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
  bal_acc = balanced_accuracy_score(y_true, y_pred)
  acc = accuracy_score(y_true, y_pred)
  try:
    auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
  except Exception:
    auroc = float("nan")

  return {"macro_f1": float(macro_f1), "balanced_acc": float(bal_acc), "accuracy": float(acc), "auroc": float(auroc)}


def train_experiment(
  exp: Dict[str, str],
  splits: Dict[str, pd.DataFrame],
  ref_img_path: Path,
  epochs: int = 8,
  batch_size: int = 64,
  lr: float = 1e-3,
  device_str: str = "cuda",
) -> Dict[str, Any]:
  """Train one experiment cell and compute in-domain and out-of-domain robustness."""
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader

  device = torch.device(device_str)
  ref_rgb = np.array(Image.open(ref_img_path).convert("RGB"))

  # Setup normalizer
  norm_fn = None
  if exp["norm"] == "reinhard":
    ref_stats = reinhard_fit(ref_rgb)
    norm_fn = lambda img: reinhard_apply(img, ref_stats)
  elif exp["norm"] == "macenko":
    ref_params = macenko_fit(ref_rgb)
    norm_fn = lambda img: macenko_apply(img, ref_params)

  train_ds = HistologyDataset(splits["train"], PatchTransform(policy=exp["aug"], norm_fn=norm_fn, is_train=True))
  val_ds = HistologyDataset(splits["val_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
  test_id_ds = HistologyDataset(splits["test_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
  test_ood_ds = HistologyDataset(splits["test_ood"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))

  num_workers = 4 if os.name != "nt" else 0
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  test_id_loader = DataLoader(test_id_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  test_ood_loader = DataLoader(test_ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  model = build_model(exp["backbone"]).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr if exp["backbone"] != "phikon" else 3e-4, weight_decay=0.05) #https://arxiv.org/pdf/1711.05101
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5) #https://arxiv.org/pdf/1608.03983
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #Kỹ thuật Regularization Label Smoothing (Szegedy et al., 2016)
  scaler = torch.amp.GradScaler("cuda")

  best_val_f1 = -1.0
  best_state = None
  t0 = time.time()

  print(f"\n[{exp['id']}] Training {exp['backbone']} (norm={exp['norm']}, aug={exp['aug']}) for {epochs} epochs...")
  for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for images, targets in train_loader:
      images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast("cuda"):
        loss = criterion(model(images), targets)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      total_loss += loss.item()

    val_res = evaluate_model(model, val_loader, device)
    if val_res["macro_f1"] > best_val_f1:
      best_val_f1 = val_res["macro_f1"]
      best_state = copy.deepcopy(model.state_dict())

    print(f"  ep {epoch}/{epochs} | loss={total_loss / len(train_loader):.4f} | val_f1={val_res['macro_f1']:.4f} (best={best_val_f1:.4f})")

  # Load best checkpoint for test evaluations
  if best_state is not None:
    model.load_state_dict(best_state)

  id_res = evaluate_model(model, test_id_loader, device)
  ood_res = evaluate_model(model, test_ood_loader, device)
  trained_min = round((time.time() - t0) / 60.0, 2)

  delta_f1 = round(id_res["macro_f1"] - ood_res["macro_f1"], 4)
  rr_f1 = round((ood_res["macro_f1"] / max(id_res["macro_f1"], 1e-6)) * 100.0, 2)

  result = {
    "exp_id": exp["id"],
    "stage": exp["stage"],
    "backbone": exp["backbone"],
    "norm": exp["norm"],
    "aug": exp["aug"],
    "best_val_f1": round(best_val_f1, 4),
    "test_id_f1": round(id_res["macro_f1"], 4),
    "test_ood_f1": round(ood_res["macro_f1"], 4),
    "delta_f1": delta_f1,
    "rr_f1": rr_f1,
    "test_id_acc": round(id_res["accuracy"], 4),
    "test_ood_acc": round(ood_res["accuracy"], 4),
    "minutes": trained_min,
  }
  print(f"[{exp['id']} Result] ID F1={result['test_id_f1']} | OOD F1={result['test_ood_f1']} | Delta F1={delta_f1} | RR={rr_f1}% ({trained_min}m)")
  return result


# Main Orchestrator
def main():
  parser = argparse.ArgumentParser(description="Run 13-cell Histopathology Staining Robustness Ablations")
  parser.add_argument("--data-root", default="data/raw/NCT-CRC-HE-100K-NONORM", help="Path to in-domain dataset")
  parser.add_argument("--target-root", default="data/raw/CRC-VAL-HE-7K", help="Path to out-of-domain dataset")
  parser.add_argument("--out-dir", default="results", help="Directory for output metrics and tables")
  parser.add_argument("--subset", type=int, default=25000, help="Subset size (default 25000; 0 for full 100k)")
  parser.add_argument("--epochs", type=int, default=8, help="Epochs per experiment")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--experiments", nargs="*", default=None, help="Filter experiment IDs to run (e.g. EXP-01 EXP-02)")
  parser.add_argument("--dry-run", action="store_true", help="Print experiment plan and exit")
  args = parser.parse_args()

  out_path = Path(args.out_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  exp_list = EXPERIMENTS
  if args.experiments:
    wanted = set(args.experiments)
    exp_list = [e for e in exp_list if e["id"] in wanted]

  if args.dry_run:
    print(f"Plan: {len(exp_list)} experiments queued (subset={args.subset}, epochs={args.epochs}, batch={args.batch_size})")
    for e in exp_list:
      print(f"  {e['id']} | {e['stage']} | backbone={e['backbone']:<13} | norm={e['norm']:<8} | aug={e['aug']:<12} | {e['notes']}")
    return

  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Running on device: {device} | Total experiments: {len(exp_list)}")

  splits_dir = out_path / "splits"
  splits, ref_tile = prepare_dataset_splits(
    source_dir=Path(args.data_root),
    target_dir=Path(args.target_root),
    out_dir=splits_dir,
    subset_size=args.subset,
  )

  results = []
  for exp in exp_list:
    res = train_experiment(
      exp=exp,
      splits=splits,
      ref_img_path=ref_tile,
      epochs=args.epochs,
      batch_size=args.batch_size,
      device_str=device,
    )
    results.append(res)

  # Output final summary table
  res_df = pd.DataFrame(results)
  csv_file = out_path / "summary_results.csv"
  md_file = out_path / "RESULTS_TABLE.md"

  res_df.to_csv(csv_file, index=False)
  try:
    md_content = res_df.to_markdown(index=False)
  except Exception:
    md_content = res_df.to_string(index=False)

  md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")

  print("\n" + "=" * 80)
  print("FINAL ABLATION RESULTS TABLE")
  print("=" * 80)
  print(md_content)
  print(f"\n[Artifacts] Wrote {csv_file} and {md_file}")


if __name__ == "__main__":
  main()
