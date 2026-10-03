from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from histo_robust.config import CLASSES, NUM_CLASSES

SUPPORTED_IMAGE_EXTS: Set[str] = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


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
  source_dir: Path,
  target_dir: Path,
  out_dir: Path,
  subset_size: int = 25000,
  seed: int = 42,
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
      source_records.append({
        "image_path": str(p),
        "class_name": c_name,
        "label_idx": c_idx,
        "domain": "source",
      })

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
    source_df = (
      source_df.groupby("class_name", group_keys=False)
      .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
      .reset_index(drop=True)
    )

  # Stratified 70 / 15 / 15 split
  train_df, rest_df = train_test_split(
    source_df, test_size=0.30, stratify=source_df["class_name"], random_state=seed
  )
  val_df, test_id_df = train_test_split(
    rest_df, test_size=0.50, stratify=rest_df["class_name"], random_state=seed
  )

  # Collect target domain (100% out-of-domain firewall)
  target_records = []
  for c_idx, c_name in enumerate(CLASSES):
    c_images = find_class_images(target_dir, c_name)
    for p in c_images:
      target_records.append({
        "image_path": str(p),
        "class_name": c_name,
        "label_idx": c_idx,
        "domain": "target",
      })

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
