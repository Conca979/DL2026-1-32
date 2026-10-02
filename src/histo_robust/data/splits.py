from __future__ import annotations

from pathlib import Path
from typing import List, Set

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
