#!/usr/bin/env python
"""Create dist/histo-robust-code.zip containing only run_experiments.py for Kaggle."""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
  out_dir = REPO_ROOT / "dist"
  out_dir.mkdir(parents=True, exist_ok=True)
  zip_path = out_dir / "histo-robust-code.zip"
  target_file = REPO_ROOT / "run_experiments.py"

  if not target_file.exists():
    raise FileNotFoundError(f"Missing {target_file}")

  print(f"Packaging {target_file.name} into {zip_path}...")
  with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.write(target_file, arcname="run_experiments.py")

  print(f"[OK] Wrote {zip_path} (1 file, {zip_path.stat().st_size / 1024:.1f} KB)")
  print("Upload this zip to Kaggle dataset 'histo-robust-code'.")


if __name__ == "__main__":
  main()
