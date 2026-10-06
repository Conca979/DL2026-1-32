"""Entrypoint for running the 13-cell histopathology staining robustness ablation suite.

Isolated runner located inside `src/`. It only accesses the external `data/` folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is in sys.path so `histo_robust` is importable regardless of working directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))


def main():
  try:
    from histo_robust.engine.runner import main as run_pipeline
    run_pipeline()
  except ImportError:
    # Fallback during Sprint 0 / 1 transition
    repo_root = SRC_DIR.parent
    mono_script = repo_root / "run_experiments.py"
    if mono_script.exists():
      import runpy
      print("[Notice] histo_robust package modularization in progress. Running fallback...")
      runpy.run_path(str(mono_script), run_name="__main__")
    else:
      raise


if __name__ == "__main__":
  main()