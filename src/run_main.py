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

# Also ensure repo root is in sys.path for fallback compatibility if peer modules are not yet migrated
REPO_ROOT = SRC_DIR.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.append(str(REPO_ROOT))

from histo_robust.engine.runner import main

if __name__ == "__main__":
  main()