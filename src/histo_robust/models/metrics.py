"""Evaluation metrics for digital histopathology tissue classification.

Implements:
- Macro-averaged F1 (primary metric treating all 9 tissue types equally)
- Balanced Accuracy (macro-averaged sensitivity)
- Top-1 Accuracy
- Macro One-vs-Rest AUROC
- Confusion Matrix and Per-class metrics
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  confusion_matrix,
  f1_score,
  precision_recall_fscore_support,
  roc_auc_score,
)

from ..config import CLASSES, NUM_CLASSES


def compute_metrics(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
  """Compute comprehensive evaluation metrics from true and predicted arrays."""
  macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
  bal_acc = float(balanced_accuracy_score(y_true, y_pred))
  acc = float(accuracy_score(y_true, y_pred))

  auroc = float("nan")
  if y_prob is not None:
    try:
      auroc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
      auroc = float("nan")

  return {
    "macro_f1": macro_f1,
    "balanced_acc": bal_acc,
    "accuracy": acc,
    "auroc": auroc,
  }


def evaluate_model(
  model: torch.nn.Module,
  loader: Any,
  device: torch.device | str,
  return_per_class: bool = False,
) -> Dict[str, Any]:
  """Evaluate a PyTorch model on a DataLoader.

  Compatible with both CUDA (automatic mixed precision) and CPU execution.

  Args:
    model: PyTorch classification model.
    loader: DataLoader yielding (images, targets).
    device: Execution device ('cuda' or 'cpu' or torch.device).
    return_per_class: Whether to include per-class metrics and confusion matrix.

  Returns:
    Dict containing 'macro_f1', 'balanced_acc', 'accuracy', 'auroc'
    and optionally 'per_class' and 'confusion_matrix'.
  """
  if isinstance(device, str):
    device = torch.device(device)

  model.eval()
  all_preds, all_probs, all_targets = [], [], []
  is_cuda = device.type == "cuda"

  with torch.no_grad():
    for images, targets in loader:
      images = images.to(device, non_blocking=True)
      if is_cuda:
        with torch.amp.autocast("cuda"):
          logits = model(images)
      else:
        logits = model(images)

      probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
      all_probs.append(probs)
      all_preds.append(np.argmax(probs, axis=1))

      if isinstance(targets, torch.Tensor):
        all_targets.append(targets.cpu().numpy())
      else:
        all_targets.append(np.array(targets))

  y_true = np.concatenate(all_targets)
  y_pred = np.concatenate(all_preds)
  y_prob = np.concatenate(all_probs)

  metrics = compute_metrics(y_true, y_pred, y_prob)

  if return_per_class:
    p, r, f, s = precision_recall_fscore_support(
      y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class = {}
    for i, c_name in enumerate(CLASSES):
      per_class[c_name] = {
        "precision": float(p[i]),
        "recall": float(r[i]),
        "f1": float(f[i]),
        "support": int(s[i]),
      }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = cm.tolist()

  return metrics
