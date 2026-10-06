from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image

# Modular imports with graceful fallback to monolithic run_experiments for peer branches
try:
  from histo_robust.data.dataset import HistologyDataset
except (ImportError, ModuleNotFoundError):
  from run_experiments import HistologyDataset

try:
  from histo_robust.transforms.normalization import (
    macenko_apply,
    macenko_fit,
    reinhard_apply,
    reinhard_fit,
  )
  from histo_robust.transforms.pipeline import PatchTransform
except (ImportError, ModuleNotFoundError):
  from run_experiments import (
    PatchTransform,
    macenko_apply,
    macenko_fit,
    reinhard_apply,
    reinhard_fit,
  )

try:
  from histo_robust.models.backbones import build_model
  from histo_robust.models.metrics import evaluate_model
except (ImportError, ModuleNotFoundError):
  from run_experiments import build_model, evaluate_model


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
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr if exp["backbone"] != "phikon" else 3e-4, weight_decay=0.05)  # https://arxiv.org/pdf/1711.05101
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)  # https://arxiv.org/pdf/1608.03983
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Kỹ thuật Regularization Label Smoothing (Szegedy et al., 2016)
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


__all__ = ["train_experiment"]
