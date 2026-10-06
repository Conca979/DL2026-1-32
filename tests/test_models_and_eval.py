"""Unit and Integration Tests for Ánh Dương's models and evaluation module.

Tests:
1. Config constants and 9 classes integrity.
2. Model building: ResNet-50 and ConvNeXt-Tiny architectures.
3. Forward pass tensor shapes: (B, 3, 224, 224) -> (B, 9).
4. Evaluation metrics computation (Macro-F1, Balanced Acc, Top-1 Acc, AUROC).
5. evaluate_model execution with DataLoader.
6. Confusion matrix & per-class dictionary output.
"""

import sys
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure src/ is on Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from histo_robust.config import (
  CLASSES,
  CLASS_TO_IDX,
  NUM_CLASSES,
  EXPERIMENTS,
  IMAGENET_MEAN,
  IMAGENET_STD,
)
from histo_robust.models.backbones import build_model
from histo_robust.models.metrics import compute_metrics, evaluate_model


class TestHistoConfig(unittest.TestCase):
  """Verify configuration integrity for histopathology pipeline."""

  def test_classes_integrity(self):
    self.assertEqual(NUM_CLASSES, 9)
    self.assertEqual(len(CLASSES), 9)
    self.assertIn("TUM", CLASSES)
    self.assertIn("STR", CLASSES)
    self.assertIn("MUS", CLASSES)
    self.assertEqual(CLASS_TO_IDX["TUM"], 8)
    self.assertEqual(CLASS_TO_IDX["ADI"], 0)

  def test_experiments_matrix(self):
    self.assertEqual(len(EXPERIMENTS), 13)
    exp_ids = [e["id"] for e in EXPERIMENTS]
    expected_ids = [f"EXP-{i:02d}" for i in range(1, 14)]
    self.assertEqual(exp_ids, expected_ids)


class TestModelBackbones(unittest.TestCase):
  """Verify model building and forward pass for vision backbones."""

  def test_resnet50_build_and_forward(self):
    model = build_model("resnet50", num_classes=9, pretrained=False)
    self.assertIsInstance(model, nn.Module)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      output = model(dummy_input)
    self.assertEqual(output.shape, (2, 9))

  def test_convnext_tiny_build_and_forward(self):
    model = build_model("convnext_tiny", num_classes=9, pretrained=False)
    self.assertIsInstance(model, nn.Module)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
      output = model(dummy_input)
    self.assertEqual(output.shape, (2, 9))


class TestEvaluationMetrics(unittest.TestCase):
  """Verify metric computations and edge cases."""

  def test_perfect_predictions(self):
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    y_prob = np.eye(9)

    res = compute_metrics(y_true, y_pred, y_prob)
    self.assertAlmostEqual(res["macro_f1"], 1.0)
    self.assertAlmostEqual(res["balanced_acc"], 1.0)
    self.assertAlmostEqual(res["accuracy"], 1.0)
    self.assertAlmostEqual(res["auroc"], 1.0)

  def test_partially_incorrect_predictions(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # class 0: p=1/1=1.0, r=1/2=0.5 -> f1=2/3
    # class 1: p=2/3, r=2/2=1.0 -> f1=4/5
    res = compute_metrics(y_true, y_pred)
    self.assertGreater(res["macro_f1"], 0.0)
    self.assertLess(res["macro_f1"], 1.0)

  def test_evaluate_model_dataloader(self):
    # Dummy linear model with 9 classes
    class SimpleToyModel(nn.Module):
      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 9)

      def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

    toy_model = SimpleToyModel()
    dummy_x = torch.randn(18, 3, 8, 8)
    dummy_y = torch.tensor([i % 9 for i in range(18)], dtype=torch.long)
    dataset = TensorDataset(dummy_x, dummy_y)
    loader = DataLoader(dataset, batch_size=6, shuffle=False)

    results = evaluate_model(
      model=toy_model,
      loader=loader,
      device="cpu",
      return_per_class=True,
    )

    self.assertIn("macro_f1", results)
    self.assertIn("accuracy", results)
    self.assertIn("per_class", results)
    self.assertIn("confusion_matrix", results)
    self.assertEqual(len(results["per_class"]), 9)
    self.assertEqual(len(results["confusion_matrix"]), 9)
    self.assertEqual(len(results["confusion_matrix"][0]), 9)


if __name__ == "__main__":
  unittest.main(verbosity=2)
