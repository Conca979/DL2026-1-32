"""Model architectures for histopathology image classification.

Supports:
- ResNet-50 (timm: resnet50.a1_in1k)
- ConvNeXt-Tiny (timm: convnext_tiny.fb_in1k)
- Phikon Foundation Model (owkin/phikon: ViT-B/16 with frozen encoder and linear probe)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import NUM_CLASSES


class PhikonClassifier(nn.Module):
  """Linear probe classifier on top of frozen Owkin Phikon ViT-B/16 features.

  The ViT-B/16 encoder was pretrained on 43 million TCGA histology patches using
  the iBOT self-supervised framework. All 86 million backbone parameters are frozen,
  leaving only the 768 -> num_classes linear head trainable.
  """

  def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    super().__init__()
    from transformers import AutoConfig, AutoModel

    if pretrained:
      self.encoder = AutoModel.from_pretrained("owkin/phikon")
    else:
      config = AutoConfig.from_pretrained("owkin/phikon")
      self.encoder = AutoModel.from_config(config)

    # Freeze ViT backbone parameters for linear probing
    for p in self.encoder.parameters():
      p.requires_grad = False

    self.fc = nn.Linear(768, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Extract [CLS] token at index 0 from last hidden state
    feat = self.encoder(x).last_hidden_state[:, 0]
    return self.fc(feat)


def build_model(
  backbone_name: str,
  num_classes: int = NUM_CLASSES,
  pretrained: bool = True,
) -> nn.Module:
  """Build and initialize classification model backbone.

  Args:
    backbone_name: One of 'resnet50', 'convnext_tiny', 'phikon'.
    num_classes: Number of output target classes (default 9).
    pretrained: Whether to load pre-trained weights (ImageNet-1k or TCGA).

  Returns:
    nn.Module configured for the specified number of classes.
  """
  name = backbone_name.lower().strip()

  if name == "phikon":
    return PhikonClassifier(num_classes=num_classes, pretrained=pretrained)

  import timm

  if name in ("convnext_tiny", "convnext-tiny", "convnext"):
    return timm.create_model("convnext_tiny.fb_in1k", pretrained=pretrained, num_classes=num_classes)

  # Default and baseline: ResNet-50
  return timm.create_model("resnet50.a1_in1k", pretrained=pretrained, num_classes=num_classes)
