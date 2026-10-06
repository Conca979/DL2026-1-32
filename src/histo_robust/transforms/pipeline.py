import random
from typing import Any

import numpy as np
from PIL import Image

from .augmentation import hed_stain_jitter

# Import IMAGENET_MEAN and IMAGENET_STD from config if available, 
# otherwise define fallback for testing (Member 4 should define in config.py)
try:
  from ..config import IMAGENET_MEAN, IMAGENET_STD
except ImportError:
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]


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
