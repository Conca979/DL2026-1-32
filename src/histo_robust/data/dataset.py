from __future__ import annotations

from typing import Any, Callable, Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HistologyDataset(Dataset):
  """PyTorch Dataset loading histology patch images and assigning class indices."""
  def __init__(self, df: pd.DataFrame, transform: Optional[Callable[[Image.Image], Any]] = None):
    self.df = df.reset_index(drop=True)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.df)

  def __getitem__(self, idx: int):
    row = self.df.iloc[idx]
    img = Image.open(row["image_path"])
    sample = self.transform(img) if self.transform is not None else img
    return sample, int(row["label_idx"])
