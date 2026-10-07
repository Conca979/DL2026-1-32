"""Lớp nhà máy khởi tạo các PyTorch DataLoader cho các phân tập thí nghiệm."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module tương tác hệ điều hành
import os
# Các kiểu dữ liệu phụ trợ
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
# Thư viện mảng số học NumPy
import numpy as np

# Nhập kiểu liệt kê chính sách tăng cường
from src.augmentation.policies import AugmentationPolicy
# Nhập pipeline biến đổi patch
from src.augmentation.transforms import PatchTransform
# Nhập lớp Dataset mô học
from src.data.dataset import HistologyDataset

# Hỗ trợ type hinting tĩnh
if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.data import DataLoader


def create_dataloaders(
    splits: Dict[str, Any],
    norm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    aug_policy: str = "none",
    batch_size: int = 64,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
) -> Dict[str, Any]:
    """Xây dựng 4 bộ nạp dữ liệu PyTorch DataLoader tiêu chuẩn cho 4 phân tập thí nghiệm.

    Tham số:
        splits: Từ điển chứa các DataFrame 'train', 'val_id', 'test_id', và 'test_ood'.
        norm_fn: Hàm chuẩn hóa màu sắc tùy chọn.
        aug_policy: Tên chính sách tăng cường dữ liệu (ví dụ: 'none', 'aug_geo', 'aug_combined').
        batch_size: Kích thước lô nạp mẫu cho huấn luyện và đánh giá.
        num_workers: Số tiến trình nạp dữ liệu song song (mặc định 4 trên Linux, 0 trên Windows).
        pin_memory: Khóa bộ nhớ vật lý để tăng tốc độ truyền qua GPU bằng DMA.

    Trả về:
        Từ điển ánh xạ tên phân tập sang đối tượng PyTorch DataLoader tương ứng.
    """
    # Nhập DataLoader an toàn trong thời gian chạy
    try:
        from torch.utils.data import DataLoader
    except ImportError as err:
        raise ImportError(
            "PyTorch is required to build DataLoaders: pip install torch"
        ) from err

    # Thiết lập số luồng nếu chưa chỉ định rõ
    if num_workers is None:
        num_workers = 4 if os.name != "nt" else 0

    # Khởi tạo pipeline biến đổi cho tập huấn luyện (bật cờ is_train=True)
    train_transform = PatchTransform(
        policy=AugmentationPolicy.from_str(aug_policy),
        norm_fn=norm_fn,
        is_train=True,
    )
    # Khởi tạo pipeline biến đổi cho các tập đánh giá (tắt cờ is_train=False, không augment)
    eval_transform = PatchTransform(
        policy=AugmentationPolicy.NONE,
        norm_fn=norm_fn,
        is_train=False,
    )

    # Đóng gói các tập Dataset
    train_ds = HistologyDataset(splits["train"], train_transform)
    val_ds = HistologyDataset(splits["val_id"], eval_transform)
    test_id_ds = HistologyDataset(splits["test_id"], eval_transform)
    test_ood_ds = HistologyDataset(splits["test_ood"], eval_transform)

    # Trả về từ điển các DataLoader
    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,              # Xáo trộn dữ liệu huấn luyện sau mỗi epoch
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val_id": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,             # Giữ nguyên thứ tự đánh giá nhất quán
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test_id": DataLoader(
            test_id_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test_ood": DataLoader(
            test_ood_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }
