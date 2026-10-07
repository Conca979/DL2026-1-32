"""Đường ống biến đổi patch thống nhất: kết hợp Chuẩn hóa, Tăng cường và Chuyển đổi Tensor."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module sinh số ngẫu nhiên
import random
# Các kiểu dữ liệu phụ trợ
from typing import Any, Callable, Optional, Union
# Thư viện tính toán mảng số học NumPy
import numpy as np
# Thư viện ảnh Pillow
from PIL import Image

# Nhập các hằng số phân phối của ImageNet
from src.config.constants import IMAGENET_MEAN, IMAGENET_STD
# Nhập hàm gây nhiễu màu sinh học
from src.augmentation.hed_jitter import hed_stain_jitter
# Nhập kiểu liệt kê chính sách tăng cường
from src.augmentation.policies import AugmentationPolicy


class PatchTransform:
    """Pipeline biến đổi thống nhất thực thi chuẩn hóa màu tùy chọn, augment và chuẩn hóa tensor."""

    def __init__(
        self,
        policy: Union[str, AugmentationPolicy] = AugmentationPolicy.NONE,
        norm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        is_train: bool = True,
    ) -> None:
        # Chuỗi tên chính sách tăng cường dữ liệu
        self.policy: str = policy.value if isinstance(policy, AugmentationPolicy) else str(policy)
        # Hàm chuẩn hóa màu sắc (Reinhard, Macenko hoặc None)
        self.norm_fn = norm_fn
        # Cờ phân biệt tập huấn luyện hay tập đánh giá
        self.is_train = is_train

    def __call__(self, img_pil: Image.Image) -> Any:
        """Thực thi toàn bộ chuỗi biến đổi tuần tự trên đối tượng ảnh PIL."""
        # Nhập các hàm biến đổi chức năng từ torchvision
        import torchvision.transforms.functional as TF

        # 1. Chuyển ảnh PIL sang mảng NumPy RGB uint8
        arr = np.array(img_pil.convert("RGB"))

        # 2. Áp dụng chuẩn hóa màu sắc nếu có cấu hình
        if self.norm_fn is not None:
            arr = self.norm_fn(arr)

        # 3. Áp dụng nhiễu nồng độ màu nhuộm trong khi huấn luyện (với xác suất 80%)
        if self.is_train and self.policy in ("aug_stain", "aug_combined"):
            if random.random() > 0.2:
                arr = hed_stain_jitter(arr)

        # 4. Chuyển đổi mảng NumPy sang PyTorch Tensor (co giãn từ [0, 255] về [0.0, 1.0])
        tensor = TF.to_tensor(arr)

        # 5. Áp dụng biến đổi không gian hình học khi huấn luyện
        if self.is_train and self.policy in ("aug_geo", "aug_combined"):
            # Lật ảnh ngẫu nhiên theo chiều ngang (xác suất 50%)
            if random.random() > 0.5:
                tensor = TF.hflip(tensor)
            # Lật ảnh ngẫu nhiên theo chiều dọc (xác suất 50%)
            if random.random() > 0.5:
                tensor = TF.vflip(tensor)
            # Chọn góc xoay ngẫu nhiên từ tập các góc trực giao
            rot = random.choice([0, 90, 180, 270])
            # Xoay ảnh nếu góc chọn lớn hơn 0 độ
            if rot > 0:
                tensor = TF.rotate(tensor, rot)

        # 6. Chuẩn hóa phân phối chuẩn theo ImageNet mean và std
        return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
