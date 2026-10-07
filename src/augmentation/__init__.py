"""Gói con tăng cường dữ liệu và tiền xử lý hình ảnh."""

# Nhập kiểu liệt kê chính sách tăng cường
from src.augmentation.policies import AugmentationPolicy
# Nhập hàm và lớp nhiễu nồng độ HED
from src.augmentation.hed_jitter import hed_stain_jitter, HEDStainJitterTransform
# Nhập pipeline biến đổi patch tổng thể
from src.augmentation.transforms import PatchTransform

# Danh sách biểu tượng xuất khẩu của gói augmentation
__all__ = [
    "AugmentationPolicy",
    "hed_stain_jitter",
    "HEDStainJitterTransform",
    "PatchTransform",
]
