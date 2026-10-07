"""Gói con chuẩn hóa màu nhuộm mô bệnh học."""

# Nhập giao diện cơ sở trừu tượng
from src.normalization.base import BaseStainNormalizer
# Nhập bộ chuẩn hóa Reinhard và các hàm tiện ích liên quan
from src.normalization.reinhard import (
    ReinhardNormalizer,
    reinhard_fit,
    reinhard_apply,
    rgb_to_lab,
    lab_to_rgb,
)
# Nhập bộ chuẩn hóa Macenko và các hàm tiện ích liên quan
from src.normalization.macenko import (
    MacenkoNormalizer,
    macenko_fit,
    macenko_apply,
)
# Nhập nhà máy khởi tạo NormalizerFactory
from src.normalization.factory import NormalizerFactory

# Danh sách xuất khẩu công khai của gói con normalization
__all__ = [
    "BaseStainNormalizer",
    "ReinhardNormalizer",
    "reinhard_fit",
    "reinhard_apply",
    "rgb_to_lab",
    "lab_to_rgb",
    "MacenkoNormalizer",
    "macenko_fit",
    "macenko_apply",
    "NormalizerFactory",
]
