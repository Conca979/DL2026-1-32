"""Lớp cơ sở trừu tượng (Abstract Base Class) cho các thuật toán chuẩn hóa màu nhuộm."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Nhập các công cụ định nghĩa lớp trừu tượng
from abc import ABC, abstractmethod
# Thư viện mảng số học NumPy
import numpy as np


class BaseStainNormalizer(ABC):
    """Giao diện tiêu chuẩn (Interface) cho tất cả các bộ chuẩn hóa màu nhuộm mô bệnh học."""

    def __init__(self) -> None:
        # Cờ trạng thái ghi nhận xem bộ chuẩn hóa đã được khớp (fit) với ảnh tham chiếu hay chưa
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Thuộc tính trả về True nếu các thống kê hoặc ma trận chất nhuộm đã được ước lượng."""
        return self._is_fitted

    @abstractmethod
    def fit(self, reference_rgb: np.ndarray) -> BaseStainNormalizer:
        """Ước lượng các giá trị thống kê hoặc véc-tơ màu nhuộm từ ảnh tham chiếu chuẩn.

        Tham số:
            reference_rgb: Mảng ảnh RGB kích thước (H, W, 3) với giá trị pixel trong khoảng [0, 255].

        Trả về:
            Chính đối tượng normalizer đã ở trạng thái fitted.
        """
        pass

    @abstractmethod
    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Chuẩn hóa một mảnh mô bệnh học mục tiêu về hệ quy chiếu màu đã khớp.

        Tham số:
            image_rgb: Mảng ảnh RGB kích thước (H, W, 3) với giá trị pixel trong khoảng [0, 255].

        Trả về:
            Mảng ảnh RGB đã chuẩn hóa có kích thước (H, W, 3) với kiểu dữ liệu uint8 [0, 255].
        """
        pass

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Cho phép gọi đối tượng trực tiếp như một hàm, chuyển tiếp đến phương thức transform."""
        return self.transform(image_rgb)
