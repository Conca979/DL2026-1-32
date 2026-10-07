"""Lớp Factory thiết kế theo mẫu thiết kế Factory Pattern khởi tạo các bộ chuẩn hóa màu nhuộm."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Các kiểu dữ liệu phụ trợ
from typing import Optional
# Thư viện tính toán mảng số học NumPy
import numpy as np
# Nhập lớp cơ sở trừu tượng
from src.normalization.base import BaseStainNormalizer
# Nhập bộ chuẩn hóa màu Reinhard
from src.normalization.reinhard import ReinhardNormalizer
# Nhập bộ chuẩn hóa màu Macenko
from src.normalization.macenko import MacenkoNormalizer


class NormalizerFactory:
    """Lớp nhà máy khởi tạo các thuật toán chuẩn hóa màu nhuộm theo tên chính sách."""

    @staticmethod
    def create(
        norm_type: str,
        reference_rgb: Optional[np.ndarray] = None,
    ) -> Optional[BaseStainNormalizer]:
        """Khởi tạo và tùy chọn khớp (fit) ngay lập tức bộ chuẩn hóa với ảnh mẫu tham chiếu.

        Tham số:
            norm_type: Tên phương pháp ('none', 'reinhard', 'macenko').
            reference_rgb: Mảng ảnh tham chiếu tùy chọn để fit ngay sau khi khởi tạo.

        Trả về:
            Đối tượng kế thừa BaseStainNormalizer đã được fit, hoặc None nếu norm_type là 'none'.
        """
        # Làm sạch chuỗi tên phương pháp: loại bỏ khoảng trắng và chuyển thành chữ thường
        normalized_name = (norm_type or "none").strip().lower()

        # Trường hợp không dùng chuẩn hóa màu (giữ nguyên ảnh gốc thô)
        if normalized_name in ("", "none", "raw"):
            return None

        # Trường hợp dùng chuẩn hóa thống kê Reinhard
        if normalized_name == "reinhard":
            normalizer = ReinhardNormalizer()
            # Nếu có truyền ảnh tham chiếu thì thực hiện fit luôn
            if reference_rgb is not None:
                normalizer.fit(reference_rgb)
            return normalizer

        # Trường hợp dùng chuẩn hóa phân rã quang học Macenko
        if normalized_name == "macenko":
            normalizer = MacenkoNormalizer()
            # Nếu có truyền ảnh tham chiếu thì thực hiện fit luôn
            if reference_rgb is not None:
                normalizer.fit(reference_rgb)
            return normalizer

        # Ném ra lỗi ngoại lệ nếu tên phương pháp không nằm trong danh mục hỗ trợ
        raise ValueError(
            f"Unsupported normalization type: '{norm_type}'. "
            f"Available types: ['none', 'reinhard', 'macenko']"
        )
