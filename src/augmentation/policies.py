"""Định nghĩa kiểu liệt kê (Enum) các chính sách tăng cường dữ liệu cho các mảnh mô bệnh học."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Lớp Enum cơ sở của Python
from enum import Enum


class AugmentationPolicy(str, Enum):
    """Tập hợp các chính sách tăng cường dữ liệu được hỗ trợ trong nghiên cứu ablation."""

    # Không áp dụng tăng cường dữ liệu nào
    NONE = "none"
    # Tăng cường biến đổi không gian hình học (Lật ngang, lật dọc, xoay 90 độ)
    AUG_GEO = "aug_geo"
    # Tăng cường nhiễu nồng độ màu nhuộm sinh học (HED Stain Jitter)
    AUG_STAIN = "aug_stain"
    # Kết hợp đồng thời cả biến đổi không gian và nhiễu nồng độ màu
    AUG_COMBINED = "aug_combined"

    @classmethod
    def from_str(cls, value: str) -> AugmentationPolicy:
        """Chuyển đổi chuỗi ký tự thành đối tượng AugmentationPolicy an toàn."""
        # Chuẩn hóa chuỗi văn bản đầu vào
        val = (value or "none").strip().lower()
        # Duyệt qua các thành phần của Enum
        for item in cls:
            # Nếu trùng khớp giá trị chuỗi, trả về mục Enum tương ứng
            if item.value == val:
                return item
        # Nếu không hợp lệ, ném ra ngoại lệ thông báo các lựa chọn khả dĩ
        raise ValueError(
            f"Unknown augmentation policy: '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )
