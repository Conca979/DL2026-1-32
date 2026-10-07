"""Chuẩn hóa màu thống kê Reinhard trong không gian màu Ruderman LAB.

Tài liệu tham khảo:
    Reinhard et al., "Color Transfer between Images", IEEE CGA 2001.
    Ruderman et al., "Statistics of cone responses to natural images", JOSA A 1998.
"""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Các kiểu dữ liệu phụ trợ
from typing import Dict, Optional
# Thư viện mảng số học đa chiều NumPy
import numpy as np
# Kế thừa từ giao diện chuẩn BaseStainNormalizer
from src.normalization.base import BaseStainNormalizer

# Ma trận chuyển đổi từ RGB sang không gian nón cảm thụ thị giác LMS (Ruderman et al., 1998)
_LMS_MAT: np.ndarray = np.array([
    [0.3811, 0.5783, 0.0402],  # Hệ số cho kênh bước sóng dài (Long wavelength)
    [0.1967, 0.7244, 0.0782],  # Hệ số cho kênh bước sóng trung bình (Medium wavelength)
    [0.0241, 0.1288, 0.8444],  # Hệ số cho kênh bước sóng ngắn (Short wavelength)
], dtype=np.float64)

# Ma trận trực giao chuyển đổi từ log(LMS) sang không gian màu đối kháng Ruderman LAB
_LAB_MAT: np.ndarray = np.array([
    [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],  # Kênh L: Độ sáng phi sắc (Achromatic)
    [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],  # Kênh alpha: Đối kháng Vàng - Lam
    [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],                 # Kênh beta: Đối kháng Đỏ - Lục
], dtype=np.float64)

# Ma trận nghịch đảo khôi phục log(LMS) từ Ruderman LAB
_INV_LAB_MAT: np.ndarray = np.array([
    [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0),  1.0 / np.sqrt(2.0)],  # Khôi phục log(L)
    [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],  # Khôi phục log(M)
    [1.0 / np.sqrt(3.0), -2.0 / np.sqrt(6.0),  0.0],                 # Khôi phục log(S)
], dtype=np.float64)

# Ma trận nghịch đảo LMS để chuyển đổi ngược từ LMS về RGB
_INV_LMS_MAT: np.ndarray = np.linalg.inv(_LMS_MAT)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Chuyển đổi hình ảnh từ không gian màu RGB sang không gian đối kháng Ruderman LAB."""
    # Chuẩn hóa giá trị pixel về [1e-4, 1.0] tránh chia cho 0 hoặc lỗi logarit
    norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
    # Nhân ma trận chuyển sang không gian nón LMS
    lms = norm_rgb @ _LMS_MAT.T
    # Lấy log10 nồng độ kích thích tế bào nón
    log_lms = np.log10(np.clip(lms, 1e-4, None))
    # Chiếu log(LMS) sang các trục đối kháng LAB không tương quan
    return log_lms @ _LAB_MAT.T


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Chuyển đổi ngược mảng hình ảnh từ Ruderman LAB về lại không gian RGB (uint8)."""
    # Chiếu ngược từ LAB về log(LMS)
    log_lms = lab @ _INV_LAB_MAT.T
    # Đảo ngược logarit bằng hàm lũy thừa cơ số 10
    lms = 10.0 ** log_lms
    # Nhân ma trận nghịch đảo để thu được các kênh RGB
    rgb = lms @ _INV_LMS_MAT.T
    # Giới hạn cường độ trong khoảng [0, 255] và trả về mảng số nguyên không dấu uint8
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


class ReinhardNormalizer(BaseStainNormalizer):
    """Lớp chuẩn hóa Reinhard căn chỉnh phân phối trung bình và độ lệch chuẩn trong LAB."""

    def __init__(self, target_stats: Optional[Dict[str, np.ndarray]] = None) -> None:
        super().__init__()
        # Kỳ vọng trung bình của ảnh tham chiếu trên 3 kênh [L, alpha, beta]
        self.target_mean: Optional[np.ndarray] = None
        # Độ lệch chuẩn của ảnh tham chiếu trên 3 kênh [L, alpha, beta]
        self.target_std: Optional[np.ndarray] = None

        # Nếu truyền sẵn các chỉ số thống kê thì nạp trực tiếp và đánh dấu đã fit
        if target_stats is not None:
            self.target_mean = target_stats["mean"]
            self.target_std = target_stats["std"]
            self._is_fitted = True

    def fit(self, reference_rgb: np.ndarray) -> ReinhardNormalizer:
        """Ước lượng trung bình và độ lệch chuẩn từ ảnh tham chiếu trong không gian LAB."""
        # Gọi hàm tính toán thống kê
        stats = reinhard_fit(reference_rgb)
        # Lưu các giá trị trung bình đích
        self.target_mean = stats["mean"]
        # Lưu các giá trị độ lệch chuẩn đích
        self.target_std = stats["std"]
        # Đánh dấu trạng thái đã sẵn sàng chuyển giao màu
        self._is_fitted = True
        return self

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Áp dụng chuyển giao màu thống kê Reinhard lên ảnh nguồn."""
        # Kiểm tra điều kiện tiên quyết: phải fit trước khi transform
        if not self._is_fitted or self.target_mean is None or self.target_std is None:
            raise RuntimeError("ReinhardNormalizer must be fitted before calling transform.")
        # Gọi hàm thực hiện căn chỉnh phân phối
        return reinhard_apply(
            image_rgb,
            {"mean": self.target_mean, "std": self.target_std},
        )


def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Trích xuất trung bình và độ lệch chuẩn trên từng kênh trong không gian Reinhard LAB."""
    # Chuyển ảnh sang không gian Ruderman LAB
    lab = rgb_to_lab(reference_rgb)
    # Trả về từ điển thống kê trung bình và độ lệch chuẩn kèm hằng số ổn định số học
    return {
        "mean": np.mean(lab, axis=(0, 1)),
        "std": np.std(lab, axis=(0, 1)) + 1e-6,
    }


def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Áp dụng phép chuyển giao màu thống kê Reinhard (sử dụng thuần NumPy)."""
    # Chuyển ảnh nguồn sang không gian LAB
    lab = rgb_to_lab(image_rgb)
    # Tính trung bình hiện tại của ảnh nguồn
    mean = np.mean(lab, axis=(0, 1))
    # Tính độ lệch chuẩn hiện tại của ảnh nguồn
    std = np.std(lab, axis=(0, 1)) + 1e-6

    # Khởi tạo ma trận chứa ảnh LAB đã chuẩn hóa
    norm_lab = np.zeros_like(lab)
    # Chuẩn hóa chuẩn Z-score và co giãn theo phân phối của ảnh tham chiếu đích
    for c in range(3):
        norm_lab[:, :, c] = (
            ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]
        )

    # Chuyển ngược lại về không gian màu RGB
    return lab_to_rgb(norm_lab)
