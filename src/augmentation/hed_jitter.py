"""Kỹ thuật tăng cường gây nhiễu ngẫu nhiên nồng độ màu nhuộm sinh học Hematoxylin & Eosin."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module sinh số ngẫu nhiên
import random
# Các kiểu dữ liệu phụ trợ
from typing import Optional
# Thư viện mảng số học NumPy
import numpy as np
# Nhập ma trận hấp thụ H&E chuẩn
from src.config.constants import HE_STAIN_MATRIX


def hed_stain_jitter(
    image_rgb: np.ndarray,
    sigma: float = 0.2,
    bias: float = 0.05,
    stain_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
  """Áp dụng co giãn và dịch chuyển ngẫu nhiên nồng độ trên từng kênh chất nhuộm H&E.

  Tham số:
      image_rgb: Mảng ảnh RGB kích thước (H, W, 3) trong phạm vi [0, 255].
      sigma: Độ lệch chuẩn biến thiên hệ số co giãn (+-sigma xung quanh giá trị 1.0).
      bias: Biên độ dịch chuyển nồng độ (+-bias xung quanh giá trị 0.0).
      stain_matrix: Ma trận véc-tơ hấp thụ màu (3x2) đã chuẩn hóa đơn vị.

  Trả về:
      Mảng ảnh RGB uint8 sau khi đã được gây nhiễu nồng độ.
  """
  # Sử dụng ma trận mặc định nếu không truyền ma trận riêng
  matrix = HE_STAIN_MATRIX if stain_matrix is None else stain_matrix
  # Chuyển đổi từ RGB sang mật độ quang học Beer-Lambert (OD)
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  # Lưu lại kích thước chiều cao và chiều rộng của ảnh
  h, w, _ = od.shape
  # Trải phẳng ma trận quang học thành mảng 2 chiều
  flat_od = od.reshape(-1, 3)

  # Giải ma trận nồng độ c = OD * pinv(matrix)^T
  c = flat_od @ np.linalg.pinv(matrix).T
  # Sinh ngẫu nhiên hệ số co giãn alpha độc lập cho 2 chất nhuộm trong khoảng [1-sigma, 1+sigma]
  alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
  # Sinh ngẫu nhiên độ dịch chuyển beta độc lập cho 2 chất nhuộm trong khoảng [-bias, bias]
  beta = np.random.uniform(-bias, bias, size=2)

  # Biến đổi nồng độ Hematoxylin và chặn dưới không âm
  c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
  # Biến đổi nồng độ Eosin và chặn dưới không âm
  c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)

  # Tái tạo lại mật độ quang học sau khi nồng độ đã bị biến đổi
  od_jittered = c @ matrix.T
  # Nghịch đảo từ mật độ quang học trở lại cường độ điểm ảnh RGB
  rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
  # Cắt giá trị trong đoạn [0, 255] và định hình lại kích thước ảnh gốc uint8
  return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)


class HEDStainJitterTransform:
    """Lớp biến đổi callable bọc hàm hed_stain_jitter với xác suất kích hoạt p."""

    def __init__(self, p: float = 0.8, sigma: float = 0.2, bias: float = 0.05) -> None:
        # Xác suất kích hoạt áp dụng nhiễu nồng độ (mặc định 80%)
        self.p = p
        # Biên độ co giãn nồng độ
        self.sigma = sigma
        # Biên độ dịch chuyển nồng độ
        self.bias = bias

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Thực thi biến đổi ảnh với xác suất p đã cấu hình."""
        # Nếu số ngẫu nhiên nhỏ hơn p, thực hiện gây nhiễu
        if random.random() < self.p:
            return hed_stain_jitter(image_rgb, sigma=self.sigma, bias=self.bias)
        # Ngược lại giữ nguyên ảnh
        return image_rgb
