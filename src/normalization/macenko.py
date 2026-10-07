"""Thuật toán chuẩn hóa phân rã màu nhuộm Macenko dựa trên mật độ quang học và SVD.

Tài liệu tham khảo:
    Macenko et al., "A method for normalizing histology slides for quantitative analysis",
    IEEE ISBI 2009.
"""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Thư viện ghi log hệ thống
import logging
# Các kiểu dữ liệu phụ trợ
from typing import Dict, Optional
# Thư viện tính toán đại số ma trận NumPy
import numpy as np
# Kế thừa từ giao diện BaseStainNormalizer
from src.normalization.base import BaseStainNormalizer

# Khởi tạo logger riêng cho module Macenko
logger = logging.getLogger(__name__)


class MacenkoNormalizer(BaseStainNormalizer):
    """Lớp chuẩn hóa Macenko phân rã ma trận chất nhuộm H&E theo định luật Beer-Lambert."""

    def __init__(
        self,
        target_params: Optional[Dict[str, np.ndarray]] = None,
        od_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        # Ngưỡng mật độ quang học OD để loại bỏ nền kính hiển vi trong suốt
        self.od_threshold = od_threshold
        # Ma trận véc-tơ màu nhuộm tham chiếu (3x2)
        self.ref_stain_matrix: Optional[np.ndarray] = None
        # Nồng độ phân vị thứ 99 tham chiếu của 2 chất nhuộm
        self.ref_q99: Optional[np.ndarray] = None

        # Nếu nạp sẵn tham số tham chiếu, gán trực tiếp và đánh dấu đã fit
        if target_params is not None:
            self.ref_stain_matrix = target_params["stain_matrix"]
            self.ref_q99 = target_params["q99"]
            self._is_fitted = True

    def fit(self, reference_rgb: np.ndarray) -> MacenkoNormalizer:
        """Ước lượng ma trận véc-tơ màu nhuộm và nồng độ phân vị thứ 99 từ ảnh tham chiếu."""
        # Gọi hàm macenko_fit
        params = macenko_fit(reference_rgb, od_threshold=self.od_threshold)
        # Lưu ma trận chất nhuộm chuẩn
        self.ref_stain_matrix = params["stain_matrix"]
        # Lưu nồng độ chuẩn cực đại
        self.ref_q99 = params["q99"]
        # Cập nhật cờ trạng thái
        self._is_fitted = True
        return self

    def transform(self, image_rgb: np.ndarray) -> np.ndarray:
        """Chuẩn hóa một mảnh mô bệnh học về ma trận màu và nồng độ tham chiếu chuẩn."""
        # Kiểm tra trạng thái đã fit
        if not self._is_fitted or self.ref_stain_matrix is None or self.ref_q99 is None:
            raise RuntimeError("MacenkoNormalizer must be fitted before calling transform.")
        # Gọi hàm macenko_apply
        return macenko_apply(
            image_rgb,
            {"stain_matrix": self.ref_stain_matrix, "q99": self.ref_q99},
            od_threshold=self.od_threshold,
        )


def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
  """Ước lượng véc-tơ màu nhuộm Hematoxylin/Eosin và nồng độ phân vị thứ 99 từ ảnh tham chiếu."""
  # Chuyển đổi cường độ điểm ảnh sang mật độ quang học (Optical Density - OD)
  od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
  # Trải phẳng ma trận OD thành mảng (N_pixels, 3)
  flat_od = od.reshape(-1, 3)
  # Lọc bỏ các pixel nền trong suốt có độ dài véc-tơ OD nhỏ hơn ngưỡng
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold
  # Giữ lại các điểm thuộc về cấu trúc tế bào và mô
  flat_od = flat_od[mask]

  # Phân tích suy biến giá trị kỳ dị (SVD) tìm mặt phẳng 2D chứa các véc-tơ chất nhuộm
  _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
  # Chiếu các điểm OD lên mặt phẳng span bởi 2 thành phần kỳ dị đầu tiên
  proj = flat_od @ vh[:2].T
  # Tính góc pha (phi) của từng điểm trong tọa độ cực
  phi = np.arctan2(proj[:, 1], proj[:, 0])

  # Tìm góc cực tiểu tại phân vị 1%
  min_phi = np.percentile(phi, 1.0)
  # Tìm góc cực đại tại phân vị 99%
  max_phi = np.percentile(phi, 99.0)

  # Chiếu ngược góc cực về lại không gian quang học 3 chiều để tìm véc-tơ hấp thụ thứ 1
  v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
  # Chiếu ngược góc cực về lại không gian quang học 3 chiều để tìm véc-tơ hấp thụ thứ 2
  v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

  # Quy ước đảm bảo Hematoxylin nằm ở cột 0 (Hematoxylin hấp thụ ánh sáng đỏ mạnh nhất)
  if v1[0] < v2[0]:
      v1, v2 = v2, v1

  # Ghép hai véc-tơ đơn vị thành ma trận véc-tơ chất nhuộm (3x2)
  stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
  # Dùng giả nghịch đảo để phân rã OD thành nồng độ của từng chất nhuộm
  concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
  # Lấy phân vị thứ 99 của nồng độ mỗi chất nhuộm làm ngưỡng trần
  q99 = np.percentile(concentrations, 99.0, axis=0)

  # Trả về từ điển thông số chất nhuộm tham chiếu
  return {"stain_matrix": stain_matrix, "q99": q99}


def macenko_apply(
    image_rgb: np.ndarray, ref_params: Dict[str, np.ndarray], od_threshold: float = 0.15
) -> np.ndarray:
    """Chuẩn hóa một mảnh mô H&E về véc-tơ màu và nồng độ chất nhuộm của ảnh tham chiếu."""
    # Chuyển ảnh nguồn sang mật độ quang học Beer-Lambert OD
    od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
    # Lấy chiều cao, chiều rộng và số kênh
    h, w, _ = od.shape
    # Trải phẳng ma trận OD
    flat_od = od.reshape(-1, 3)
    # Lọc bỏ pixel nền kính tiêu bản
    mask = np.linalg.norm(flat_od, axis=1) > od_threshold

    # Nếu diện tích mô chiếm dưới 20% khung hình, an toàn trả về ảnh gốc tránh phân rã sai
    if mask.sum() < 0.20 * h * w:
        return image_rgb

    # Gom các điểm quang học thuộc vùng mô
    valid_od = flat_od[mask]
    try:
        # Thực hiện SVD ước lượng véc-tơ màu nhuộm riêng của chính ảnh nguồn
        _, _, vh = np.linalg.svd(valid_od, full_matrices=False)
        # Chiếu lên mặt phẳng 2D
        proj = valid_od @ vh[:2].T
        # Tính góc cực
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        # Góc phân vị 1% và 99%
        min_phi = np.percentile(phi, 1.0)
        max_phi = np.percentile(phi, 99.0)

        # Tái tạo véc-tơ màu nhuộm của tile nguồn
        v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
        # Sắp xếp Hematoxylin trước
        if v1[0] < v2[0]:
            v1, v2 = v2, v1
        # Ma trận chất nhuộm riêng của tile nguồn
        tile_stain = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])

        # Tính toán nồng độ chất nhuộm thực tế của tile nguồn
        tile_conc = flat_od @ np.linalg.pinv(tile_stain).T
        # Tính phân vị thứ 99 của nồng độ trên tile nguồn
        q99 = np.percentile(tile_conc[mask], 99.0, axis=0) + 1e-6

        # Co giãn nồng độ chất nhuộm của tile nguồn cho tương đồng với ảnh tham chiếu chuẩn
        norm_conc = tile_conc * (ref_params["q99"] / q99)
        # Tái tạo lại ma trận OD bằng cách nhân nồng độ chuẩn hóa với ma trận màu của ảnh tham chiếu
        norm_od = norm_conc @ ref_params["stain_matrix"].T
        # Nghịch đảo từ mật độ quang học trở lại không gian RGB
        norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0
        # Giới hạn giá trị trong khoảng [0, 255] định dạng uint8
        return np.clip(norm_rgb.reshape(h, w, 3), 0, 255).astype(np.uint8)
    except Exception as exc:
        # Nếu có lỗi phân tích SVD, ghi log gỡ lỗi và trả về ảnh gốc
        logger.debug("Macenko decomposition exception, returning original tile: %s", exc)
        return image_rgb
