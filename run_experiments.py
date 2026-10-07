# Cho phép hỗ trợ type hinting hiện đại của Python (hỗ trợ annotation dạng hoãn nạp)
from __future__ import annotations

# Thư viện phân tích tham số dòng lệnh CLI tiêu chuẩn
import argparse
# Thư viện sao chép đối tượng trong bộ nhớ (dùng để deepcopy trạng thái mô hình PyTorch)
import copy
# Thư viện tương tác với hệ điều hành (đọc tên hệ điều hành, biến môi trường, đường dẫn)
import os
# Thư viện sinh số ngẫu nhiên ngẫu nhiên cho việc xáo trộn và chọn chính sách augment
import random
# Thư viện đo lường thời gian thực thi của các epoch huấn luyện
import time
# Thư viện thao tác với đường dẫn tập tin theo phong cách hướng đối tượng hiện đại
from pathlib import Path
# Các kiểu dữ liệu hỗ trợ ép kiểu tĩnh (Type Annotations)
from typing import Any, Dict, List, Tuple

# Thư viện tính toán đại số tuyến tính và xử lý ma trận mảng số đa chiều
import numpy as np
# Thư viện xử lý dữ liệu dạng bảng (DataFrames) cho các tập split phân tầng
import pandas as pd
# Thư viện Pillow dùng để đọc, giải mã và lưu trữ hình ảnh định dạng RGB
from PIL import Image

# Danh sách 9 lớp mô học bệnh học đại trực tràng chuẩn trong bộ dữ liệu NCT-CRC-HE
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
# Ánh xạ từ tên lớp (chuỗi ký tự) sang chỉ số số nguyên (0 đến 8) để huấn luyện mô hình
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
# Tổng số lượng phân lớp mô học cần phân loại (ở đây là 9 lớp)
NUM_CLASSES = len(CLASSES)

# Giá trị trung bình (Mean) của bộ dữ liệu ImageNet cho 3 kênh màu [Red, Green, Blue]
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # Chuẩn hóa chuẩn của PyTorch Vision
# Độ lệch chuẩn (Std) của bộ dữ liệu ImageNet cho 3 kênh màu [Red, Green, Blue]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Định nghĩa ma trận thực nghiệm 13 cell ablation study đo lường độ bền vững (robustness)
EXPERIMENTS: List[Dict[str, str]] = [
  # Stage 0: Thí nghiệm đối chứng gốc (Anchor Baseline) - không áp dụng bất kỳ phòng vệ nào
  {"id": "EXP-01", "stage": "Stage 0", "backbone": "resnet50",      "norm": "none",     "aug": "none",        "notes": "Raw baseline"},
  # Stage 1: Kiểm tra riêng lẻ tác động của chuẩn hóa màu nhuộm (Color Normalization)
  {"id": "EXP-02", "stage": "Stage 1", "backbone": "resnet50",      "norm": "reinhard", "aug": "none",        "notes": "Statistical LAB transfer"},
  {"id": "EXP-03", "stage": "Stage 1", "backbone": "resnet50",      "norm": "macenko",  "aug": "none",        "notes": "Optical density deconvolution"},
  # Stage 2: Kiểm tra riêng lẻ chính sách tăng cường dữ liệu (Augmentation Policy)
  {"id": "EXP-04", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_geo",     "notes": "Spatial invariance (flips/rot)"},
  {"id": "EXP-05", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_stain",   "notes": "HED stain jitter"},
  {"id": "EXP-06", "stage": "Stage 2", "backbone": "resnet50",      "norm": "none",     "aug": "aug_combined","notes": "Spatial + HED jitter"},
  # Stage 3: Tương tác kết hợp giữa Chuẩn hóa màu và Tăng cường dữ liệu (Normalization x Augmentation)
  {"id": "EXP-07", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_geo",     "notes": "Macenko + spatial"},
  {"id": "EXP-08", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_stain",   "notes": "Macenko + stain jitter"},
  {"id": "EXP-09", "stage": "Stage 3", "backbone": "resnet50",      "norm": "macenko",  "aug": "aug_combined","notes": "Full defense ResNet"},
  # Stage 4: Đánh giá khả năng tổng quát hóa của các Backbone thị giác hiện đại và Foundation Model
  {"id": "EXP-10", "stage": "Stage 4", "backbone": "convnext_tiny", "norm": "none",     "aug": "none",        "notes": "ConvNeXt-Tiny raw"},
  {"id": "EXP-11", "stage": "Stage 4", "backbone": "convnext_tiny", "norm": "macenko",  "aug": "aug_combined","notes": "ConvNeXt-Tiny defended"},
  {"id": "EXP-12", "stage": "Stage 4", "backbone": "phikon",        "norm": "none",     "aug": "none",        "notes": "Phikon (TCGA SSL ViT) raw"},
  {"id": "EXP-13", "stage": "Stage 4", "backbone": "phikon",        "norm": "macenko",  "aug": "aug_combined","notes": "Phikon defended"},
]


# ==============================================================================
# Thuật toán Chuẩn hóa Màu nhuộm: Reinhard (2001) & Macenko (2009)
# ==============================================================================

# Ma trận biến đổi từ không gian màu RGB sang không gian nón cảm thụ võng mạc LMS (Ruderman et al., 1998)
_LMS_MAT = np.array([
  [0.3811, 0.5783, 0.0402],  # Hệ số chuyển đổi cho kênh L (Long wavelength)
  [0.1967, 0.7244, 0.0782],  # Hệ số chuyển đổi cho kênh M (Medium wavelength)
  [0.0241, 0.1288, 0.8444],  # Hệ số chuyển đổi cho kênh S (Short wavelength)
], dtype=np.float64)

# Ma trận trực giao chuyển đổi từ không gian log(LMS) sang không gian đối kháng Ruderman LAB
_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],  # Kênh L: Độ sáng achromatic
  [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],  # Kênh alpha: Đối kháng vàng - xanh dương
  [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],                 # Kênh beta: Đối kháng đỏ - xanh lá
], dtype=np.float64)

# Ma trận nghịch đảo của _LAB_MAT để biến đổi ngược từ Ruderman LAB về log(LMS)
_INV_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0),  1.0 / np.sqrt(2.0)],  # Cột 1 khôi phục log(L)
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],  # Cột 2 khôi phục log(M)
  [1.0 / np.sqrt(3.0), -2.0 / np.sqrt(6.0),  0.0],                 # Cột 3 khôi phục log(S)
], dtype=np.float64)

# Ma trận nghịch đảo đại số tuyến tính của ma trận LMS để chuyển ngược từ LMS về RGB
_INV_LMS_MAT = np.linalg.inv(_LMS_MAT)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
  """Chuyển đổi hình ảnh từ không gian màu RGB sang không gian màu Ruderman LAB."""
  # Chuẩn hóa giá trị pixel về khoảng [1e-4, 1.0] tránh lỗi số học khi lấy logarit
  norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
  # Nhân ma trận nhân chuyển đổi từ không gian RGB sang không gian võng mạc LMS
  lms = norm_rgb @ _LMS_MAT.T
  # Lấy logarit cơ số 10 của các giá trị nón cảm thụ LMS
  log_lms = np.log10(np.clip(lms, 1e-4, None))
  # Nhân ma trận chuyển đổi từ log(LMS) sang không gian màu đối kháng LAB
  return log_lms @ _LAB_MAT.T


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
  """Chuyển đổi ngược hình ảnh từ không gian Ruderman LAB trở về không gian màu chuẩn RGB."""
  # Nhân ma trận nghịch đảo để khôi phục lại log(LMS)
  log_lms = lab @ _INV_LAB_MAT.T
  # Lũy thừa 10 để đảo ngược phép tính log10, khôi phục nồng độ LMS tuyến tính
  lms = 10.0 ** log_lms
  # Nhân ma trận nghịch đảo LMS để tái tạo lại 3 kênh màu RGB
  rgb = lms @ _INV_LMS_MAT.T
  # Cắt giá trị trong khoảng hợp lệ [0, 255] và ép kiểu về số nguyên không dấu uint8
  return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
  """Tính toán kỳ vọng trung bình (mean) và độ lệch chuẩn (std) của ảnh mẫu tham chiếu trong LAB."""
  # Đưa ảnh tham chiếu về không gian màu Ruderman LAB
  lab = _rgb_to_lab(reference_rgb)
  # Trả về từ điển chứa giá trị thống kê trung bình và độ lệch chuẩn theo từng kênh
  return {
    "mean": np.mean(lab, axis=(0, 1)),       # Trung bình dọc theo chiều cao và rộng của ảnh
    "std": np.std(lab, axis=(0, 1)) + 1e-6,  # Độ lệch chuẩn cộng thêm epsilon nhỏ chống chia cho 0
  }


def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
  """Áp dụng phép chuyển giao màu thống kê Reinhard giữa ảnh nguồn và thông số ảnh tham chiếu."""
  # Đưa ảnh cần chuẩn hóa về không gian Ruderman LAB
  lab = _rgb_to_lab(image_rgb)
  # Tính toán trung bình hiện tại của ảnh cần chuẩn hóa
  mean = np.mean(lab, axis=(0, 1))
  # Tính toán độ lệch chuẩn hiện tại của ảnh cần chuẩn hóa
  std = np.std(lab, axis=(0, 1)) + 1e-6

  # Khởi tạo ma trận rỗng có cùng kích thước để chứa kết quả chuẩn hóa
  norm_lab = np.zeros_like(lab)
  # Căn chỉnh phân phối Gaussian từng kênh: trừ mean nguồn, chia std nguồn, nhân std đích, cộng mean đích
  for c in range(3):
    norm_lab[:, :, c] = ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]

  # Biến đổi ngược lại ảnh đã căn chỉnh phân phối màu về không gian RGB uint8
  return _lab_to_rgb(norm_lab)


def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
  """Ước lượng véc-tơ màu nhuộm Hematoxylin/Eosin và nồng độ phân vị thứ 99 từ ảnh tham chiếu."""
  # Chuyển đổi giá trị RGB sang Mật độ quang học (Optical Density - OD) theo định luật Beer-Lambert
  od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
  # Trải phẳng ma trận OD thành mảng 2 chiều (số pixel, 3 kênh màu)
  flat_od = od.reshape(-1, 3)
  # Tạo mặt nạ lọc bỏ các pixel nền trắng trong suốt có độ hấp thụ ánh sáng nhỏ hơn ngưỡng
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold
  # Giữ lại các pixel thực sự chứa mô sinh học
  flat_od = flat_od[mask]

  # Phân tích suy biến giá trị đơn kỳ SVD để tìm mặt phẳng 2 chiều chứa véc-tơ hấp thụ màu H&E
  _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
  # Chiếu các điểm OD lên mặt phẳng span bởi 2 thành phần kỳ dị đầu tiên
  proj = flat_od @ vh[:2].T
  # Tính toán góc cực (angle) của từng pixel trên mặt phẳng 2 chiều
  phi = np.arctan2(proj[:, 1], proj[:, 0])

  # Tìm phân vị góc thứ 1% đại diện cho biên véc-tơ màu nhuộm thứ nhất
  min_phi = np.percentile(phi, 1.0)
  # Tìm phân vị góc thứ 99% đại diện cho biên véc-tơ màu nhuộm thứ hai
  max_phi = np.percentile(phi, 99.0)

  # Chiếu ngược góc cực về lại không gian quang học OD để lấy véc-tơ hấp thụ 1
  v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
  # Chiếu ngược góc cực về lại không gian quang học OD để lấy véc-tơ hấp thụ 2
  v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

  # Đảm bảo Hematoxylin luôn nằm ở cột đầu tiên (Hematoxylin hấp thụ ánh sáng đỏ mạnh hơn Eosin)
  if v1[0] < v2[0]:
    v1, v2 = v2, v1

  # Ghép hai véc-tơ đơn vị đã chuẩn hóa độ dài thành ma trận véc-tơ màu nhuộm kích thước 3x2
  stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
  # Sử dụng giả nghịch đảo Moore-Penrose để phân rã ma trận OD thành ma trận nồng độ màu (Concentrations)
  concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
  # Lấy phân vị thứ 99 của nồng độ mỗi chất nhuộm làm giá trị cực đại tham chiếu
  q99 = np.percentile(concentrations, 99.0, axis=0)

  # Trả về ma trận chất nhuộm và nồng độ tham chiếu chuẩn
  return {"stain_matrix": stain_matrix, "q99": q99}


def macenko_apply(image_rgb: np.ndarray, ref_params: Dict[str, np.ndarray], od_threshold: float = 0.15) -> np.ndarray:
  """Chuẩn hóa một mảnh mô H&E về véc-tơ màu và nồng độ chất nhuộm của ảnh mẫu tham chiếu chuẩn."""
  # Tính toán mật độ quang học OD cho ảnh đầu vào
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  # Lấy kích thước chiều cao, chiều rộng và số kênh của ảnh
  h, w, _ = od.shape
  # Trải phẳng ma trận OD phục vụ tính toán đại số
  flat_od = od.reshape(-1, 3)
  # Lọc bỏ pixel nền kính tiêu bản
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold

  # Nếu diện tích mô hữu hiệu quá ít (< 20% khung hình), giữ nguyên ảnh gốc tránh phân rã sai lệch
  if mask.sum() < 0.20 * h * w:
    return image_rgb

  # Lấy toàn bộ các điểm quang học thuộc vùng mô
  valid_od = flat_od[mask]
  try:
    # Thực hiện SVD phân rã tìm ma trận véc-tơ màu nhuộm riêng của chính ảnh hiện tại
    _, _, vh = np.linalg.svd(valid_od, full_matrices=False)
    # Chiếu OD lên không gian 2D
    proj = valid_od @ vh[:2].T
    # Tính góc cực của các điểm OD
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    # Xác định góc phân vị 1% và 99%
    min_phi = np.percentile(phi, 1.0)
    max_phi = np.percentile(phi, 99.0)

    # Tái tạo véc-tơ nhuộm của ảnh hiện tại
    v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
    # Sắp xếp đúng thứ tự Hematoxylin trước
    if v1[0] < v2[0]:
      v1, v2 = v2, v1
    # Ma trận màu nhuộm riêng của tile ảnh hiện tại
    tile_stain = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])

    # Tính toán nồng độ thực tế của từng chất nhuộm trên tile ảnh hiện tại
    tile_conc = flat_od @ np.linalg.pinv(tile_stain).T
    # Tính phân vị thứ 99 của nồng độ hiện tại
    q99 = np.percentile(tile_conc[mask], 99.0, axis=0) + 1e-6

    # Chuẩn hóa co giãn nồng độ chất nhuộm của tile khớp với nồng độ của ảnh tham chiếu chuẩn
    norm_conc = tile_conc * (ref_params["q99"] / q99)
    # Tái tạo lại ma trận mật độ quang học bằng cách nhân nồng độ chuẩn hóa với ma trận màu của ảnh tham chiếu
    norm_od = norm_conc @ ref_params["stain_matrix"].T
    # Nghịch đảo định luật Beer-Lambert để chuyển đổi từ OD trở lại không gian RGB
    norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0
    # Giới hạn giá trị [0, 255] và định hình lại kích thước ảnh gốc ban đầu
    return np.clip(norm_rgb.reshape(h, w, 3), 0, 255).astype(np.uint8)
  except Exception:
    # Nếu SVD không hội tụ hoặc gặp lỗi ma trận kỳ dị, an toàn trả về ảnh gốc
    return image_rgb


# ==============================================================================
# Tăng cường dữ liệu (Data Augmentations): Biến đổi không gian & Nhiễu HED
# ==============================================================================

# Ma trận hấp thụ màu H&E chuẩn thực nghiệm dùng cho kỹ thuật gây nhiễu nồng độ HED Jitter
HE_STAIN_MATRIX = np.array([
  [0.650, 0.072],  # Hệ số hấp thụ kênh R của Hematoxylin và Eosin
  [0.704, 0.990],  # Hệ số hấp thụ kênh G của Hematoxylin và Eosin
  [0.286, 0.105],  # Hệ số hấp thụ kênh B của Hematoxylin và Eosin
], dtype=np.float64)
# Chuẩn hóa chuẩn Euclide cho từng cột véc-tơ màu nhuộm
HE_STAIN_MATRIX /= np.linalg.norm(HE_STAIN_MATRIX, axis=0, keepdims=True)


def hed_stain_jitter(image_rgb: np.ndarray, sigma: float = 0.2, bias: float = 0.05) -> np.ndarray:
  """Áp dụng kỹ thuật gây nhiễu ngẫu nhiên nồng độ màu nhuộm sinh học (H&E Stain Jitter)."""
  # Chuyển đổi ảnh RGB sang mật độ quang học Beer-Lambert OD
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  # Lưu lại kích thước chiều cao và chiều rộng của ảnh
  h, w, _ = od.shape
  # Trải phẳng ma trận quang học OD
  flat_od = od.reshape(-1, 3)

  # Giải ma trận nồng độ H và E bằng giả nghịch đảo Moore-Penrose
  c = flat_od @ np.linalg.pinv(HE_STAIN_MATRIX).T
  # Lấy mẫu ngẫu nhiên hệ số nhân co giãn alpha độc lập cho từng chất nhuộm trong khoảng [1-sigma, 1+sigma]
  alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
  # Lấy mẫu ngẫu nhiên độ dịch chuyển cộng bias độc lập cho từng chất nhuộm trong khoảng [-bias, bias]
  beta = np.random.uniform(-bias, bias, size=2)

  # Biến đổi nồng độ Hematoxylin: co giãn theo alpha và dịch chuyển theo beta, chặn dưới không âm
  c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
  # Biến đổi nồng độ Eosin: co giãn theo alpha và dịch chuyển theo beta, chặn dưới không âm
  c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)

  # Tái tổng hợp lại mật độ quang học OD sau khi nồng độ đã bị gây nhiễu
  od_jittered = c @ HE_STAIN_MATRIX.T
  # Nghịch đảo từ mật độ quang học về lại không gian cường độ điểm ảnh RGB
  rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
  # Ép giá trị điểm ảnh về phạm vi hợp lệ [0, 255] định dạng uint8
  return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)


class PatchTransform:
  """Pipeline tiền xử lý thống nhất: Chuẩn hóa màu, Tăng cường dữ liệu, Chuyển đổi Tensor."""

  def __init__(self, policy: str = "none", norm_fn: Any = None, is_train: bool = True):
    # Lưu lại tên chính sách tăng cường (none, aug_geo, aug_stain, aug_combined)
    self.policy = policy
    # Lưu hàm chuẩn hóa màu nhuộm được truyền vào (Reinhard, Macenko hoặc None)
    self.norm_fn = norm_fn
    # Cờ xác định xem đây là tập huấn luyện (True) hay tập kiểm thử/đánh giá (False)
    self.is_train = is_train

  def __call__(self, img_pil: Image.Image) -> Any:
    # Nhập thư viện hàm chức năng chuyển đổi của Torchvision
    import torchvision.transforms.functional as TF

    # Chuyển đổi đối tượng ảnh Pillow sang mảng NumPy định dạng RGB
    arr = np.array(img_pil.convert("RGB"))
    # Nếu có cấu hình hàm chuẩn hóa màu, áp dụng chuẩn hóa trước tiên
    if self.norm_fn is not None:
      arr = self.norm_fn(arr)

    # Nếu đang huấn luyện và chính sách yêu cầu nhiễu màu, áp dụng HED Jitter với xác suất 80%
    if self.is_train and self.policy in ("aug_stain", "aug_combined"):
      if random.random() > 0.2:
        arr = hed_stain_jitter(arr)

    # Chuyển mảng NumPy [0, 255] sang PyTorch Tensor dạng số thực float trong khoảng [0.0, 1.0]
    tensor = TF.to_tensor(arr)

    # Nếu đang huấn luyện và chính sách yêu cầu biến đổi không gian hình học
    if self.is_train and self.policy in ("aug_geo", "aug_combined"):
      # Lật ảnh ngẫu nhiên theo chiều ngang với xác suất 50%
      if random.random() > 0.5:
        tensor = TF.hflip(tensor)
      # Lật ảnh ngẫu nhiên theo chiều dọc với xác suất 50%
      if random.random() > 0.5:
        tensor = TF.vflip(tensor)
      # Chọn ngẫu nhiên góc xoay trực giao: 0, 90, 180 hoặc 270 độ
      rot = random.choice([0, 90, 180, 270])
      # Áp dụng phép xoay nếu góc xoay lớn hơn 0 độ
      if rot > 0:
        tensor = TF.rotate(tensor, rot)

    # Chuẩn hóa chuẩn hóa Z-score theo thông số kênh màu của ImageNet
    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ==============================================================================
# Lớp Dataset và Chuẩn bị Phân chia dữ liệu (Splits)
# ==============================================================================

class HistologyDataset:
  """Lớp đóng gói Dataset PyTorch nạp các mảnh mô học từ danh sách đường dẫn DataFrame."""
  def __init__(self, df: pd.DataFrame, transform: PatchTransform):
    # Đặt lại chỉ mục cho DataFrame để truy cập chỉ số tuần tự an toàn
    self.df = df.reset_index(drop=True)
    # Lưu hàm biến đổi PatchTransform tương ứng
    self.transform = transform

  def __len__(self) -> int:
    # Trả về tổng số lượng mẫu ảnh có trong tập dữ liệu
    return len(self.df)

  def __getitem__(self, idx: int):
    # Lấy thông tin dòng dữ liệu tại chỉ số idx
    row = self.df.iloc[idx]
    # Nạp ảnh từ đường dẫn lưu trên ổ đĩa bằng thư viện Pillow
    img = Image.open(row["image_path"])
    # Áp dụng hàm biến đổi transform lên ảnh và trả về cặp (Tensor ảnh, Nhãn lớp)
    return self.transform(img), int(row["label_idx"])


# Tập hợp các đuôi tệp tin hình ảnh được hệ thống hỗ trợ quét
SUPPORTED_IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def find_class_images(base_dir: Path, class_name: str) -> List[Path]:
  """Tìm kiếm tất cả ảnh thuộc một lớp mô học, hỗ trợ cả chữ hoa/thường và thư mục lồng nhau."""
  # Nếu thư mục gốc không tồn tại hoặc không phải là thư mục, trả về danh sách rỗng
  if not base_dir.exists() or not base_dir.is_dir():
    return []

  # Biến lưu trữ thư mục con của lớp mô học
  c_folder = None
  # Bước 1: Kiểm tra khớp trực tiếp thư mục con theo tên lớp
  if (base_dir / class_name).is_dir():
    c_folder = base_dir / class_name
  else:
    # Quét các thư mục con cấp 1 không phân biệt chữ hoa hay chữ thường
    for child in base_dir.iterdir():
      if child.is_dir() and child.name.upper() == class_name.upper():
        c_folder = child
        break

  # Bước 2: Nếu chưa tìm thấy, quét đệ quy qua toàn bộ các thư mục lồng sâu bên trong
  if c_folder is None:
    for child in base_dir.rglob("*"):
      if child.is_dir() and child.name.upper() == class_name.upper():
        c_folder = child
        break

  # Nếu vẫn không tìm thấy thư mục của lớp, trả về danh sách rỗng
  if c_folder is None or not c_folder.is_dir():
    return []

  # Lấy tất cả tệp ảnh khớp với định dạng hỗ trợ nằm trong thư mục lớp
  images = [
    p for p in c_folder.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
  ]
  # Nếu thư mục lớp lại chứa các thư mục con khác, quét đệ quy rglob tìm toàn bộ ảnh
  if not images:
    images = [
      p for p in c_folder.rglob("*")
      if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
  # Sắp xếp danh sách đường dẫn để đảm bảo tính tái lập tuyệt đối
  return sorted(images)


def prepare_dataset_splits(
  source_dir: Path, target_dir: Path, out_dir: Path, subset_size: int = 25000, seed: int = 100
) -> Tuple[Dict[str, pd.DataFrame], Path]:
  """Tạo tập phân tầng 70/15/15 cho miền nguồn và trích xuất tile tham chiếu nhuộm chuẩn."""
  # Tạo thư mục đầu ra nếu chưa tồn tại
  out_dir.mkdir(parents=True, exist_ok=True)
  # Thiết lập seed cho trình sinh số ngẫu nhiên của NumPy đảm bảo phân chia dữ liệu tái lập được
  np.random.seed(seed)
  # Thiết lập seed cho module random chuẩn của Python
  random.seed(seed)

  # Danh sách gom các bản ghi mẫu ảnh thuộc miền nguồn (In-Domain Source)
  source_records = []
  # Duyệt qua từng lớp mô học trong danh sách 9 lớp chuẩn
  for c_idx, c_name in enumerate(CLASSES):
    # Tìm kiếm toàn bộ ảnh của lớp trong thư mục miền nguồn
    c_images = find_class_images(source_dir, c_name)
    # Thêm từng ảnh vào danh sách bản ghi kèm tên lớp, chỉ số nhãn và tên miền
    for p in c_images:
      source_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "source"})

  # Khởi tạo bảng dữ liệu Pandas DataFrame từ danh sách bản ghi nguồn
  source_df = pd.DataFrame(source_records)
  # Kiểm tra tính hợp lệ: nếu không tìm thấy ảnh nào, ném ra ngoại lệ thông báo chi tiết
  if len(source_df) == 0:
    items = [p.name for p in list(source_dir.iterdir())[:15]] if source_dir.is_dir() else "directory does not exist"
    raise RuntimeError(
      f"No source images found under {source_dir}.\n"
      f"Contents found at {source_dir}: {items}.\n"
      f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
    )

  # Nếu có yêu cầu lấy tập con (subset_size > 0), thực hiện lấy mẫu phân tầng đồng đều mỗi lớp
  if 0 < subset_size < len(source_df):
    per_class = subset_size // NUM_CLASSES
    source_df = source_df.groupby("class_name", group_keys=False).apply(
      lambda g: g.sample(min(len(g), per_class), random_state=seed)
    ).reset_index(drop=True)

  # Nhập hàm chia tập từ thư viện scikit-learn
  from sklearn.model_selection import train_test_split
  # Chia tập nguồn thành 70% huấn luyện (train) và 30% phần còn lại (rest), phân tầng theo nhãn lớp
  train_df, rest_df = train_test_split(source_df, test_size=0.30, stratify=source_df["class_name"], random_state=seed)
  # Chia 30% còn lại thành 15% validation nội miền (val_id) và 15% kiểm thử nội miền (test_id)
  val_df, test_id_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["class_name"], random_state=seed)

  # Danh sách gom các bản ghi mẫu ảnh thuộc miền đích ngoại viện (Out-of-Domain Target Cohort)
  target_records = []
  # Duyệt qua từng lớp mô học trong thư mục miền đích
  for c_idx, c_name in enumerate(CLASSES):
    c_images = find_class_images(target_dir, c_name)
    for p in c_images:
      target_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "target"})

  # Khởi tạo bảng dữ liệu Pandas DataFrame cho tập kiểm thử ngoại miền OOD
  test_ood_df = pd.DataFrame(target_records)
  # Kiểm tra tính hợp lệ của thư mục miền đích ngoại viện
  if len(test_ood_df) == 0:
    items = [p.name for p in list(target_dir.iterdir())[:15]] if target_dir.is_dir() else "directory does not exist"
    raise RuntimeError(
      f"No target images found under {target_dir}.\n"
      f"Contents found at {target_dir}: {items}.\n"
      f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
    )

  # Đóng gói 4 phân tập dữ liệu thành từ điển
  splits = {"train": train_df, "val_id": val_df, "test_id": test_id_df, "test_ood": test_ood_df}
  # Lưu từng phân tập ra tệp CSV riêng biệt trong thư mục kết quả để tái lập
  for name, df in splits.items():
    df.to_csv(out_dir / f"{name}.csv", index=False)

  # Chọn tile ảnh tham chiếu chuẩn nghiêm ngặt từ tập train thuộc lớp biểu mô u TUM
  tum_train = train_df[train_df["class_name"] == "TUM"]
  # Lấy đường dẫn của mẫu biểu mô u đầu tiên trong tập train
  ref_path = Path(tum_train.iloc[0]["image_path"])
  # Mở ảnh tham chiếu và chuyển về hệ màu RGB chuẩn
  ref_img = Image.open(ref_path).convert("RGB")
  # Đường dẫn lưu trữ ảnh tham chiếu chuẩn hóa màu
  ref_save_path = out_dir / "reference_stain.png"
  # Lưu ảnh tham chiếu ra tệp PNG
  ref_img.save(ref_save_path)

  # In nhật ký số lượng mẫu của từng phân tập dữ liệu
  print(f"[splits] Train={len(train_df)} | Val={len(val_df)} | Test-ID={len(test_id_df)} | Test-OOD={len(test_ood_df)}")
  # In tên tệp tile ảnh tham chiếu chuẩn được chọn
  print(f"[reference] Canonical stain reference: {ref_path.name}")
  # Trả về từ điển các phân tập và đường dẫn ảnh tham chiếu
  return splits, ref_save_path


# ==============================================================================
# Kiến trúc Mô hình (Model Architectures): ResNet-50, ConvNeXt-Tiny, Phikon
# ==============================================================================

def build_model(backbone_name: str, num_classes: int = NUM_CLASSES):
  """Khởi tạo kiến trúc mô hình học sâu tương ứng theo tên backbone chỉ định."""
  # Thư viện PyTorch Image Models (timm) cung cấp các backbone thị giác tiên tiến
  import timm
  # Module mạng nơ-ron cơ bản của PyTorch
  import torch.nn as nn

  # Trường hợp sử dụng Foundation Model chuyên biệt cho mô học Phikon (Owkin)
  if backbone_name == "phikon":
    from transformers import AutoModel
    # Định nghĩa lớp phân loại Linear Probing dựa trên bộ trích xuất đặc trưng Phikon
    class PhikonClassifier(nn.Module):
      def __init__(self):
        super().__init__()
        # Nạp trọng số tiền huấn luyện tự giám sát TCGA SSL của Phikon từ HuggingFace Hub
        self.encoder = AutoModel.from_pretrained("owkin/phikon")
        # Giao thức Linear Probing: Đóng băng toàn bộ trọng số của Vision Transformer backbone
        for p in self.encoder.parameters():
          p.requires_grad = False
        # Đầu phân loại tuyến tính ánh xạ từ véc-tơ đặc trưng 768 chiều sang 9 lớp mô học
        self.fc = nn.Linear(768, num_classes)

      def forward(self, x):
        # Trích xuất biểu diễn của token đại diện [CLS] tại vị trí index 0
        feat = self.encoder(x).last_hidden_state[:, 0]
        # Đưa qua tầng tuyến tính để tính toán phân phối logits của 9 lớp
        return self.fc(feat)

    # Khởi tạo và trả về đối tượng phân loại Phikon
    return PhikonClassifier()

  # Trường hợp sử dụng kiến trúc hiện đại ConvNeXt-Tiny tiền huấn luyện ImageNet-1K
  if backbone_name == "convnext_tiny":
    return timm.create_model("convnext_tiny.fb_in1k", pretrained=True, num_classes=num_classes)

  # Mặc định sử dụng kiến trúc ResNet-50 kinh điển tiền huấn luyện ImageNet-1K
  return timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)


# ==============================================================================
# Các hàm Huấn luyện (Training) & Đánh giá (Evaluation)
# ==============================================================================

def evaluate_model(model, loader, device) -> Dict[str, float]:
  """Đánh giá mô hình trên một tập dữ liệu và tính toán các chỉ số Macro-F1, Acc, AUROC."""
  import torch
  # Nhập các hàm đo lường chỉ số từ scikit-learn
  from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

  # Chuyển mô hình sang chế độ suy luận evaluation mode (tắt dropout, cố định batchnorm)
  model.eval()
  # Danh sách gom kết quả dự đoán, xác suất và nhãn thực tế
  all_preds, all_probs, all_targets = [], [], []

  # Vô hiệu hóa tính toán đạo hàm autograd để tiết kiệm bộ nhớ GPU và tăng tốc
  with torch.no_grad():
    # Lặp qua từng batch dữ liệu từ DataLoader
    for images, targets in loader:
      # Đẩy tensor ảnh lên thiết bị tính toán (GPU hoặc CPU) với non_blocking tối ưu truyền dữ liệu
      images = images.to(device, non_blocking=True)
      # Tự động chuyển đổi độ chính xác hỗn hợp AMP fp16 tăng tốc độ suy luận
      with torch.amp.autocast("cuda"):
        logits = model(images)
      # Tính toán xác suất dự đoán bằng hàm softmax dọc theo chiều số lớp
      probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
      # Lưu lại xác suất phục vụ tính diện tích dưới đường cong ROC (AUROC)
      all_probs.append(probs)
      # Lấy chỉ số có xác suất lớn nhất làm lớp dự đoán
      all_preds.append(np.argmax(probs, axis=1))
      # Chuyển đổi nhãn thực tế sang mảng NumPy
      all_targets.append(targets.numpy())

  # Nối toàn bộ nhãn thực tế của tất cả các batch thành mảng 1 chiều
  y_true = np.concatenate(all_targets)
  # Nối toàn bộ nhãn dự đoán của tất cả các batch
  y_pred = np.concatenate(all_preds)
  # Nối ma trận xác suất dự đoán của tất cả các batch
  y_prob = np.concatenate(all_probs)

  # Tính Macro-F1: trung bình cộng không trọng số điểm F1 của cả 9 lớp mô học
  macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
  # Tính Balanced Accuracy: độ chính xác trung bình cân bằng giữa các lớp
  bal_acc = balanced_accuracy_score(y_true, y_pred)
  # Tính Accuracy tiêu chuẩn tổng thể
  acc = accuracy_score(y_true, y_pred)
  try:
    # Tính Macro AUROC theo chiến lược Một-đối-tất-cả (One-vs-Rest)
    auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
  except Exception:
    # Nếu gặp trường hợp ngoại lệ (thiếu lớp), gán giá trị nan an toàn
    auroc = float("nan")

  # Trả về từ điển kết quả đo lường chất lượng mô hình
  return {"macro_f1": float(macro_f1), "balanced_acc": float(bal_acc), "accuracy": float(acc), "auroc": float(auroc)}


def train_experiment(
  exp: Dict[str, str],
  splits: Dict[str, pd.DataFrame],
  ref_img_path: Path,
  epochs: int = 8,
  batch_size: int = 64,
  lr: float = 1e-3,
  device_str: str = "cuda",
) -> Dict[str, Any]:
  """Huấn luyện một cấu hình thí nghiệm và đánh giá độ bền vững giữa nội miền và ngoại miền."""
  import torch
  import torch.nn as nn
  from torch.utils.data import DataLoader

  # Khởi tạo đối tượng thiết bị phần cứng thực thi PyTorch
  device = torch.device(device_str)
  # Đọc ảnh tham chiếu chuẩn từ tệp và chuyển sang mảng NumPy RGB
  ref_rgb = np.array(Image.open(ref_img_path).convert("RGB"))

  # Khởi tạo hàm chuẩn hóa màu nhuộm dựa trên cấu hình thí nghiệm
  norm_fn = None
  if exp["norm"] == "reinhard":
    # Tính thống kê màu LAB trên ảnh tham chiếu
    ref_stats = reinhard_fit(ref_rgb)
    # Gán hàm lambda chuẩn hóa Reinhard
    norm_fn = lambda img: reinhard_apply(img, ref_stats)
  elif exp["norm"] == "macenko":
    # Phân tích ma trận màu nhuộm và nồng độ SVD trên ảnh tham chiếu
    ref_params = macenko_fit(ref_rgb)
    # Gán hàm lambda chuẩn hóa Macenko
    norm_fn = lambda img: macenko_apply(img, ref_params)

  # Khởi tạo tập dữ liệu huấn luyện áp dụng chính sách tăng cường tương ứng
  train_ds = HistologyDataset(splits["train"], PatchTransform(policy=exp["aug"], norm_fn=norm_fn, is_train=True))
  # Khởi tạo tập validation nội miền không tăng cường
  val_ds = HistologyDataset(splits["val_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
  # Khởi tạo tập kiểm thử nội miền (In-Domain Test)
  test_id_ds = HistologyDataset(splits["test_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
  # Khởi tạo tập kiểm thử ngoại miền bệnh viện độc lập (Out-of-Domain Test)
  test_ood_ds = HistologyDataset(splits["test_ood"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))

  # Thiết lập số luồng nạp dữ liệu: 4 luồng trên Linux, 0 luồng trên Windows tránh lỗi multiprocessing
  num_workers = 4 if os.name != "nt" else 0
  # DataLoader cho tập huấn luyện: xáo trộn dữ liệu (shuffle=True), khóa bộ nhớ GPU (pin_memory=True)
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  # DataLoader cho tập kiểm định validation
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  # DataLoader cho tập kiểm thử nội miền ID
  test_id_loader = DataLoader(test_id_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
  # DataLoader cho tập kiểm thử ngoại viện OOD
  test_ood_loader = DataLoader(test_ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

  # Xây dựng mô hình mạng nơ-ron và chuyển toàn bộ tham số lên thiết bị tính toán
  model = build_model(exp["backbone"]).to(device)
  # Thuật toán tối ưu hóa AdamW với hệ số phân rã trọng số weight decay chống overfitting
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr if exp["backbone"] != "phikon" else 3e-4, weight_decay=0.05)
  # Bộ điều chỉnh tốc độ học Cosine Annealing giảm dần learning rate theo hình sin
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
  # Hàm mất mát CrossEntropyLoss kết hợp kỹ thuật làm mịn nhãn Label Smoothing 0.1
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  # Bộ chia tỷ lệ gradient GradScaler cho tính toán độ chính xác hỗn hợp AMP trên GPU
  scaler = torch.amp.GradScaler("cuda")

  # Biến theo dõi điểm Macro-F1 cao nhất trên tập validation
  best_val_f1 = -1.0
  # Biến lưu trữ bản sao trọng số tốt nhất của mô hình
  best_state = None
  # Ghi nhận thời điểm bắt đầu huấn luyện
  t0 = time.time()

  # In thông tin bắt đầu huấn luyện cell thí nghiệm
  print(f"\n[{exp['id']}] Training {exp['backbone']} (norm={exp['norm']}, aug={exp['aug']}) for {epochs} epochs...")
  # Vòng lặp huấn luyện qua từng epoch
  for epoch in range(1, epochs + 1):
    # Đặt mô hình ở chế độ huấn luyện train mode (bật dropout, cập nhật batchnorm)
    model.train()
    # Tổng mất mát tích lũy của epoch
    total_loss = 0.0
    # Duyệt qua từng batch dữ liệu huấn luyện
    for images, targets in train_loader:
      # Chuyển ảnh và nhãn lên GPU
      images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
      # Xóa sạch đạo hàm gradient tích lũy của bước trước
      optimizer.zero_grad(set_to_none=True)
      # Tính toán lan truyền xuôi (forward pass) với chế độ AMP fp16
      with torch.amp.autocast("cuda"):
        loss = criterion(model(images), targets)
      # Lan truyền ngược gradient có co giãn tỷ lệ chống tràn số dưới (underflow)
      scaler.scale(loss).backward()
      # Cập nhật trọng số mô hình thông qua scaler
      scaler.step(optimizer)
      # Cập nhật hệ số co giãn scaler cho bước tiếp theo
      scaler.update()
      # Cập nhật lịch trình tốc độ học Cosine
      scheduler.step()
      # Cộng dồn giá trị mất mát của batch
      total_loss += loss.item()

    # Đánh giá chất lượng mô hình trên tập kiểm định validation sau mỗi epoch
    val_res = evaluate_model(model, val_loader, device)
    # Nếu điểm Macro-F1 vượt qua kỷ lục trước đó, lưu lại checkpoint trọng số tốt nhất
    if val_res["macro_f1"] > best_val_f1:
      best_val_f1 = val_res["macro_f1"]
      best_state = copy.deepcopy(model.state_dict())

    # In nhật ký tiến trình từng epoch ra màn hình console
    print(f"  ep {epoch}/{epochs} | loss={total_loss / len(train_loader):.4f} | val_f1={val_res['macro_f1']:.4f} (best={best_val_f1:.4f})")

  # Phục hồi lại trọng số tốt nhất đã lưu để thực hiện kiểm thử độc lập
  if best_state is not None:
    model.load_state_dict(best_state)

  # Đánh giá hiệu năng trên tập kiểm thử nội miền (In-Domain Test ID)
  id_res = evaluate_model(model, test_id_loader, device)
  # Đánh giá hiệu năng trên tập kiểm thử ngoại viện độc lập (Out-of-Domain Test OOD)
  ood_res = evaluate_model(model, test_ood_loader, device)
  # Tính tổng thời gian huấn luyện tính theo phút
  trained_min = round((time.time() - t0) / 60.0, 2)

  # Tính độ suy giảm hiệu năng Delta-F1 = ID F1 - OOD F1 (càng nhỏ càng ít bị suy giảm khi chuyển viện)
  delta_f1 = round(id_res["macro_f1"] - ood_res["macro_f1"], 4)
  # Tính Tỷ lệ bền vững Robustness Ratio RR = (OOD F1 / ID F1) * 100% (càng gần 100% càng tốt)
  rr_f1 = round((ood_res["macro_f1"] / max(id_res["macro_f1"], 1e-6)) * 100.0, 2)

  # Tổng hợp toàn bộ kết quả đo lường thành đối tượng từ điển có cấu trúc
  result = {
    "exp_id": exp["id"],
    "stage": exp["stage"],
    "backbone": exp["backbone"],
    "norm": exp["norm"],
    "aug": exp["aug"],
    "best_val_f1": round(best_val_f1, 4),
    "test_id_f1": round(id_res["macro_f1"], 4),
    "test_ood_f1": round(ood_res["macro_f1"], 4),
    "delta_f1": delta_f1,
    "rr_f1": rr_f1,
    "test_id_acc": round(id_res["accuracy"], 4),
    "test_ood_acc": round(ood_res["accuracy"], 4),
    "minutes": trained_min,
  }
  # In kết quả tóm tắt cuối cùng của cell thí nghiệm
  print(f"[{exp['id']} Result] ID F1={result['test_id_f1']} | OOD F1={result['test_ood_f1']} | Delta F1={delta_f1} | RR={rr_f1}% ({trained_min}m)")
  # Trả về từ điển kết quả
  return result


# ==============================================================================
# Hàm điều phối chính (Main Orchestrator)
# ==============================================================================

def main():
  # Khởi tạo bộ phân tích tham số dòng lệnh CLI
  parser = argparse.ArgumentParser(description="Run 13-cell Histopathology Staining Robustness Ablations")
  # Tham số đường dẫn thư mục tập dữ liệu nội miền nguồn
  parser.add_argument("--data-root", default="data/raw/NCT-CRC-HE-100K-NONORM", help="Path to in-domain dataset")
  # Tham số đường dẫn thư mục tập dữ liệu ngoại viện đích
  parser.add_argument("--target-root", default="data/raw/CRC-VAL-HE-7K", help="Path to out-of-domain dataset")
  # Tham số đường dẫn thư mục lưu kết quả đầu ra
  parser.add_argument("--out-dir", default="results", help="Directory for output metrics and tables")
  # Tham số kích thước tập mẫu con (mặc định lấy 25,000 mẫu đại diện)
  parser.add_argument("--subset", type=int, default=25000, help="Subset size (default 25000; 0 for full 100k)")
  # Tham số số epoch huấn luyện cho mỗi thí nghiệm
  parser.add_argument("--epochs", type=int, default=8, help="Epochs per experiment")
  # Tham số kích thước batch size
  parser.add_argument("--batch-size", type=int, default=64)
  # Tham số tùy chọn chỉ chạy một số thí nghiệm cụ thể
  parser.add_argument("--experiments", nargs="*", default=None, help="Filter experiment IDs to run (e.g. EXP-01 EXP-02)")
  # Cờ chạy thử nghiệm kiểm tra kế hoạch mà không tốn tài nguyên GPU
  parser.add_argument("--dry-run", action="store_true", help="Print experiment plan and exit")
  # Tiến hành phân tích các đối số truyền vào từ dòng lệnh
  args = parser.parse_args()

  # Chuyển đổi đường dẫn thư mục kết quả sang đối tượng Path
  out_path = Path(args.out_dir)
  # Tạo thư mục kết quả nếu chưa tồn tại
  out_path.mkdir(parents=True, exist_ok=True)

  # Lấy danh sách đầy đủ 13 thí nghiệm mặc định
  exp_list = EXPERIMENTS
  # Nếu người dùng chỉ định danh sách thí nghiệm cụ thể, tiến hành lọc
  if args.experiments:
    wanted = set(args.experiments)
    exp_list = [e for e in exp_list if e["id"] in wanted]

  # Nếu ở chế độ chạy thử nghiệm Dry Run, in danh sách kế hoạch và kết thúc sớm
  if args.dry_run:
    print(f"Plan: {len(exp_list)} experiments queued (subset={args.subset}, epochs={args.epochs}, batch={args.batch_size})")
    for e in exp_list:
      print(f"  {e['id']} | {e['stage']} | backbone={e['backbone']:<13} | norm={e['norm']:<8} | aug={e['aug']:<12} | {e['notes']}")
    return

  # Nhập PyTorch để kiểm tra phần cứng
  import torch
  # Tự động chọn GPU CUDA nếu có sẵn, ngược lại dùng CPU
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # In thông tin thiết bị và tổng số thí nghiệm sẽ chạy
  print(f"Running on device: {device} | Total experiments: {len(exp_list)}")

  # Đường dẫn thư mục lưu các phân tập CSV
  splits_dir = out_path / "splits"
  # Thực hiện quét dữ liệu và tạo các phân tập phân tầng
  splits, ref_tile = prepare_dataset_splits(
    source_dir=Path(args.data_root),
    target_dir=Path(args.target_root),
    out_dir=splits_dir,
    subset_size=args.subset,
  )

  # Danh sách gom kết quả của từng cell thí nghiệm
  results = []
  # Duyệt tuần tự qua từng cell thí nghiệm trong danh sách
  for exp in exp_list:
    # Huấn luyện và nhận kết quả đo lường
    res = train_experiment(
      exp=exp,
      splits=splits,
      ref_img_path=ref_tile,
      epochs=args.epochs,
      batch_size=args.batch_size,
      device_str=device,
    )
    # Thêm kết quả vào danh sách tổng
    results.append(res)

  # Chuyển đổi danh sách kết quả thành bảng DataFrame Pandas
  res_df = pd.DataFrame(results)
  # Đường dẫn tệp bảng kết quả CSV
  csv_file = out_path / "summary_results.csv"
  # Đường dẫn tệp bảng kết quả định dạng Markdown
  md_file = out_path / "RESULTS_TABLE.md"

  # Xuất bảng kết quả ra tệp CSV
  res_df.to_csv(csv_file, index=False)
  try:
    # Định dạng DataFrame thành bảng Markdown trực quan
    md_content = res_df.to_markdown(index=False)
  except Exception:
    # Dự phòng nếu thiếu thư viện tabulate
    md_content = res_df.to_string(index=False)

  # Ghi nội dung bảng Markdown ra tệp
  md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")

  # In đường phân cách thẩm mỹ ra console
  print("\n" + "=" * 80)
  print("FINAL ABLATION RESULTS TABLE")
  print("=" * 80)
  # In toàn bộ bảng kết quả Markdown hoàn chỉnh
  print(md_content)
  # Thông báo vị trí các tệp kết quả vừa được tạo
  print(f"\n[Artifacts] Wrote {csv_file} and {md_file}")


# Điểm vào chính của chương trình khi thực thi trực tiếp từ terminal
if __name__ == "__main__":
  main()
