"""Các hằng số toàn cục và thông số kỹ thuật cho nghiên cứu độ bền vững mô bệnh học."""

# Hỗ trợ tính năng type hinting hiện đại của Python (trì hoãn nạp kiểu)
from __future__ import annotations

# Các kiểu dữ liệu hỗ trợ ép kiểu tĩnh
from typing import Dict, List, Set
# Thư viện tính toán ma trận mảng số đa chiều NumPy
import numpy as np

# Danh sách 9 lớp mô học mô bệnh học ung thư đại trực tràng chuẩn (NCT-CRC-HE)
CLASSES: List[str] = [
    "ADI",   # Mô mỡ (Adipose tissue)
    "BACK",  # Nền kính hiển vi / Khoảng trống không chứa mô (Background)
    "DEB",   # Mảnh vụn tế bào / Hoại tử (Debris)
    "LYM",   # Tế bào lympho miễn dịch xâm nhập (Lymphocytes)
    "MUC",   # Dịch nhầy nhầy biểu mô (Mucus)
    "MUS",   # Mô cơ trơn / Cơ vân (Muscle)
    "NORM",  # Niêm mạc đại tràng lành tính bình thường (Normal mucosa)
    "STR",   # Mô đệm ung thư liên kết (Stroma)
    "TUM",   # Biểu mô tuyến ung thư biểu mô đại tràng (Colorectal adenocarcinoma epithelium)
]

# Ánh xạ từ chuỗi tên phân lớp mô học sang số nguyên (0 đến 8) để mô hình xử lý
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
# Ánh xạ ngược từ chỉ số số nguyên sang tên phân lớp mô học
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASSES)}
# Tổng số phân lớp mô bệnh học mục tiêu (9 lớp)
NUM_CLASSES: int = len(CLASSES)

# Giá trị trung bình (Mean) chuẩn của bộ dữ liệu ImageNet theo thứ tự 3 kênh [R, G, B]
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
# Độ lệch chuẩn (Std) chuẩn của bộ dữ liệu ImageNet theo thứ tự 3 kênh [R, G, B]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Tập hợp các phần mở rộng định dạng tệp ảnh được hỗ trợ quét
SUPPORTED_IMAGE_EXTS: Set[str] = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

# Ma trận hấp thụ màu nhuộm Hematoxylin & Eosin (H&E) thực nghiệm thô (kích thước 3x2)
_RAW_HE_STAIN_MATRIX = np.array([
    [0.650, 0.072],  # Hệ số hấp thụ quang học kênh R của Hematoxylin và Eosin
    [0.704, 0.990],  # Hệ số hấp thụ quang học kênh G của Hematoxylin và Eosin
    [0.286, 0.105],  # Hệ số hấp thụ quang học kênh B của Hematoxylin và Eosin
], dtype=np.float64)

# Chuẩn hóa chuẩn véc-tơ đơn vị L2 dọc theo từng cột của ma trận hấp thụ
HE_STAIN_MATRIX: np.ndarray = _RAW_HE_STAIN_MATRIX / np.linalg.norm(
    _RAW_HE_STAIN_MATRIX, axis=0, keepdims=True
)
