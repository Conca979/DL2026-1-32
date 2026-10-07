"""Danh mục ma trận 13 cell thí nghiệm ablation kiểm tra độ bền vững mô bệnh học."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Các kiểu dữ liệu phụ trợ
from typing import Dict, List, Optional
# Nhập đối tượng cấu hình ô thí nghiệm
from src.config.schemas import ExperimentCell

# Danh mục 13 ô thí nghiệm ablation chính thức với kiểu dữ liệu ExperimentCell chặt chẽ
EXPERIMENTS_CATALOG: List[ExperimentCell] = [
    # Giai đoạn 0: Thí nghiệm đối chứng gốc (Anchor Baseline) - ResNet-50 thô không phòng vệ
    ExperimentCell(
        id="EXP-01",
        stage="Stage 0",
        backbone="resnet50",
        norm="none",
        aug="none",
        notes="Raw baseline",
    ),
    # Giai đoạn 1: Chuẩn hóa màu nhuộm thống kê Reinhard trong không gian LAB
    ExperimentCell(
        id="EXP-02",
        stage="Stage 1",
        backbone="resnet50",
        norm="reinhard",
        aug="none",
        notes="Statistical LAB transfer",
    ),
    # Giai đoạn 1: Chuẩn hóa màu nhuộm phân rã quang học Macenko bằng SVD
    ExperimentCell(
        id="EXP-03",
        stage="Stage 1",
        backbone="resnet50",
        norm="macenko",
        aug="none",
        notes="Optical density deconvolution",
    ),
    # Giai đoạn 2: Tăng cường dữ liệu biến đổi không gian (lật ngang, dọc, xoay 90 độ)
    ExperimentCell(
        id="EXP-04",
        stage="Stage 2",
        backbone="resnet50",
        norm="none",
        aug="aug_geo",
        notes="Spatial invariance (flips/rot)",
    ),
    # Giai đoạn 2: Tăng cường dữ liệu biến đổi nồng độ màu sinh học HED Jitter
    ExperimentCell(
        id="EXP-05",
        stage="Stage 2",
        backbone="resnet50",
        norm="none",
        aug="aug_stain",
        notes="HED stain jitter",
    ),
    # Giai đoạn 2: Kết hợp đồng thời tăng cường không gian và nhiễu nồng độ HED
    ExperimentCell(
        id="EXP-06",
        stage="Stage 2",
        backbone="resnet50",
        norm="none",
        aug="aug_combined",
        notes="Spatial + HED jitter",
    ),
    # Giai đoạn 3: Tương tác Macenko kết hợp tăng cường không gian hình học
    ExperimentCell(
        id="EXP-07",
        stage="Stage 3",
        backbone="resnet50",
        norm="macenko",
        aug="aug_geo",
        notes="Macenko + spatial",
    ),
    # Giai đoạn 3: Tương tác Macenko kết hợp tăng cường nhiễu nồng độ HED
    ExperimentCell(
        id="EXP-08",
        stage="Stage 3",
        backbone="resnet50",
        norm="macenko",
        aug="aug_stain",
        notes="Macenko + stain jitter",
    ),
    # Giai đoạn 3: ResNet-50 phòng vệ toàn diện (Macenko + Tăng cường kết hợp)
    ExperimentCell(
        id="EXP-09",
        stage="Stage 3",
        backbone="resnet50",
        norm="macenko",
        aug="aug_combined",
        notes="Full defense ResNet",
    ),
    # Giai đoạn 4: Đánh giá kiến trúc ConvNeXt-Tiny thô chưa phòng vệ
    ExperimentCell(
        id="EXP-10",
        stage="Stage 4",
        backbone="convnext_tiny",
        norm="none",
        aug="none",
        notes="ConvNeXt-Tiny raw",
    ),
    # Giai đoạn 4: Đánh giá kiến trúc ConvNeXt-Tiny kết hợp phòng vệ toàn diện
    ExperimentCell(
        id="EXP-11",
        stage="Stage 4",
        backbone="convnext_tiny",
        norm="macenko",
        aug="aug_combined",
        notes="ConvNeXt-Tiny defended",
    ),
    # Giai đoạn 4: Đánh giá Foundation Model Phikon (TCGA SSL ViT) thô chưa phòng vệ
    ExperimentCell(
        id="EXP-12",
        stage="Stage 4",
        backbone="phikon",
        norm="none",
        aug="none",
        notes="Phikon (TCGA SSL ViT) raw",
    ),
    # Giai đoạn 4: Đánh giá Foundation Model Phikon khi áp dụng cơ chế phòng vệ
    ExperimentCell(
        id="EXP-13",
        stage="Stage 4",
        backbone="phikon",
        norm="macenko",
        aug="aug_combined",
        notes="Phikon defended",
    ),
]

# Danh sách kiểu từ điển Dict để tương thích ngược với mã nguồn nguyên bản
EXPERIMENTS: List[Dict[str, str]] = [exp.to_dict() for exp in EXPERIMENTS_CATALOG]


def get_all_experiments() -> List[ExperimentCell]:
    """Trả về bản sao danh sách đầy đủ 13 ô thí nghiệm nghiên cứu."""
    return list(EXPERIMENTS_CATALOG)


def get_experiment_by_id(exp_id: str) -> Optional[ExperimentCell]:
    """Tìm kiếm một ô thí nghiệm theo mã định danh duy nhất (không phân biệt hoa thường)."""
    # Duyệt qua danh mục thí nghiệm
    for exp in EXPERIMENTS_CATALOG:
        # Nếu trùng khớp mã định danh, trả về đối tượng cấu hình
        if exp.id.upper() == exp_id.upper():
            return exp
    # Không tìm thấy thì trả về None
    return None


def filter_experiments(experiment_ids: Optional[List[str]] = None) -> List[ExperimentCell]:
    """Lọc danh mục thí nghiệm theo danh sách các mã ID được người dùng truyền vào."""
    # Nếu danh sách rỗng hoặc None, trả về tất cả 13 thí nghiệm
    if not experiment_ids:
        return get_all_experiments()
    # Tập hợp các ID mục tiêu viết hoa
    target_ids = {i.upper() for i in experiment_ids}
    # Lọc và trả về các ô thí nghiệm nằm trong danh sách yêu cầu
    return [e for e in EXPERIMENTS_CATALOG if e.id.upper() in target_ids]
