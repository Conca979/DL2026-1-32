"""Các Dataclass định nghĩa cấu hình cho thí nghiệm, phân chia dữ liệu và huấn luyện."""

# Cho phép hỗ trợ type hinting hiện đại
from __future__ import annotations

# Nhập dataclass và field để tạo các lớp dữ liệu bất biến hoặc có giá trị mặc định
from dataclasses import dataclass, field
# Nhập Path để làm việc an toàn với đường dẫn tệp tin trên đa nền tảng OS
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import List, Optional


# Dataclass bất biến định nghĩa thông số của một ô thí nghiệm ablation
@dataclass(frozen=True)
class ExperimentCell:
    """Đặc tả thông số cấu hình của một ô trong ma trận nghiên cứu ablation."""

    # Mã định danh thí nghiệm duy nhất (ví dụ: 'EXP-01')
    id: str
    # Tên giai đoạn thí nghiệm (ví dụ: 'Stage 0', 'Stage 1')
    stage: str
    # Tên kiến trúc mạng nơ-ron nền tảng (ví dụ: 'resnet50', 'phikon')
    backbone: str
    # Phương pháp chuẩn hóa màu sắc (ví dụ: 'none', 'reinhard', 'macenko')
    norm: str
    # Chính sách tăng cường dữ liệu (ví dụ: 'none', 'aug_geo', 'aug_combined')
    aug: str
    # Ghi chú mô tả giả thuyết hoặc mục tiêu của thí nghiệm
    notes: str = ""

    def to_dict(self) -> dict:
        """Chuyển đổi đối tượng Dataclass sang Dictionary hỗ trợ tương thích mã cũ."""
        return {
            "id": self.id,
            "stage": self.stage,
            "backbone": self.backbone,
            "norm": self.norm,
            "aug": self.aug,
            "notes": self.notes,
        }


# Dataclass cấu hình cho việc phân tách tập dữ liệu nguồn và đích
@dataclass
class SplitConfig:
    """Cấu hình các tham số phân tách dữ liệu và trích chọn mẫu."""

    # Đường dẫn thư mục chứa dữ liệu nội miền nguồn (In-Domain Source)
    source_dir: Path = Path("data/raw/NCT-CRC-HE-100K-NONORM")
    # Đường dẫn thư mục chứa dữ liệu ngoại miền đích (Out-of-Domain Target)
    target_dir: Path = Path("data/raw/CRC-VAL-HE-7K")
    # Thư mục lưu trữ các tệp phân tách CSV và ảnh tham chiếu chuẩn hóa
    out_dir: Path = Path("results/splits")
    # Kích thước tập con lấy mẫu phân tầng (0 nếu lấy toàn bộ dữ liệu)
    subset_size: int = 25000
    # Tỷ lệ phân chia tập huấn luyện (70%)
    train_ratio: float = 0.70
    # Tỷ lệ phân chia tập kiểm định nội miền (15%)
    val_ratio: float = 0.15
    # Tỷ lệ phân chia tập kiểm thử nội miền (15%)
    test_ratio: float = 0.15
    # Hạt giống số ngẫu nhiên đảm bảo tính tái lập kết quả
    seed: int = 100


# Dataclass cấu hình các siêu tham số cho động cơ huấn luyện mô hình học sâu
@dataclass
class TrainingConfig:
    """Siêu tham số cho quá trình huấn luyện và tối ưu mô hình học sâu."""

    # Số lượng chu kỳ huấn luyện (Epochs) cho mỗi thí nghiệm
    epochs: int = 8
    # Kích thước lô huấn luyện (Batch size)
    batch_size: int = 64
    # Tốc độ học cơ sở (Base Learning Rate) cho mạng tích chập thông thường
    learning_rate: float = 1e-3
    # Tốc độ học riêng biệt cho mô hình nền tảng Phikon
    phikon_lr: float = 3e-4
    # Hệ số phân rã trọng số (Weight Decay) cho thuật toán AdamW
    weight_decay: float = 0.05
    # Hệ số làm mịn nhãn (Label Smoothing) cho hàm mất mát CrossEntropy
    label_smoothing: float = 0.1
    # Tốc độ học tối thiểu cho bộ lập lịch Cosine Annealing
    min_lr: float = 1e-5
    # Thiết bị phần cứng thực thi ('auto', 'cuda', hoặc 'cpu')
    device: str = "auto"
    # Số luồng nạp dữ liệu song song (None để tự động phát hiện theo OS)
    num_workers: Optional[int] = None


# Dataclass tổng thể gom toàn bộ cấu hình điều phối đường ống thực thi
@dataclass
class PipelineConfig:
    """Cấu hình tổng quan của toàn bộ hệ thống pipeline thí nghiệm."""

    # Cấu hình phân chia dữ liệu
    split: SplitConfig = field(default_factory=SplitConfig)
    # Cấu hình huấn luyện mô hình
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # Thư mục lưu trữ các báo cáo và bảng kết quả cuối cùng
    out_dir: Path = Path("results")
    # Danh sách các mã thí nghiệm cần chạy (None nếu muốn chạy tất cả 13 cell)
    experiment_ids: Optional[List[str]] = None
    # Cờ chạy kiểm tra kế hoạch mà không huấn luyện thực tế
    dry_run: bool = False
