"""Định nghĩa cấu trúc dữ liệu lưu trữ kết quả và chỉ số đo lường thí nghiệm."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Lớp Dataclass chuẩn của Python
from dataclasses import dataclass
# Các kiểu dữ liệu phụ trợ
from typing import Any, Dict


@dataclass
class ExperimentResult:
    """Dataclass lưu trữ đầy đủ kết quả đo lường độ bền vững của một ô thí nghiệm."""

    # Mã định danh thí nghiệm (ví dụ: 'EXP-01')
    exp_id: str
    # Tên giai đoạn ablation
    stage: str
    # Tên kiến trúc mạng nơ-ron
    backbone: str
    # Tên thuật toán chuẩn hóa màu
    norm: str
    # Tên chính sách tăng cường dữ liệu
    aug: str
    # Điểm Macro-F1 cao nhất trên tập validation
    best_val_f1: float
    # Điểm Macro-F1 trên tập kiểm thử nội miền ID
    test_id_f1: float
    # Điểm Macro-F1 trên tập kiểm thử ngoại miền độc lập OOD
    test_ood_f1: float
    # Độ suy giảm hiệu năng Delta-F1 = ID F1 - OOD F1
    delta_f1: float
    # Tỷ lệ bền vững Robustness Ratio RR = (OOD F1 / ID F1) * 100%
    rr_f1: float
    # Độ chính xác Accuracy trên tập kiểm thử nội miền
    test_id_acc: float
    # Độ chính xác Accuracy trên tập kiểm thử ngoại miền
    test_ood_acc: float
    # Tổng thời gian huấn luyện tính theo phút
    minutes: float

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi đối tượng sang định dạng từ điển chuẩn."""
        return {
            "exp_id": self.exp_id,
            "stage": self.stage,
            "backbone": self.backbone,
            "norm": self.norm,
            "aug": self.aug,
            "best_val_f1": self.best_val_f1,
            "test_id_f1": self.test_id_f1,
            "test_ood_f1": self.test_ood_f1,
            "delta_f1": self.delta_f1,
            "rr_f1": self.rr_f1,
            "test_id_acc": self.test_id_acc,
            "test_ood_acc": self.test_ood_acc,
            "minutes": self.minutes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentResult:
        """Tạo đối tượng ExperimentResult từ từ điển dữ liệu có sẵn."""
        return cls(
            exp_id=str(data["exp_id"]),
            stage=str(data["stage"]),
            backbone=str(data["backbone"]),
            norm=str(data["norm"]),
            aug=str(data["aug"]),
            best_val_f1=float(data["best_val_f1"]),
            test_id_f1=float(data["test_id_f1"]),
            test_ood_f1=float(data["test_ood_f1"]),
            delta_f1=float(data["delta_f1"]),
            rr_f1=float(data["rr_f1"]),
            test_id_acc=float(data["test_id_acc"]),
            test_ood_acc=float(data["test_ood_acc"]),
            minutes=float(data["minutes"]),
        )
