"""Gói con động cơ huấn luyện và đánh giá mô hình."""

# Nhập lớp đánh giá Evaluator và hàm tiện ích
from src.engine.evaluator import Evaluator, evaluate_model
# Nhập lớp huấn luyện ExperimentTrainer và hàm tiện ích
from src.engine.trainer import ExperimentTrainer, train_experiment

# Danh sách biểu tượng xuất khẩu công khai của gói engine
__all__ = [
    "Evaluator",
    "evaluate_model",
    "ExperimentTrainer",
    "train_experiment",
]
