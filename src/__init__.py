"""Gói phần mềm nghiên cứu độ bền vững phân loại mô bệnh học dưới sự biến thiên màu nhuộm.

Một framework dạng module hướng đối tượng chuyên nghiệp phục vụ đánh giá các cơ chế
phòng vệ biến thiên màu nhuộm lâm sàng qua các phương pháp chuẩn hóa, tăng cường dữ liệu và backbone thị giác.
"""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Phiên bản phần mềm chính thức
__version__ = "1.0.0"

# 1. Các hằng số toàn cục
from src.config.constants import (
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SUPPORTED_IMAGE_EXTS,
    HE_STAIN_MATRIX,
)
# 2. Các dataclass cấu hình
from src.config.schemas import (
    ExperimentCell,
    SplitConfig,
    TrainingConfig,
    PipelineConfig,
)
# 3. Danh mục các ô thí nghiệm ablation
from src.config.experiments import (
    EXPERIMENTS,
    EXPERIMENTS_CATALOG,
    get_all_experiments,
    get_experiment_by_id,
    filter_experiments,
)
# 4. Các thuật toán chuẩn hóa màu nhuộm
from src.normalization.base import BaseStainNormalizer
from src.normalization.reinhard import (
    ReinhardNormalizer,
    reinhard_fit,
    reinhard_apply,
    rgb_to_lab,
    lab_to_rgb,
)
from src.normalization.macenko import (
    MacenkoNormalizer,
    macenko_fit,
    macenko_apply,
)
from src.normalization.factory import NormalizerFactory
# 5. Các công cụ tăng cường dữ liệu
from src.augmentation.policies import AugmentationPolicy
from src.augmentation.hed_jitter import hed_stain_jitter, HEDStainJitterTransform
from src.augmentation.transforms import PatchTransform
# 6. Các công cụ quản lý và chia dữ liệu
from src.data.discovery import find_class_images
from src.data.dataset import HistologyDataset
from src.data.splitter import DatasetSplitter, prepare_dataset_splits
from src.data.dataloader import create_dataloaders
# 7. Các kiến trúc mô hình học sâu
from src.models.phikon import PhikonClassifier
from src.models.factory import ModelFactory, build_model
# 8. Động cơ huấn luyện và đánh giá
from src.engine.evaluator import Evaluator, evaluate_model
from src.engine.trainer import ExperimentTrainer, train_experiment
# 9. Báo cáo và xuất kết quả
from src.reporting.metrics import ExperimentResult
from src.reporting.reporter import ResultsReporter
# 10. Bộ điều phối đường ống cấp cao
from src.pipeline import ExperimentPipeline

# Danh sách đầy đủ các biểu tượng được công khai khi import gói `src`
__all__ = [
    "__version__",
    # Config
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SUPPORTED_IMAGE_EXTS",
    "HE_STAIN_MATRIX",
    "ExperimentCell",
    "SplitConfig",
    "TrainingConfig",
    "PipelineConfig",
    "EXPERIMENTS",
    "EXPERIMENTS_CATALOG",
    "get_all_experiments",
    "get_experiment_by_id",
    "filter_experiments",
    # Normalization
    "BaseStainNormalizer",
    "ReinhardNormalizer",
    "reinhard_fit",
    "reinhard_apply",
    "rgb_to_lab",
    "lab_to_rgb",
    "MacenkoNormalizer",
    "macenko_fit",
    "macenko_apply",
    "NormalizerFactory",
    # Augmentation
    "AugmentationPolicy",
    "hed_stain_jitter",
    "HEDStainJitterTransform",
    "PatchTransform",
    # Data
    "find_class_images",
    "HistologyDataset",
    "DatasetSplitter",
    "prepare_dataset_splits",
    "create_dataloaders",
    # Models
    "PhikonClassifier",
    "ModelFactory",
    "build_model",
    # Engine
    "Evaluator",
    "evaluate_model",
    "ExperimentTrainer",
    "train_experiment",
    # Reporting
    "ExperimentResult",
    "ResultsReporter",
    # Pipeline
    "ExperimentPipeline",
]
