"""Gói cấu hình và hằng số cho dự án nghiên cứu độ bền vững mô bệnh học."""

# Nhập các hằng số từ module constants
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
# Nhập các schema Dataclass cấu hình
from src.config.schemas import (
    ExperimentCell,
    SplitConfig,
    TrainingConfig,
    PipelineConfig,
)
# Nhập danh mục thí nghiệm và các hàm tiện ích lọc
from src.config.experiments import (
    EXPERIMENTS,
    EXPERIMENTS_CATALOG,
    get_all_experiments,
    get_experiment_by_id,
    filter_experiments,
)

# Danh sách các biểu tượng được công khai ra ngoài khi import gói
__all__ = [
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
]
