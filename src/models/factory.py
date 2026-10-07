"""Lớp nhà máy khởi tạo kiến trúc mô hình học sâu hỗ trợ ResNet-50, ConvNeXt-Tiny và Phikon."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module ghi nhật ký log
import logging
# Các kiểu dữ liệu phụ trợ
from typing import Any, Optional

# Nhập hằng số số lượng lớp mô học
from src.config.constants import NUM_CLASSES
# Nhập bộ phân loại Phikon
from src.models.phikon import PhikonClassifier

# Khởi tạo logger
logger = logging.getLogger(__name__)


class ModelFactory:
    """Lớp nhà máy khởi tạo các backbone thị giác phục vụ bài toán phân loại mô bệnh học."""

    @staticmethod
    def build(
        backbone_name: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
    ) -> Any:
        """Xây dựng kiến trúc mô hình tương ứng dựa trên tên chuỗi định danh backbone.

        Tham số:
            backbone_name: Tên kiến trúc ('resnet50', 'convnext_tiny', 'phikon').
            num_classes: Số lượng lớp mô học mục tiêu (mặc định là 9).
            pretrained: Có nạp trọng số tiền huấn luyện hay không (mặc định True).

        Trả về:
            Đối tượng nn.Module sẵn sàng cho quá trình huấn luyện hoặc đánh giá.
        """
        # Làm sạch chuỗi tên kiến trúc
        name = backbone_name.strip().lower()

        # Trường hợp sử dụng Foundation Model Phikon
        if name == "phikon":
            logger.info("Initializing Phikon ViT backbone (frozen) + linear probe head...")
            # Khởi tạo bộ phân loại Phikon với backbone đóng băng
            return PhikonClassifier(num_classes=num_classes, freeze_encoder=True)

        # Nhập thư viện timm an toàn
        try:
            import timm
        except ImportError as err:
            raise ImportError("timm is required for backbone instantiation: pip install timm") from err

        # Trường hợp sử dụng kiến trúc mạng tích chập hiện đại ConvNeXt-Tiny
        if name == "convnext_tiny":
            logger.info("Initializing timm ConvNeXt-Tiny backbone...")
            return timm.create_model(
                "convnext_tiny.fb_in1k",
                pretrained=pretrained,
                num_classes=num_classes,
            )

        # Trường hợp sử dụng kiến trúc mạng ResNet-50 kinh điển
        if name in ("resnet50", "resnet_50", "default"):
            logger.info("Initializing timm ResNet-50 backbone...")
            return timm.create_model(
                "resnet50.a1_in1k",
                pretrained=pretrained,
                num_classes=num_classes,
            )

        # Phương án dự phòng: Nạp mô hình bất kỳ từ kho kiến trúc của timm
        logger.info("Attempting to initialize generic timm model: %s", backbone_name)
        return timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )


def build_model(backbone_name: str, num_classes: int = NUM_CLASSES) -> Any:
    """Hàm bọc thủ tục hỗ trợ tương thích trực tiếp với interface của run_experiments.py."""
    return ModelFactory.build(backbone_name=backbone_name, num_classes=num_classes)
