"""Lớp đóng gói Foundation Model mô bệnh học Phikon với đầu phân loại Linear Probing."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Các kiểu dữ liệu phụ trợ
from typing import Any

# Nạp an toàn lớp cơ sở nn.Module
try:
    import torch
    import torch.nn as nn
    _ModuleBase = nn.Module
except ImportError:
    class _ModuleBase:  # type: ignore
        """Lớp cơ sở dự phòng khi chưa cài đặt PyTorch trong môi trường tĩnh."""
        pass


class PhikonClassifier(_ModuleBase):
    """Đầu phân loại Linear Probing gắn trên bộ mã hóa Vision Transformer đóng băng của Phikon."""

    def __init__(
        self,
        pretrained_name: str = "owkin/phikon",
        num_classes: int = 9,
        freeze_encoder: bool = True,
    ) -> None:
        """Khởi tạo mô hình Phikon từ HuggingFace Hub và cấu hình tầng phân loại.

        Tham số:
            pretrained_name: Tên định danh repo trên HuggingFace (mặc định 'owkin/phikon').
            num_classes: Số lượng lớp phân loại đầu ra (9 lớp mô học).
            freeze_encoder: Cờ đóng băng trọng số của backbone ViT (Linear Probing).
        """
        # Kiểm tra sự có mặt của torch và transformers
        try:
            import torch.nn as nn
            from transformers import AutoModel
        except ImportError as err:
            raise ImportError(
                "torch and transformers are required for PhikonClassifier: "
                "pip install torch transformers"
            ) from err

        super().__init__()
        # Nạp mô hình AutoModel chứa trọng số tiền huấn luyện tự giám sát (iBOT ViT-Base)
        self.encoder = AutoModel.from_pretrained(pretrained_name)

        # Giao thức Linear Probing: Đóng băng toàn bộ các tham số của ViT encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Lấy kích thước ẩn biểu diễn đặc trưng (mặc định của ViT-Base là 768)
        hidden_dim = self.encoder.config.hidden_size if hasattr(self.encoder, "config") else 768
        # Khởi tạo tầng phân loại tuyến tính học từ đầu
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Any) -> Any:
        """Lan truyền xuôi: Trích xuất token [CLS] và tính toán phân phối logits 9 lớp."""
        # Chuyển ảnh qua bộ mã hóa ViT
        outputs = self.encoder(x)
        # Lấy véc-tơ biểu diễn của token đại diện lớp học [CLS] tại chỉ số đầu tiên (index 0)
        cls_feat = outputs.last_hidden_state[:, 0]
        # Chiếu qua tầng tuyến tính để thu được logits chưa qua softmax
        return self.fc(cls_feat)
