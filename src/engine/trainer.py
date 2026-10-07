"""Động cơ huấn luyện thực thi các cell thí nghiệm trong ma trận ablation."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module sao chép sâu đối tượng
import copy
# Module ghi nhật ký log
import logging
# Module hệ điều hành
import os
# Module đo thời gian
import time
# Lớp thao tác đường dẫn
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import Any, Dict, Optional, Union
# Thư viện mảng số học NumPy
import numpy as np
# Thư viện ảnh Pillow
from PIL import Image

# Nhập các schema cấu hình
from src.config.schemas import ExperimentCell, TrainingConfig
# Nhập nhà máy khởi tạo bộ chuẩn hóa màu
from src.normalization.factory import NormalizerFactory
# Nhập Dataset mô học
from src.data.dataset import HistologyDataset
# Nhập pipeline biến đổi patch
from src.augmentation.transforms import PatchTransform
# Nhập nhà máy khởi tạo mô hình
from src.models.factory import ModelFactory
# Nhập động cơ đánh giá
from src.engine.evaluator import Evaluator

# Khởi tạo logger
logger = logging.getLogger(__name__)


class ExperimentTrainer:
    """Lớp quản lý quá trình huấn luyện một cell thí nghiệm và đo lường độ bền vững khi chuyển miền."""

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        # Nạp cấu hình huấn luyện mặc định nếu không truyền riêng
        self.config = config or TrainingConfig()

    def train(
        self,
        exp: Union[ExperimentCell, Dict[str, str]],
        splits: Dict[str, Any],
        ref_img_path: Union[str, Path],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        device_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Huấn luyện một ô thí nghiệm với lịch trình Cosine Annealing và tính toán AMP.

        Tham số:
            exp: Đối tượng ExperimentCell hoặc Dict chứa thông tin thí nghiệm.
            splits: Từ điển chứa 4 DataFrame 'train', 'val_id', 'test_id', 'test_ood'.
            ref_img_path: Đường dẫn tới ảnh tham chiếu chuẩn hóa màu.
            epochs: Ghi đè số epoch huấn luyện nếu cần.
            batch_size: Ghi đè kích thước batch size nếu cần.
            lr: Ghi đè tốc độ học cơ sở nếu cần.
            device_str: Tên thiết bị phần cứng ('cuda' hoặc 'cpu').

        Trả về:
            Từ điển chứa tất cả các chỉ số đo lường hiệu năng và thông tin cấu hình.
        """
        # Nhập các gói phụ thuộc PyTorch an toàn
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
        except ImportError as err:
            raise ImportError(
                "torch is required for training: pip install torch"
            ) from err

        # Chuẩn hóa đối tượng exp về dạng ExperimentCell
        cell = (
            exp if isinstance(exp, ExperimentCell)
            else ExperimentCell(
                id=exp["id"],
                stage=exp.get("stage", ""),
                backbone=exp["backbone"],
                norm=exp["norm"],
                aug=exp["aug"],
                notes=exp.get("notes", ""),
            )
        )

        # Lấy số epoch thực thi
        n_epochs = epochs or self.config.epochs
        # Lấy kích thước batch
        b_size = batch_size or self.config.batch_size
        # Tự động chọn thiết bị nếu chưa chỉ định
        dev_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        # Khởi tạo đối tượng torch.device
        device = torch.device(dev_str)
        # Kiểm tra cờ thiết bị CUDA
        is_cuda = device.type == "cuda"

        # 1. Khởi tạo và khớp bộ chuẩn hóa màu nhuộm từ ảnh tham chiếu
        ref_rgb = np.array(Image.open(ref_img_path).convert("RGB"))
        normalizer = NormalizerFactory.create(cell.norm, reference_rgb=ref_rgb)
        norm_fn = normalizer.transform if normalizer is not None else None

        # 2. Xây dựng các đối tượng Dataset cho từng phân tập
        train_ds = HistologyDataset(
            splits["train"],
            PatchTransform(policy=cell.aug, norm_fn=norm_fn, is_train=True),
        )
        val_ds = HistologyDataset(
            splits["val_id"],
            PatchTransform(policy="none", norm_fn=norm_fn, is_train=False),
        )
        test_id_ds = HistologyDataset(
            splits["test_id"],
            PatchTransform(policy="none", norm_fn=norm_fn, is_train=False),
        )
        test_ood_ds = HistologyDataset(
            splits["test_ood"],
            PatchTransform(policy="none", norm_fn=norm_fn, is_train=False),
        )

        # Thiết lập số luồng nạp dữ liệu
        num_workers = self.config.num_workers
        if num_workers is None:
            num_workers = 4 if os.name != "nt" else 0

        # Khởi tạo DataLoader cho 4 phân tập
        train_loader = DataLoader(
            train_ds, batch_size=b_size, shuffle=True,
            num_workers=num_workers, pin_memory=is_cuda
        )
        val_loader = DataLoader(
            val_ds, batch_size=b_size, shuffle=False,
            num_workers=num_workers, pin_memory=is_cuda
        )
        test_id_loader = DataLoader(
            test_id_ds, batch_size=b_size, shuffle=False, num_workers=num_workers
        )
        test_ood_loader = DataLoader(
            test_ood_ds, batch_size=b_size, shuffle=False, num_workers=num_workers
        )

        # 3. Khởi tạo Mô hình, Bộ tối ưu, Bộ điều chỉnh tốc độ học, Hàm mất mát
        model = ModelFactory.build(cell.backbone).to(device)

        # Tốc độ học: Phikon sử dụng lr nhỏ hơn (3e-4) do đã có đặc trưng tốt
        base_lr = lr or (self.config.phikon_lr if cell.backbone == "phikon" else self.config.learning_rate)
        # Thuật toán AdamW với phân rã trọng số weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=self.config.weight_decay,
        )
        # Bộ lập lịch Cosine Annealing hạ tốc độ học mềm mại theo hình sin
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs * max(len(train_loader), 1),
            eta_min=self.config.min_lr,
        )
        # Hàm mất mát CrossEntropyLoss với làm mịn nhãn Label Smoothing 0.1
        criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        # Bộ co giãn gradient GradScaler cho tính toán AMP fp16
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

        # Điểm Macro-F1 cao nhất trên tập validation
        best_val_f1 = -1.0
        # Trọng số tốt nhất của mô hình
        best_state = None
        # Thời điểm bắt đầu
        t0 = time.time()

        # In thông báo bắt đầu huấn luyện ô thí nghiệm
        print(
            f"\n[{cell.id}] Training {cell.backbone} (norm={cell.norm}, aug={cell.aug}) "
            f"for {n_epochs} epochs..."
        )

        # Vòng lặp chính qua từng epoch
        for epoch in range(1, n_epochs + 1):
            # Đặt mô hình ở chế độ huấn luyện
            model.train()
            # Tổng mất mát của epoch
            total_loss = 0.0

            # Duyệt qua từng batch dữ liệu
            for images, targets in train_loader:
                # Đẩy dữ liệu lên GPU
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                # Xóa sạch gradient bước trước
                optimizer.zero_grad(set_to_none=True)

                # Lan truyền xuôi kết hợp tính toán AMP
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        loss = criterion(model(images), targets)
                    # Lan truyền ngược có co giãn scaler
                    scaler.scale(loss).backward()
                    # Cập nhật trọng số mô hình
                    scaler.step(optimizer)
                    # Cập nhật hệ số scaler
                    scaler.update()
                else:
                    loss = criterion(model(images), targets)
                    loss.backward()
                    optimizer.step()

                # Cập nhật bước lập lịch learning rate
                scheduler.step()
                # Cộng dồn mất mát
                total_loss += loss.item()

            # Đánh giá trên tập validation nội miền
            val_res = Evaluator.evaluate(model, val_loader, device)
            # Nếu điểm F1 vượt kỷ lục cũ, lưu bản sao trọng số mô hình
            if val_res["macro_f1"] > best_val_f1:
                best_val_f1 = val_res["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())

            # Tính trung bình mất mát của epoch
            avg_loss = total_loss / max(len(train_loader), 1)
            # In nhật ký tiến trình ra console
            print(
                f"  ep {epoch}/{n_epochs} | loss={avg_loss:.4f} | "
                f"val_f1={val_res['macro_f1']:.4f} (best={best_val_f1:.4f})"
            )

        # 4. Nạp lại trọng số tốt nhất để thực hiện kiểm thử độc lập
        if best_state is not None:
            model.load_state_dict(best_state)

        # Đánh giá trên tập kiểm thử nội miền (In-Domain Test ID)
        id_res = Evaluator.evaluate(model, test_id_loader, device)
        # Đánh giá trên tập kiểm thử ngoại viện độc lập (Out-of-Domain Test OOD)
        ood_res = Evaluator.evaluate(model, test_ood_loader, device)
        # Tính thời gian thực thi (phút)
        trained_min = round((time.time() - t0) / 60.0, 2)

        # Độ suy giảm Delta-F1 = ID - OOD (càng nhỏ càng giữ vững chất lượng)
        delta_f1 = round(id_res["macro_f1"] - ood_res["macro_f1"], 4)
        # Tỷ lệ bền vững Robustness Ratio RR = (OOD / ID) * 100% (càng cao càng tốt)
        rr_f1 = round((ood_res["macro_f1"] / max(id_res["macro_f1"], 1e-6)) * 100.0, 2)

        # Tổng hợp toàn bộ kết quả đo lường vào từ điển
        result: Dict[str, Any] = {
            "exp_id": cell.id,
            "stage": cell.stage,
            "backbone": cell.backbone,
            "norm": cell.norm,
            "aug": cell.aug,
            "best_val_f1": round(best_val_f1, 4),
            "test_id_f1": round(id_res["macro_f1"], 4),
            "test_ood_f1": round(ood_res["macro_f1"], 4),
            "delta_f1": delta_f1,
            "rr_f1": rr_f1,
            "test_id_acc": round(id_res["accuracy"], 4),
            "test_ood_acc": round(ood_res["accuracy"], 4),
            "minutes": trained_min,
        }

        # In kết quả hoàn thành thí nghiệm
        print(
            f"[{cell.id} Result] ID F1={result['test_id_f1']} | "
            f"OOD F1={result['test_ood_f1']} | Delta F1={delta_f1} | "
            f"RR={rr_f1}% ({trained_min}m)"
        )
        return result


def train_experiment(
    exp: Dict[str, str],
    splits: Dict[str, Any],
    ref_img_path: Union[str, Path],
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_str: str = "cuda",
) -> Dict[str, Any]:
    """Hàm bọc thủ tục hỗ trợ tương thích trực tiếp với interface của run_experiments.py."""
    trainer = ExperimentTrainer()
    return trainer.train(
        exp=exp,
        splits=splits,
        ref_img_path=ref_img_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device_str=device_str,
    )
