"""Động cơ đánh giá mô hình: tính toán Macro-F1, Balanced Accuracy, Accuracy và AUROC."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module ghi log
import logging
# Các kiểu dữ liệu phụ trợ
from typing import Any, Dict, Union
# Thư viện mảng số học NumPy
import numpy as np

# Khởi tạo logger
logger = logging.getLogger(__name__)


class Evaluator:
    """Lớp đánh giá chất lượng mô hình trên các phân tập nội miền hoặc ngoại viện."""

    @staticmethod
    def evaluate(
        model: Any,
        loader: Any,
        device: Any,
    ) -> Dict[str, float]:
        """Tính toán các chỉ số phân loại lâm sàng đa lớp trên một DataLoader.

        Tham số:
            model: Mô hình PyTorch cần đánh giá.
            loader: Bộ nạp dữ liệu DataLoader cung cấp các batch (ảnh, nhãn).
            device: Thiết bị tính toán ('cuda' hoặc 'cpu').

        Trả về:
            Từ điển chứa các chỉ số 'macro_f1', 'balanced_acc', 'accuracy', và 'auroc'.
        """
        # Nhập các thư viện cần thiết tại thời điểm chạy
        try:
            import torch
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
        except ImportError as err:
            raise ImportError(
                "torch and scikit-learn are required for evaluation: "
                "pip install torch scikit-learn"
            ) from err

        # Xác định đối tượng thiết bị torch.device
        target_device = torch.device(device) if isinstance(device, str) else device
        # Chuyển mô hình sang chế độ đánh giá eval (tắt dropout, cố định batchnorm)
        model.eval()

        # Danh sách lưu trữ kết quả dự đoán, xác suất và nhãn thực tế
        all_preds = []
        all_probs = []
        all_targets = []
        # Kiểm tra xem thiết bị có phải là GPU CUDA hay không
        is_cuda = target_device.type == "cuda"

        # Tắt cơ chế tính đạo hàm autograd
        with torch.no_grad():
            # Duyệt qua từng batch dữ liệu
            for images, targets in loader:
                # Đẩy batch ảnh lên thiết bị tính toán
                images = images.to(target_device, non_blocking=True)
                # Tự động chuyển đổi kiểu dữ liệu AMP fp16 nếu đang chạy trên CUDA
                if is_cuda:
                    with torch.amp.autocast("cuda"):
                        logits = model(images)
                else:
                    logits = model(images)

                # Tính xác suất bằng hàm softmax
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                # Lưu ma trận xác suất
                all_probs.append(probs)
                # Lấy chỉ số có xác suất cao nhất làm lớp dự đoán
                all_preds.append(np.argmax(probs, axis=1))
                # Chuyển đổi nhãn về mảng NumPy
                all_targets.append(targets.numpy() if hasattr(targets, "numpy") else np.array(targets))

        # Nối toàn bộ nhãn thực tế thành một mảng duy nhất
        y_true = np.concatenate(all_targets)
        # Nối toàn bộ nhãn dự đoán thành một mảng duy nhất
        y_pred = np.concatenate(all_preds)
        # Nối toàn bộ ma trận xác suất
        y_prob = np.concatenate(all_probs)

        # Tính Macro-F1: trung bình không trọng số của F1-score trên cả 9 lớp mô học
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # Tính Balanced Accuracy: độ chính xác cân bằng giữa các lớp
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        # Tính Accuracy: độ chính xác tổng thể
        acc = accuracy_score(y_true, y_pred)

        # Tính Macro AUROC theo chiến lược One-vs-Rest
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception as exc:
            logger.debug("AUROC computation failed (likely single-class batch or NaNs): %s", exc)
            auroc = float("nan")

        # Trả về từ điển kết quả đo lường
        return {
            "macro_f1": float(macro_f1),
            "balanced_acc": float(bal_acc),
            "accuracy": float(acc),
            "auroc": float(auroc),
        }


def evaluate_model(
    model: Any,
    loader: Any,
    device: Any,
) -> Dict[str, float]:
    """Hàm bọc thủ tục hỗ trợ tương thích trực tiếp với interface của run_experiments.py."""
    return Evaluator.evaluate(model, loader, device)
