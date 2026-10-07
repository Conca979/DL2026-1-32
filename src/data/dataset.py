"""Hiện thực hóa lớp Dataset của PyTorch cho các mảnh mô bệnh học."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Các kiểu dữ liệu phụ trợ
from typing import Any, Callable, Tuple, TYPE_CHECKING
# Thư viện xử lý hình ảnh Pillow
from PIL import Image

# Nhập pandas phục vụ kiểm tra kiểu tĩnh
if TYPE_CHECKING:
    import pandas as pd

# Nhập an toàn PyTorch Dataset (với fallback lớp cơ sở nếu môi trường chưa cài PyTorch)
try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset:  # type: ignore
        """Lớp cơ sở dự phòng khi chưa cài đặt PyTorch trong môi trường tĩnh."""
        pass


class HistologyDataset(Dataset):
    """Lớp Dataset nạp các mảnh patch mô học và nhãn số nguyên tương ứng."""

    def __init__(self, df: Any, transform: Callable[[Image.Image], Any]) -> None:
        """Khởi tạo tập dữ liệu từ bảng Pandas DataFrame.

        Tham số:
            df: DataFrame chứa ít nhất hai cột 'image_path' và 'label_idx'.
            transform: Hàm callable thực hiện chuẩn hóa màu, augment và chuyển đổi tensor.
        """
        # Đặt lại chỉ mục index từ 0 để đảm bảo truy cập liên tục không bị lỗi ngắt quãng
        self.df = df.reset_index(drop=True)
        # Lưu lại hàm biến đổi hình ảnh
        self.transform = transform

    def __len__(self) -> int:
        """Trả về tổng số lượng mẫu ảnh có trong tập dữ liệu."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Truy xuất một mẫu dữ liệu tại vị trí chỉ số idx.

        Tham số:
            idx: Chỉ số nguyên của mẫu dữ liệu.

        Trả về:
            Bộ đôi (Tensor ảnh đã qua biến đổi, Chỉ số nhãn lớp nguyên).
        """
        # Lấy bản ghi thông tin dòng tại chỉ số idx
        row = self.df.iloc[idx]
        # Mở tệp ảnh từ đường dẫn đĩa cứng
        img = Image.open(row["image_path"])
        # Áp dụng chuỗi biến đổi PatchTransform và ép kiểu nhãn về số nguyên int
        return self.transform(img), int(row["label_idx"])
