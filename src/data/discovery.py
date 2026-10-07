"""Các hàm tiện ích quét và định vị hình ảnh mô bệnh học từ cây thư mục ổ đĩa."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Lớp Path xử lý đường dẫn tệp tin
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import List, Set
# Nhập tập hợp đuôi tệp tin ảnh hợp lệ
from src.config.constants import SUPPORTED_IMAGE_EXTS


def find_class_images(
    base_dir: Path,
    class_name: str,
    supported_exts: Set[str] = SUPPORTED_IMAGE_EXTS,
) -> List[Path]:
    """Tìm kiếm tất cả các tệp ảnh thuộc một lớp mô học, xử lý cả chữ hoa/thường và thư mục con lồng nhau.

    Tham số:
        base_dir: Thư mục gốc chứa dữ liệu (ví dụ: NCT-CRC-HE-100K-NONORM hoặc CRC-VAL-HE-7K).
        class_name: Tên nhãn lớp mô học (ví dụ: 'TUM', 'NORM').
        supported_exts: Tập hợp các phần mở rộng tệp tin được chấp nhận.

    Trả về:
        Danh sách các đối tượng Path được sắp xếp theo thứ tự bảng chữ cái.
    """
    # Nếu đường dẫn gốc không tồn tại hoặc không phải là thư mục, trả về danh sách rỗng
    if not base_dir.exists() or not base_dir.is_dir():
        return []

    # Khởi tạo biến lưu thư mục của lớp
    c_folder = None

    # 1. Kiểm tra trực tiếp khớp tên thư mục con (chính xác hoặc không phân biệt hoa thường)
    if (base_dir / class_name).is_dir():
        c_folder = base_dir / class_name
    else:
        # Duyệt qua các mục con cấp 1 kiểm tra tên viết hoa
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.upper() == class_name.upper():
                c_folder = child
                break

    # 2. Nếu chưa tìm thấy ở cấp 1, quét đệ quy rglob sâu hơn qua toàn bộ cây thư mục
    if c_folder is None:
        for child in base_dir.rglob("*"):
            if child.is_dir() and child.name.upper() == class_name.upper():
                c_folder = child
                break

    # Nếu vẫn không xác định được thư mục chứa lớp, trả về danh sách rỗng
    if c_folder is None or not c_folder.is_dir():
        return []

    # Thu thập tất cả các tệp tin trong thư mục có phần mở rộng hợp lệ
    images = [
        p for p in c_folder.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    ]

    # Phương án dự phòng quét đệ quy nếu các patch ảnh bị lồng thêm một cấp thư mục con
    if not images:
        images = [
            p for p in c_folder.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_exts
        ]

    # Trả về danh sách đường dẫn đã được sắp xếp tăng dần
    return sorted(images)
