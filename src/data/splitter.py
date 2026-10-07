"""Phân tách tập dữ liệu phân tầng và trích xuất ảnh tham chiếu chuẩn hóa màu."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module ghi log
import logging
# Module sinh số ngẫu nhiên
import random
# Lớp thao tác đường dẫn Path
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
# Thư viện mảng số học NumPy
import numpy as np
# Thư viện ảnh Pillow
from PIL import Image

# Nhập danh sách lớp mô học và phần mở rộng ảnh được hỗ trợ
from src.config.constants import CLASSES, NUM_CLASSES, SUPPORTED_IMAGE_EXTS
# Nhập dataclass cấu hình phân tách
from src.config.schemas import SplitConfig
# Nhập hàm tiện ích tìm ảnh
from src.data.discovery import find_class_images

# Kiểm tra kiểu dữ liệu tĩnh cho pandas
if TYPE_CHECKING:
    import pandas as pd

# Khởi tạo logger riêng cho module splitter
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Quản lý việc quét ảnh, phân tách phân tầng 70/15/15 và cô lập ảnh tham chiếu."""

    def __init__(self, config: Optional[SplitConfig] = None) -> None:
        # Nạp cấu hình phân tách mặc định nếu không truyền riêng
        self.config = config or SplitConfig()

    def run(
        self,
        source_dir: Optional[Path] = None,
        target_dir: Optional[Path] = None,
        out_dir: Optional[Path] = None,
        subset_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        """Thực thi quy trình quét, phân tầng dữ liệu và xuất các tệp CSV phân tập."""
        # Nhập pandas tại thời điểm thực thi để an toàn môi trường
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError(
                "pandas is required for dataset splitting: pip install pandas"
            ) from err

        # Lấy các tham số cấu hình tương ứng
        src_path = Path(source_dir or self.config.source_dir)
        tgt_path = Path(target_dir or self.config.target_dir)
        output_dir = Path(out_dir or self.config.out_dir)
        sub_size = self.config.subset_size if subset_size is None else subset_size
        rnd_seed = self.config.seed if seed is None else seed

        # Tạo thư mục đầu ra nếu chưa có
        output_dir.mkdir(parents=True, exist_ok=True)
        # Thiết lập hạt giống ngẫu nhiên cho NumPy
        np.random.seed(rnd_seed)
        # Thiết lập hạt giống ngẫu nhiên cho random
        random.seed(rnd_seed)

        # 1. Thu thập danh sách ảnh thuộc miền nguồn (In-Domain Source)
        source_records: List[Dict[str, object]] = []
        # Duyệt qua từng lớp mô học trong 9 lớp chuẩn
        for c_idx, c_name in enumerate(CLASSES):
            # Quét tìm các ảnh của lớp trong thư mục nguồn
            c_images = find_class_images(src_path, c_name)
            # Thêm từng ảnh vào danh sách bản ghi
            for p in c_images:
                source_records.append({
                    "image_path": str(p),
                    "class_name": c_name,
                    "label_idx": c_idx,
                    "domain": "source",
                })

        # Tạo DataFrame nguồn
        source_df = pd.DataFrame(source_records)
        # Kiểm tra tính hợp lệ nếu thư mục rỗng
        if len(source_df) == 0:
            items = (
                [p.name for p in list(src_path.iterdir())[:15]]
                if src_path.is_dir()
                else "directory does not exist"
            )
            raise RuntimeError(
                f"No source images found under {src_path}.\n"
                f"Contents found at {src_path}: {items}.\n"
                f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
            )

        # 2. Lấy mẫu tập con phân tầng nếu có yêu cầu (subset_size > 0)
        if 0 < sub_size < len(source_df):
            # Số lượng mẫu đồng đều cho mỗi lớp
            per_class = sub_size // NUM_CLASSES
            # Nhóm theo tên lớp và lấy mẫu ngẫu nhiên có hạt giống
            source_df = (
                source_df.groupby("class_name", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=rnd_seed))
                .reset_index(drop=True)
            )

        # 3. Phân tách phân tầng 70 / 15 / 15 bằng scikit-learn
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as err:
            raise ImportError(
                "scikit-learn is required for dataset splitting: pip install scikit-learn"
            ) from err

        # Chia 70% train và 30% rest có phân tầng theo cột class_name
        train_df, rest_df = train_test_split(
            source_df,
            test_size=0.30,
            stratify=source_df["class_name"],
            random_state=rnd_seed,
        )
        # Chia đều 30% rest thành 15% val_id và 15% test_id
        val_df, test_id_df = train_test_split(
            rest_df,
            test_size=0.50,
            stratify=rest_df["class_name"],
            random_state=rnd_seed,
        )

        # 4. Thu thập toàn bộ ảnh thuộc miền đích ngoại viện độc lập (100% Out-of-Domain Cohort)
        target_records: List[Dict[str, object]] = []
        for c_idx, c_name in enumerate(CLASSES):
            c_images = find_class_images(tgt_path, c_name)
            for p in c_images:
                target_records.append({
                    "image_path": str(p),
                    "class_name": c_name,
                    "label_idx": c_idx,
                    "domain": "target",
                })

        # Tạo DataFrame ngoại miền OOD
        test_ood_df = pd.DataFrame(target_records)
        # Kiểm tra tính hợp lệ nếu thư mục đích rỗng
        if len(test_ood_df) == 0:
            items = (
                [p.name for p in list(tgt_path.iterdir())[:15]]
                if tgt_path.is_dir()
                else "directory does not exist"
            )
            raise RuntimeError(
                f"No target images found under {tgt_path}.\n"
                f"Contents found at {tgt_path}: {items}.\n"
                f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
            )

        # 5. Lưu 4 phân tập ra các tệp CSV riêng biệt
        splits: Dict[str, Any] = {
            "train": train_df,
            "val_id": val_df,
            "test_id": test_id_df,
            "test_ood": test_ood_df,
        }
        for name, df in splits.items():
            df.to_csv(output_dir / f"{name}.csv", index=False)

        # 6. Trích xuất tile ảnh tham chiếu chuẩn nghiêm ngặt từ tập train thuộc lớp u TUM
        tum_train = train_df[train_df["class_name"] == "TUM"]
        if tum_train.empty:
            raise RuntimeError("Training split has no 'TUM' class instances for reference selection.")

        # Lấy ảnh biểu mô u đầu tiên trong tập train
        ref_path = Path(tum_train.iloc[0]["image_path"])
        # Mở ảnh RGB
        ref_img = Image.open(ref_path).convert("RGB")
        # Đường dẫn tệp lưu trữ
        ref_save_path = output_dir / "reference_stain.png"
        # Lưu ảnh PNG
        ref_img.save(ref_save_path)

        # Ghi log số lượng mẫu của các phân tập
        logger.info(
            "[splits] Train=%d | Val=%d | Test-ID=%d | Test-OOD=%d",
            len(train_df), len(val_df), len(test_id_df), len(test_ood_df)
        )
        logger.info("[reference] Canonical stain reference: %s", ref_path.name)
        # Trả về từ điển các phân tập và đường dẫn ảnh tham chiếu
        return splits, ref_save_path


def prepare_dataset_splits(
    source_dir: Path,
    target_dir: Path,
    out_dir: Path,
    subset_size: int = 25000,
    seed: int = 100,
) -> Tuple[Dict[str, Any], Path]:
    """Hàm bọc thủ tục hỗ trợ tương thích trực tiếp với interface của run_experiments.py."""
    splitter = DatasetSplitter()
    return splitter.run(
        source_dir=source_dir,
        target_dir=target_dir,
        out_dir=out_dir,
        subset_size=subset_size,
        seed=seed,
    )
