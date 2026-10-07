"""Giao diện dòng lệnh (CLI) phân tích tham số và khởi chạy hệ thống nghiên cứu."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module phân tích cú pháp dòng lệnh CLI
import argparse
# Lớp thao tác đường dẫn
from pathlib import Path
# Nhập các schema cấu hình
from src.config.schemas import PipelineConfig, SplitConfig, TrainingConfig
# Nhập bộ điều phối pipeline
from src.pipeline import ExperimentPipeline


def parse_args() -> PipelineConfig:
    """Phân tích các đối số dòng lệnh thành đối tượng PipelineConfig có kiểu dữ liệu an toàn."""
    # Khởi tạo parser với bộ định dạng hiển thị giá trị mặc định trực quan
    parser = argparse.ArgumentParser(
        description="Run 13-cell Histopathology Staining Robustness Ablations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Tham số đường dẫn thư mục tập nguồn nội miền
    parser.add_argument(
        "--data-root",
        default="data/raw/NCT-CRC-HE-100K-NONORM",
        help="Path to in-domain source dataset",
    )
    # Tham số đường dẫn thư mục tập đích ngoại viện OOD
    parser.add_argument(
        "--target-root",
        default="data/raw/CRC-VAL-HE-7K",
        help="Path to out-of-domain target dataset",
    )
    # Tham số đường dẫn thư mục xuất kết quả
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for output metrics and tables",
    )
    # Tham số kích thước tập mẫu con lấy mẫu phân tầng
    parser.add_argument(
        "--subset",
        type=int,
        default=25000,
        help="Subset size (default 25000; 0 for full dataset)",
    )
    # Tham số số epoch huấn luyện cho mỗi thí nghiệm
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs per experiment",
    )
    # Tham số kích thước lô nạp mẫu batch size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    # Tham số tốc độ học cơ sở
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate for CNN models",
    )
    # Tham số lựa chọn thiết bị phần cứng
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device ('cuda', 'cpu', or 'auto')",
    )
    # Tham số lọc danh sách mã thí nghiệm cần chạy
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Filter specific experiment IDs to run (e.g. EXP-01 EXP-02)",
    )
    # Tham số hạt giống ngẫu nhiên đảm bảo tái lập
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for data splitting and reproducibility",
    )
    # Cờ chạy thử nghiệm Dry Run kiểm tra kế hoạch
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments and configuration without running training",
    )

    # Thực hiện phân tích các đối số truyền vào
    args = parser.parse_args()

    # Tạo đối tượng cấu hình phân chia tập dữ liệu
    split_cfg = SplitConfig(
        source_dir=Path(args.data_root),
        target_dir=Path(args.target_root),
        out_dir=Path(args.out_dir) / "splits",
        subset_size=args.subset,
        seed=args.seed,
    )

    # Tạo đối tượng cấu hình huấn luyện
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    # Đóng gói thành cấu hình tổng thể PipelineConfig
    return PipelineConfig(
        split=split_cfg,
        training=training_cfg,
        out_dir=Path(args.out_dir),
        experiment_ids=args.experiments,
        dry_run=args.dry_run,
    )


def main() -> None:
    """Điểm nhập ứng dụng CLI khi chạy từ terminal."""
    # Nạp cấu hình từ các tham số dòng lệnh
    config = parse_args()
    # Khởi tạo đối tượng đường ống pipeline
    pipeline = ExperimentPipeline(config)
    # Thực thi pipeline
    pipeline.run()


# Chạy hàm main nếu tệp được thực thi trực tiếp
if __name__ == "__main__":
    main()
