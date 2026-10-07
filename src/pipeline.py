"""Lớp điều phối trung tâm (Pipeline Orchestrator) quản lý toàn bộ chu trình thí nghiệm."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module ghi log
import logging
# Lớp thao tác đường dẫn
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import List, Optional

# Nhập schema cấu hình đường ống
from src.config.schemas import PipelineConfig
# Nhập hàm lọc danh sách thí nghiệm
from src.config.experiments import filter_experiments
# Nhập bộ chia tập dữ liệu
from src.data.splitter import DatasetSplitter
# Nhập động cơ huấn luyện
from src.engine.trainer import ExperimentTrainer
# Nhập bộ tạo báo cáo
from src.reporting.reporter import ResultsReporter

# Khởi tạo logger
logger = logging.getLogger(__name__)


class ExperimentPipeline:
    """Lớp điều phối từ đầu đến cuối: chuẩn bị dữ liệu, chạy ma trận thí nghiệm và xuất báo cáo."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        # Nạp cấu hình đường ống (mặc định nếu không truyền)
        self.config = config or PipelineConfig()
        # Khởi tạo đối tượng chia tập dữ liệu
        self.splitter = DatasetSplitter(self.config.split)
        # Khởi tạo đối tượng huấn luyện
        self.trainer = ExperimentTrainer(self.config.training)
        # Khởi tạo đối tượng xuất báo cáo
        self.reporter = ResultsReporter(self.config.out_dir)

    def run(self) -> None:
        """Thực thi toàn bộ quy trình thí nghiệm theo cấu hình đã thiết lập."""
        # Chuyển thư mục kết quả thành đối tượng Path
        out_path = Path(self.config.out_dir)
        # Tạo thư mục nếu chưa tồn tại
        out_path.mkdir(parents=True, exist_ok=True)

        # Lọc danh sách các cell thí nghiệm cần thực thi
        exp_list = filter_experiments(self.config.experiment_ids)

        # Nếu đang ở chế độ Dry Run, in kế hoạch và kết thúc sớm
        if self.config.dry_run:
            print(
                f"Plan: {len(exp_list)} experiments queued "
                f"(subset={self.config.split.subset_size}, "
                f"epochs={self.config.training.epochs}, "
                f"batch={self.config.training.batch_size})"
            )
            # In chi tiết từng ô thí nghiệm trong hàng đợi
            for e in exp_list:
                print(
                    f"  {e.id} | {e.stage} | backbone={e.backbone:<13} | "
                    f"norm={e.norm:<8} | aug={e.aug:<12} | {e.notes}"
                )
            return

        # Kiểm tra tính sẵn sàng của GPU CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            cuda_available = False

        # Xác định thiết bị thực thi
        device = (
            self.config.training.device
            if self.config.training.device != "auto"
            else ("cuda" if cuda_available else "cpu")
        )
        # In thông báo khởi động hệ thống
        print(f"Running on device: {device} | Total experiments: {len(exp_list)}")

        # Đường dẫn lưu các tệp phân tách CSV
        splits_dir = out_path / "splits"
        # Thực hiện quét dữ liệu và tạo 4 phân tập dữ liệu
        splits, ref_tile = self.splitter.run(
            source_dir=self.config.split.source_dir,
            target_dir=self.config.split.target_dir,
            out_dir=splits_dir,
            subset_size=self.config.split.subset_size,
            seed=self.config.split.seed,
        )

        # Danh sách gom kết quả đo lường
        results = []
        # Duyệt tuần tự và huấn luyện từng ô thí nghiệm
        for exp in exp_list:
            res = self.trainer.train(
                exp=exp,
                splits=splits,
                ref_img_path=ref_tile,
                epochs=self.config.training.epochs,
                batch_size=self.config.training.batch_size,
                device_str=device,
            )
            results.append(res)

        # Xuất bảng tổng hợp kết quả ra tệp CSV và bảng Markdown
        self.reporter.export(results)
