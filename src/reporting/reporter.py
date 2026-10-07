"""Tổng hợp kết quả, tự động sinh bảng Markdown và xuất artifact báo cáo."""

# Hỗ trợ type hinting hiện đại
from __future__ import annotations

# Module ghi tệp CSV thuần
import csv
# Module ghi log
import logging
# Lớp thao tác đường dẫn Path
from pathlib import Path
# Các kiểu dữ liệu phụ trợ
from typing import Any, Dict, List, Union
# Nhập dataclass kết quả thí nghiệm
from src.reporting.metrics import ExperimentResult

# Khởi tạo logger
logger = logging.getLogger(__name__)


class ResultsReporter:
    """Lớp gom kết quả đo lường và xuất ra các tệp bảng CSV cùng bảng Markdown trực quan."""

    def __init__(self, out_dir: Union[str, Path] = Path("results")) -> None:
        # Đường dẫn thư mục xuất artifact
        self.out_dir = Path(out_dir)
        # Tạo thư mục nếu chưa tồn tại
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        results: List[Union[ExperimentResult, Dict[str, Any]]],
        csv_filename: str = "summary_results.csv",
        md_filename: str = "RESULTS_TABLE.md",
    ) -> Any:
        """Tuần tự hóa danh sách kết quả thành tệp CSV và bảng tổng kết Markdown.

        Tham số:
            results: Danh sách các đối tượng ExperimentResult hoặc từ điển kết quả.
            csv_filename: Tên tệp bảng CSV xuất ra.
            md_filename: Tên tệp bảng tóm tắt Markdown xuất ra.

        Trả về:
            DataFrame Pandas (nếu có pandas) hoặc danh sách từ điển thô.
        """
        # Chuyển đổi danh sách đối tượng sang danh sách các từ điển thuần
        raw_list = [r.to_dict() if isinstance(r, ExperimentResult) else r for r in results]
        # Đường dẫn tệp CSV đầu ra
        csv_file = self.out_dir / csv_filename
        # Đường dẫn tệp Markdown đầu ra
        md_file = self.out_dir / md_filename

        # Thử sử dụng Pandas để xuất tệp định dạng đẹp
        try:
            import pandas as pd
            res_df = pd.DataFrame(raw_list)
            # Lưu DataFrame ra tệp CSV
            res_df.to_csv(csv_file, index=False)
            # Thử định dạng bảng Markdown bằng to_markdown()
            try:
                md_content = res_df.to_markdown(index=False)
            except Exception:
                # Nếu thiếu thư viện tabulate thì chuyển sang to_string()
                md_content = res_df.to_string(index=False)
            return_val = res_df
        except ImportError:
            # Phương án dự phòng bằng thư viện chuẩn của Python nếu chưa có Pandas
            if raw_list:
                keys = list(raw_list[0].keys())
                # Ghi tệp CSV thuần
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(raw_list)

                # Tự dựng cấu trúc bảng Markdown
                headers = "| " + " | ".join(keys) + " |"
                sep = "| " + " | ".join(["---"] * len(keys)) + " |"
                rows = [
                    "| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"
                    for r in raw_list
                ]
                md_content = "\n".join([headers, sep] + rows)
            else:
                md_content = "No results recorded."
            return_val = raw_list

        # Ghi nội dung bảng Markdown ra tệp trên đĩa
        md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")

        # In đường kẻ phân cách thẩm mỹ ra màn hình console
        print("\n" + "=" * 80)
        print("FINAL ABLATION RESULTS TABLE")
        print("=" * 80)
        # In toàn bộ nội dung bảng kết quả
        print(md_content)
        # Thông báo đường dẫn các artifact vừa được tạo
        print(f"\n[Artifacts] Wrote {csv_file} and {md_file}")

        return return_val
