"""Điểm thực thi gói khi người dùng gõ lệnh `python -m src`."""

# Nhập hàm main từ module giao diện dòng lệnh cli
from src.cli import main

# Kiểm tra nếu gói được gọi trực tiếp như một module thực thi
if __name__ == "__main__":
    # Khởi chạy hàm điều phối chính
    main()
