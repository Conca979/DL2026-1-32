# Architecture & Modularization Roadmap (Isolated `src/`)

Tài liệu hướng dẫn kỹ thuật và lộ trình Sprint để mô-đun hóa codebase thành một package độc lập và tự đóng gói hoàn toàn trong thư mục `src/`. Thư mục `src/` chỉ giao tiếp ra ngoài duy nhất với thư mục dữ liệu `data/`.

---

## 1. Cấu trúc thư mục độc lập (`src/`)

```text
repo_root/
├── data/                          # Thư mục dữ liệu bên ngoài (Read-Only)
│   └── raw/
│       ├── NCT-CRC-HE-100K-NONORM/
│       └── CRC-VAL-HE-7K/
├── results/                       # Thư mục lưu kết quả xuất ra
└── src/                           # [ISOLATED APPLICATION ROOT]
    ├── README.md                  # Tài liệu hướng dẫn Sprint này
    ├── run_main.py                # [Entrypoint] File chạy chính thay cho run_experiments.py
    └── histo_robust/              # Core package Python
        ├── __init__.py
        ├── config.py              # [Sprint 0] Hằng số, danh sách 9 classes, bảng 13 experiments
        ├── data/                  # [Member 1] Pipeline đọc data và chia split
        │   ├── __init__.py
        │   ├── dataset.py         # Class HistologyDataset(Dataset)
        │   └── splits.py          # find_class_images, prepare_dataset_splits
        ├── transforms/            # [Member 2] Chuẩn hóa màu & Data Augmentation
        │   ├── __init__.py
        │   ├── normalization.py   # Reinhard (CIELAB) & Macenko (Beer-Lambert SVD)
        │   ├── augmentation.py    # HED biological stain jitter
        │   └── pipeline.py        # Class PatchTransform
        ├── models/                # [Member 3] Kiến trúc mạng & Đo lường
        │   ├── __init__.py
        │   ├── backbones.py       # build_model (ResNet-50, ConvNeXt-Tiny, Phikon ViT-B)
        │   └── metrics.py         # evaluate_model (Macro-F1, Balanced Acc, Accuracy)
        └── engine/                # [Member 4] Vòng lặp huấn luyện & CLI Runner
            ├── __init__.py
            ├── trainer.py         # train_experiment (AMP, scheduler, deepcopy checkpoint)
            └── runner.py          # CLI argparse, orchestration logic
```

### Nguyên lý thiết kế `run_main.py`
File `src/run_main.py` là điểm kích hoạt duy nhất bên trong `src/`:
```python
import sys
from pathlib import Path

# Đảm bảo src/ luôn nằm trong Python path bất kể chạy từ thư mục nào
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from histo_robust.engine.runner import main

if __name__ == "__main__":
  main()
```

---

## 2. Phân chia trách nhiệm thành viên (Team Matrix)

| Thành viên | Thư mục phụ trách | Branch Git | Nhiệm vụ chính |
|:---:|---|---|---|
| **Ha Dang Huy** | `src/histo_robust/data/` | `feature/data-pipeline` | Dataset splits, stratified sampling 24.993 ảnh, dataset loader (chỉ đọc từ `data/`). |
| **Ta Huy** | `src/histo_robust/transforms/` | `feature/transforms-stain` | Toán học Reinhard, Macenko, HED stain jitter, pipeline PatchTransform. |
| **Anh Duong** | `src/histo_robust/models/` | `feature/models-eval` | Backbones (ResNet, ConvNeXt, Phikon linear probe) & metric evaluation. |
| **Minh Duc** | `src/histo_robust/engine/` + `config.py` + `run_main.py` | `feature/engine-runner` | Setup khung ban đầu, training loop, CLI parser, tạo `src/run_main.py`, merge & nghiệm thu. |

---

## 3. Lộ trình Sprint thực hiện (Sprint Execution Sequence)

### 🚀 Sprint 0: Contract Setup & Scaffolding (Member 4 - Lead | ~15 phút)
*Mục tiêu: Dựng bộ khung độc lập trong `src/` và chốt giao diện (Interfaces) trên nhánh `main`.*

- [ ] Tạo toàn bộ các thư mục con trong `src/histo_robust/` và file `__init__.py`.
- [ ] Viết file `src/histo_robust/config.py`:
  - Khai báo `CLASSES`, `CLASS_TO_IDX`, `NUM_CLASSES`.
  - Khai báo `IMAGENET_MEAN`, `IMAGENET_STD`.
  - Khai báo mảng 13 thí nghiệm `EXPERIMENTS`.
- [ ] Tạo file `src/run_main.py` (khung gọi hàm).
- [ ] Commit và push trực tiếp lên nhánh `main`:
  ```bash
  git checkout main
  git add src/
  git commit -m "chore(scaffold): initialize isolated src/ structure, run_main.py and config"
  git push origin main
  ```

---

### ⚡ Sprint 1: Parallel Development (Tất cả 4 thành viên - ~45-60 phút)
*Mục tiêu: 4 bạn rẽ nhánh riêng, code độc lập hoàn toàn trong `src/`.*

#### Member 1 (`feature/data-pipeline`):
```bash
git checkout main && git pull
git checkout -b feature/data-pipeline
```
- [ ] Chuyển `SUPPORTED_IMAGE_EXTS` và `find_class_images()` vào `src/histo_robust/data/splits.py`.
- [ ] Chuyển `prepare_dataset_splits()` vào `src/histo_robust/data/splits.py` (đọc đường dẫn tới `data/`).
- [ ] Chuyển `HistologyDataset` vào `src/histo_robust/data/dataset.py`.
- [ ] Export các hàm trong `src/histo_robust/data/__init__.py`.

#### Member 2 (`feature/transforms-stain`):
```bash
git checkout main && git pull
git checkout -b feature/transforms-stain
```
- [ ] Chuyển ma trận Ruderman, `_rgb_to_lab`, `_lab_to_rgb`, `reinhard_fit`, `reinhard_apply`, `macenko_fit`, `macenko_apply` vào `src/histo_robust/transforms/normalization.py`.
- [ ] Chuyển `HE_STAIN_MATRIX` và `hed_stain_jitter()` vào `src/histo_robust/transforms/augmentation.py`.
- [ ] Chuyển `PatchTransform` vào `src/histo_robust/transforms/pipeline.py`.
- [ ] Export các hàm trong `src/histo_robust/transforms/__init__.py`.

#### Member 3 (`feature/models-eval`):
```bash
git checkout main && git pull
git checkout -b feature/models-eval
```
- [ ] Chuyển `PhikonClassifier` và `build_model()` vào `src/histo_robust/models/backbones.py`.
- [ ] Chuyển `evaluate_model()` vào `src/histo_robust/models/metrics.py`.
- [ ] Export các hàm trong `src/histo_robust/models/__init__.py`.

#### Member 4 (`feature/engine-runner`):
```bash
git checkout main && git pull
git checkout -b feature/engine-runner
```
- [ ] Chuyển `train_experiment()` vào `src/histo_robust/engine/trainer.py`.
- [ ] Chuyển `main()` và `argparse` vào `src/histo_robust/engine/runner.py`.
- [ ] Hoàn thiện `src/run_main.py` để làm entrypoint chính của package.
- [ ] Export trong `src/histo_robust/engine/__init__.py`.

---

### 🔀 Sprint 2: Code Review & Merging (Toàn đội - ~20 phút)
*Mục tiêu: Đưa toàn bộ code về nhánh `main`.*

- [ ] Cả 4 bạn push branch lên GitHub và mở Pull Request (PR):
  ```bash
  git push origin <tên-branch>
  ```
- [ ] Thứ tự merge khuyến nghị:
  1. Merge PR của Member 1 (`feature/data-pipeline`) -> `main`.
  2. Merge PR của Member 2 (`feature/transforms-stain`) -> `main`.
  3. Merge PR của Member 3 (`feature/models-eval`) -> `main`.
  4. Merge PR của Member 4 (`feature/engine-runner`) -> `main`.

---

### ✅ Sprint 3: Verification & Acceptance Testing (Toàn đội - ~15 phút)
*Mục tiêu: Kiểm tra độ độc lập của `src/run_main.py` khi chỉ đọc từ `data/`.*

1. **Test 1: Dry-run kiểm tra CLI (mất 2 giây):**
   ```bash
   python src/run_main.py --dry-run
   ```
   *Yêu cầu chấp nhận:* In ra toàn bộ danh sách 13 thí nghiệm mà không có bất kỳ lỗi `ImportError` nào.

2. **Test 2: Smoke test 1 epoch rút gọn:**
   ```bash
   python src/run_main.py \
     --data-root data/raw/NCT-CRC-HE-100K-NONORM \
     --target-root data/raw/CRC-VAL-HE-7K \
     --out-dir results \
     --subset 180 \
     --epochs 1 \
     --experiments EXP-01
   ```
   *Yêu cầu chấp nhận:*
   - `src/` chỉ đọc ảnh từ `data/raw/...`.
   - Kết quả xuất ra đúng thư mục `--out-dir results`.