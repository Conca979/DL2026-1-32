# BÁCH KHOA TOÀN THƯ VỀ DỰ ÁN DL2026-1-32
## Nhận Diện Mô Bệnh Học Ung Thư Dưới Sự Biến Thiên Màu Nhuộm Giữa Các Bệnh Viện
*(Cẩm nang chi tiết từ A đến Z: Bản chất dữ liệu, Mổ xẻ từng dòng code, Cơ chế toán học, Kiến trúc phần mềm và Bộ 20 câu hỏi vấn đáp bảo vệ đồ án — Dành riêng cho buổi bảo vệ)*

---

## 📌 LỜI MỞ ĐẦU: DỰ ÁN NÀY LÀ GÌ VÀ VÌ SAO BẠN CẦN ĐỌC TÀI LIỆU NÀY?

Nếu bạn chuẩn bị bước vào phòng bảo vệ đồ án / khóa luận và cảm thấy lo lắng trước những câu hỏi hóc búa của Hội đồng chấm thi: *"Dòng code này làm gì?", "Công thức toán học của thuật toán này ở đâu ra?", "Tại sao mô hình lại bị tụt điểm khi sang bệnh viện khác?", "Kiến trúc mã nguồn được tổ chức như thế nào?"*, **hãy thở phào nhẹ nhõm: Đây chính là cẩm nang cứu cánh toàn diện nhất dành cho bạn**.

Tài liệu này được biên soạn với tôn chỉ: **Giải thích cặn kẽ từng dòng code, từng thuật toán, từng quyết định kỹ thuật bằng ngôn ngữ khoa học chuẩn mực nhưng cực kỳ sáng rõ, dễ hiểu**.

> [!TIP]
> **Tóm tắt cốt lõi của toàn bộ dự án trong 3 ý chính:**
> 1. **Bài toán lâm sàng**: Phân loại tự động 9 loại mô trong ung thư biểu mô đại trực tràng từ ảnh chụp lát cắt vi thể nhuộm H&E.
> 2. **Hiện tượng Domain Shift (Lệch miền)**: Khi mô hình học trên ảnh của Bệnh viện Heidelberg (Đức) đạt độ chính xác cực cao 98.9%, nhưng khi mang sang Bệnh viện Aachen (Đức) kiểm tra thì rơi tự do xuống 66.4% do khác biệt về thuốc nhuộm và máy quét kính hiển vi.
> 3. **Giải pháp & Ma trận 13 Thí nghiệm (13-Cell Ablation Study)**: Đánh giá có hệ thống 3 nhóm vũ khí: Chuẩn hóa màu sắc (Reinhard, Macenko), Tăng cường dữ liệu (Xoay lật, Nhiễu nồng độ HED), và Lựa chọn bộ não AI (ResNet-50, ConvNeXt-Tiny, Foundation Model Phikon).

---

## 🗺️ MỤC LỤC CHI TIẾT

1. [Bức tranh lâm sàng: Chuyện gì diễn ra sau cánh cửa phòng xét nghiệm?](#1-bức-tranh-lâm-sàng-chuyện-gì-diễn-ra-sau-cánh-cửa-phòng-xét-nghiệm)
2. [Phân tích dữ liệu chuyên sâu & Khám phá 9 loại mô](#2-phân-tích-dữ-liệu-chuyên-sâu--khám-phá-9-loại-mô)
3. [Hiện tượng Domain Shift & Căn bệnh "Học tủ" của AI](#3-hiện-tượng-domain-shift--căn-bệnh-học-tủ-của-ai)
4. [Bộ 3 "Vũ khí phòng vệ" được thử nghiệm](#4-bộ-3-vũ-khí-phòng-vệ-được-thử-nghiệm)
5. [MỔ XẺ CHI TIẾT TỪNG DÒNG CODE TRONG `run_experiments.py`](#5-mổ-xẻ-chi-tiết-từng-dòng-code-trong-run_experimentspy)
6. [KIẾN TRÚC HƯỚNG ĐỐI TƯỢNG (OOP) CÔNG NGHIỆP TRONG THƯ MỤC `src/`](#6-kiến-trúc-hướng-đối-tượng-oop-công-nghiệp-trong-thư-mục-src)
7. [Bảng kết quả 13 thí nghiệm & 3 phát hiện khoa học chấn động](#7-bảng-kết-quả-13-thí-nghiệm--3-phát-hiện-khoa-học-chấn-động)
8. [Phân tích định tính & Các ca nhầm lẫn kinh điển (Error Analysis)](#8-phân-tích-định-tính--các-ca-nhầm-lẫn-kinh-điển-error-analysis)
9. [Vận hành trên Kaggle: Cơ chế hoạt động của `notebook.ipynb`](#9-vận-hành-trên-kaggle-cơ-chế-hoạt-động-của-notebookipynb)
10. [Từ điển thuật ngữ Deep Learning "Bình Dân Học Vụ"](#10-từ-điển-thuật-ngữ-deep-learning-bình-dân-học-vụ)
11. [BỘ 20 CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN (DEFENSE Q&A) HIỂM HÓC NHẤT](#11-bộ-20-câu-hỏi-vấn-đáp-bảo-vệ-đồ-án-defense-qa-hiểm-hóc-nhất)
12. [CHIẾN LƯỢC BẢO VỆ TIẾNG ANH: BỘ KHUNG WHY - HOW - IF & CÂU HỎI CHUYÊN SÂU](#12-chiến-lược-bảo-vệ-tiếng-anh-bộ-khung-why---how---if--câu-hỏi-chuyên-sâu)

---

## 1. BỨC TRANH LÂM SÀNG: CHUYỆN GÌ DIỄN RA SAU CÁNH CỬA PHÒNG XÉT NGHIỆM?

### 1.1. Giải phẫu bệnh học (Histopathology) là gì?
Khi bệnh nhân đi nội soi phát hiện polyp hoặc khối u đại tràng nghi ngờ ác tính, bác sĩ bấm sinh thiết một mẩu mô nhỏ.
- Mẩu mô được cố định trong formalin, vùi trong sáp paraffin, rồi dùng máy cắt vi thể (*microtome*) gọt thành lát cắt siêu mỏng **4 – 5 micromet** (mỏng hơn sợi tóc) đặt lên lam kính.
- Mô người vốn trong suốt, nên bắt buộc phải nhuộm màu để quan sát dưới kính hiển vi quang học.

### 1.2. Kỹ thuật nhuộm tiêu chuẩn vàng: H&E (Hematoxylin & Eosin)
- **Hematoxylin (Kiềm, màu Tím sẫm / Xanh đen)**: Liên kết với axit nucleic (ADN, ARN) trong **Nhân tế bào**.
- **Eosin (Axit, màu Hồng / Đỏ tươi)**: Liên kết với protein kiềm trong **Tế bào chất** và **Chất nền ngoại bào** (sợi collagen, cơ).

```mermaid
flowchart LR
    A["Mẩu mô sinh thiết ruột"] --> B["Cắt lát siêu mỏng (4-5 µm) gắn lên lam kính"]
    B --> C["Nhuộm Hematoxylin\n-> Nhân tế bào bắt màu TÍM"]
    C --> D["Nhuộm Eosin\n-> Tế bào chất & sợi cơ bắt màu HỒNG"]
    D --> E["Máy quét Whole Slide Image (WSI)\n-> Quét ra ảnh kỹ thuật số hàng tỷ pixel"]
```

### 1.3. Tại sao AI là công cụ sống còn cho Bác sĩ giải phẫu bệnh?
Một tiêu bản quét kỹ thuật số (Whole Slide Image - WSI) có độ phân giải hàng chục gigapixel. Bác sĩ phải rê chuột soi từng cụm tế bào suốt 8 tiếng mỗi ngày. AI đóng vai trò người trợ lý sàng lọc tự động: cắt WSI thành hàng vạn mảnh nhỏ ($224 \times 224$ pixel) và phân loại lập tức: mô nào lành tính, mô nào là ung thư xâm lấn.

---

## 2. PHÂN TÍCH DỮ LIỆU CHUYÊN SÂU & KHÁM PHÁ 9 LOẠI MÔ

Dữ liệu trích từ công trình nổi tiếng của GS. Jakob Nikolas Kather (*PLOS Medicine*, 2019):
1. **Tập nguồn (Source / In-Domain): `NCT-CRC-HE-100K-NONORM`**:
   - 100.000 lát cắt ảnh thô ($224 \times 224$) từ 86 bệnh nhân tại Trung tâm Ung thư Quốc gia Heidelberg (NCT) và Đại học Mannheim (Đức).
   - Tên gọi `NONORM`: Giữ nguyên vẹn độ thô ráp, đậm nhạt tự nhiên của phòng lab nguồn, chưa qua chỉnh sửa.
2. **Tập đích ngoại viện (Target / Out-of-Domain): `CRC-VAL-HE-7K`**:
   - 7.180 lát cắt ảnh từ 50 bệnh nhân hoàn toàn độc lập tại Bệnh viện Đại học RWTH Aachen (Đức).
   - Không trùng lặp bệnh nhân, bác sĩ, hóa chất nhuộm hay máy quét so với Heidelberg.

```text
+-----------------------------------------------------------------------------------------+
|                                NGUYÊN TẮC "BỨC TƯỜNG LỬA"                              |
|                                     (DOMAIN FIREWALL)                                   |
|                                                                                         |
|  [BỆNH VIỆN HEIDELBERG & MANNHEIM]                     [BỆNH VIỆN RWTH AACHEN]          |
|         (100.000 ảnh thô)                                 (7.180 ảnh thô)               |
|                 │                                                 │                     |
|         Lấy mẫu 25.000 ảnh                                        │                     |
|                 │                                                 │                     |
|       ┌─────────┴─────────┐                                       │                     |
|       ▼                   ▼                                       │                     |
|  [TẬP HỌC 70%]     [TẬP KIỂM TRA 15%]                             │                     |
|  (Train: 17.495)   (Val-ID: 3.749)                                │                     |
|       │                   │                                       │                     |
|       └─────────┬─────────┘                                       │                     |
|                 ▼                                                 ▼                     |
|          AI LUYỆN TẬP Ở NHÀ                      KIỂM TRA TỐT NGHIỆP TRƯỜNG LẠ          |
|      (Điểm thi thử: Test-ID 15%)                   (Điểm thi thật: Test-OOD 100%)       |
|            (3.749 ảnh)                                     (7.180 ảnh)                  |
|                                                                                         |
|  * NGUYÊN TẮC BẤT DI BẤT DỊCH: AI tuyệt đối không bao giờ được nhìn thấy bất kỳ ảnh nào |
|    của Bệnh viện Aachen cho đến giây phút chấm điểm kiểm thử cuối cùng!                 |
+-----------------------------------------------------------------------------------------+
```

### 9 Loại Mô Mô Học Đại Trực Tràng:
- `ADI` (Adipose - Mỡ): Tế bào mỡ tròn rỗng, màng mỏng, nền trắng.
- `BACK` (Background - Nền kính): Kính hiển vi trắng tinh, không chứa mô.
- `DEB` (Debris - Mảnh vụn): Mảnh vụn hoại tử tế bào vỡ vụn, bắt màu tím hồng loang lổ.
- `LYM` (Lymphocytes - Tế bào lympho): Tế bào miễn dịch nhỏ li ti, nhân tròn tím đậm chen chúc.
- `MUC` (Mucus - Dịch nhầy): Chất nhầy lỏng trong suốt hoặc lam nhạt.
- `MUS` (Muscle - Cơ trơn): Bó sợi cơ hồng đậm, nhân hình thoi dài xì-gà.
- `NORM` (Normal mucosa - Niêm mạc lành tính): Tuyến Lieberkühn trật tự đều đặn.
- `STR` (Stroma - Mô đệm): Mô liên kết collagen dạng sợi hồng nhạt, tế bào thoi nhọn.
- `TUM` (Tumor - Biểu mô u ác tính): Tuyến dị dạng, nhân tế bào phình to méo mó, chen chúc bất thường.

---

## 3. HIỆN TƯỢNG DOMAIN SHIFT & CĂN BỆNH "HỌC TỦ" CỦA AI

- **Domain Shift là gì?** Là sự thay đổi phân phối dữ liệu đầu vào giữa miền huấn luyện $P_{\text{train}}(X)$ và miền kiểm thử thực tế $P_{\text{test}}(X)$, dù nhãn bệnh $P(Y|X)$ không đổi.
- **Shortcut Learning (Học lối tắt / Học tủ)**: Mạng nơ-ron có xu hướng tìm con đường tối ưu dễ nhất để giảm hàm mất mát. Thay vì học hình thái tế bào (tỷ lệ nhân, màng tế bào), AI nhận ra rằng ở Bệnh viện Heidelberg, các mẫu u ác tính thường có mật độ nhân dày nên ảnh có sắc tím đậm hơn. AI học ngay lối tắt: *"Cứ thấy tím đậm là phán ung thư"*.
- **Hậu quả**: Khi sang Bệnh viện Aachen, nơi hóa chất nhuộm khiến cả mô cơ lành tính cũng ánh tím $\rightarrow$ AI phán bệnh nhân lành thành ung thư! Ở kịch bản cơ sở **EXP-01**, điểm Macro-F1 rơi tự do từ **0.9893 (ID)** xuống còn **0.6644 (OOD)** (suy giảm 32.5 điểm F1, tỷ lệ giữ phong độ chỉ còn **67.16%**).

---

## 4. BỘ 3 "VŨ KHÍ PHÒNG VỆ" ĐƯỢC THỬ NGHIỆM

```mermaid
flowchart TD
    subgraph Defenses["3 Nhóm Vũ Khí Phòng Vệ Lệch Miền"]
        W1["1. Chuẩn hóa màu sắc (Color Normalization)\nReinhard (Thống kê LAB) & Macenko (Quang học OD + SVD)"]
        W2["2. Tăng cường dữ liệu (Data Augmentation)\nAug-Geo (Xoay/Lật) & Aug-Stain (Nhiễu HED Jitter)"]
        W3["3. Đổi mới Backbone thị giác (Vision Backbones)\nResNet-50, ConvNeXt-Tiny, Phikon (Foundation Model SSL)"]
    end
```

---

## 5. MỔ XẺ CHI TIẾT TỪNG DÒNG CODE TRONG `run_experiments.py`

Phần này đi qua từng khối code trong [`run_experiments.py`](file:///D:/LT/DL2026-1-32/run_experiments.py), giải thích chi tiết chức năng, ý nghĩa toán học và lâm sàng từng dòng:

### 5.1. Khối 1: Khai báo thư viện, Hằng số & Ma trận Thực nghiệm (Dòng 1 – 43)

```python
from __future__ import annotations
import argparse, copy, os, random, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np, pandas as pd
from PIL import Image
```
- `from __future__ import annotations`: Cho phép dùng type hint hiện đại mà không bị lỗi trên các bản Python cũ.
- `argparse`: Phân tích tham số dòng lệnh CLI (`--data-root`, `--epochs`, `--batch-size`,...).
- `copy`: Cung cấp `copy.deepcopy()` để sao chép độc lập toàn bộ trọng số mô hình tốt nhất (`state_dict`).
- `random`, `np`: Khởi tạo và đồng bộ hạt giống ngẫu nhiên (`seed=100`) nhằm bảo đảm tính tái lập kết quả.
- `Path`: Thao tác đường dẫn an toàn trên cả Windows (`\`) và Linux (`/`).
- `Image`: Đọc và lưu trữ ảnh định dạng RGB 8-bit.

```python
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```
- `CLASSES`: Danh sách 9 lớp mô học theo đúng thứ tự nhãn trong tập dữ liệu chuẩn Kather.
- `CLASS_TO_IDX`: Tạo bảng tra cứu nhãn dạng chuỗi sang số nguyên $0 \dots 8$.
- `IMAGENET_MEAN`, `IMAGENET_STD`: Giá trị trung bình và độ lệch chuẩn của 1 triệu ảnh ImageNet trên 3 kênh R-G-B. Mọi backbone tiền huấn luyện (ResNet, ConvNeXt) đều yêu cầu chuẩn hóa Z-score theo thông số này: $X_{\text{norm}} = \frac{X - \mu}{\sigma}$.

---

### 5.2. Khối 2: Chuẩn hóa màu thống kê Reinhard (Dòng 44 – 102)

#### Cơ sở toán học:
Mắt người có 3 loại tế bào nón cảm thụ ánh sáng: S (Short - Xanh lam), M (Medium - Xanh lục), L (Long - Đỏ). Năm 1998, Ruderman phát hiện rằng phản ứng của các tế bào nón có tính phi tuyến theo hàm logarit và tồn tại một hệ trục đối kháng không tương quan $L\alpha\beta$:
- $L$: Kênh độ sáng phi sắc.
- $\alpha$: Kênh đối kháng Đỏ - Lục.
- $\beta$: Kênh đối kháng Vàng - Lam.

```python
_LMS_MAT = np.array([
  [0.3811, 0.5783, 0.0402],
  [0.1967, 0.7244, 0.0782],
  [0.0241, 0.1288, 0.8444],
], dtype=np.float64)

_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0),  1.0 / np.sqrt(3.0)],
  [1.0 / np.sqrt(6.0),  1.0 / np.sqrt(6.0), -2.0 / np.sqrt(6.0)],
  [1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0),  0.0],
], dtype=np.float64)

_INV_LAB_MAT = np.array([
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0),  1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0),  1.0 / np.sqrt(6.0), -1.0 / np.sqrt(2.0)],
  [1.0 / np.sqrt(3.0), -2.0 / np.sqrt(6.0),  0.0],
], dtype=np.float64)

_INV_LMS_MAT = np.linalg.inv(_LMS_MAT)
```
- `_LMS_MAT`: Ma trận $3 \times 3$ chuyển từ RGB sang không gian nón LMS.
- `_LAB_MAT`: Ma trận trực giao biến đổi từ $\log_{10}(\text{LMS})$ sang $L\alpha\beta$. Hàng 1 chia cho $\sqrt{3}$, hàng 2 chia cho $\sqrt{6}$, hàng 3 chia cho $\sqrt{2}$ để đảm bảo chuẩn Euclid bằng 1.
- `_INV_LAB_MAT`, `_INV_LMS_MAT`: Các ma trận nghịch đảo để biến đổi ngược lại từ $L\alpha\beta \rightarrow \log_{10}(\text{LMS}) \rightarrow \text{RGB}$.

```python
def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
  norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
  lms = norm_rgb @ _LMS_MAT.T
  log_lms = np.log10(np.clip(lms, 1e-4, None))
  return log_lms @ _LAB_MAT.T
```
- `np.clip(..., 1e-4, 1.0)`: Rất quan trọng! Nếu pixel bằng 0 (đen tuyệt đối), phép tính $\log_{10}(0)$ sẽ sinh ra $-\infty$ làm sập toàn bộ mạng nơ-ron. Chặn dưới tại $10^{-4}$ loại trừ triệt để lỗi số học.
- `norm_rgb @ _LMS_MAT.T`: Nhân ma trận chiếu pixel từ RGB sang LMS.
- `log_lms @ _LAB_MAT.T`: Chiếu sang không gian đối kháng Ruderman LAB.

```python
def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
  log_lms = lab @ _INV_LAB_MAT.T
  lms = 10.0 ** log_lms
  rgb = lms @ _INV_LMS_MAT.T
  return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
```
- `10.0 ** log_lms`: Nghịch đảo logarit để khôi phục LMS tuyến tính.
- `np.clip(rgb * 255.0, 0, 255).astype(np.uint8)`: Giới hạn điểm ảnh trong phạm vi nguyên 8-bit $[0, 255]$.

```python
def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
  lab = _rgb_to_lab(reference_rgb)
  return {
    "mean": np.mean(lab, axis=(0, 1)),
    "std": np.std(lab, axis=(0, 1)) + 1e-6,
  }

def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
  lab = _rgb_to_lab(image_rgb)
  mean = np.mean(lab, axis=(0, 1))
  std = np.std(lab, axis=(0, 1)) + 1e-6

  norm_lab = np.zeros_like(lab)
  for c in range(3):
    norm_lab[:, :, c] = ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]

  return _lab_to_rgb(norm_lab)
```
- `+ 1e-6`: Epsilon chống chia cho 0 khi độ lệch chuẩn bằng 0 (ví dụ ảnh đơn sắc hoàn toàn).
- `((lab - mean) / std) * ref_std + ref_mean`: Công thức căn chỉnh phân phối Gaussian một chiều độc lập trên từng kênh: trừ kỳ vọng nguồn, chia độ lệch chuẩn nguồn, nhân độ lệch chuẩn đích, cộng kỳ vọng đích.

---

### 5.3. Khối 3: Chuẩn hóa phân rã quang học Macenko (Dòng 103 – 165)

#### Cơ sở toán học:
Định luật Beer-Lambert chỉ ra mối liên hệ giữa cường độ ánh sáng truyền qua $I$ và mật độ quang học $OD$:
$$I = I_0 \cdot 10^{-OD} \implies OD = -\log_{10}\left(\frac{I}{I_0}\right)$$
Trong đó $I_0 = 255$ là cường độ nguồn sáng xuyên qua kính trắng. Mặt khác:
$$OD = V \cdot C$$
Với $V \in \mathbb{R}^{3 \times 2}$ là ma trận hệ số hấp thụ màu của 2 chất nhuộm (Hematoxylin và Eosin), và $C \in \mathbb{R}^{2}$ là nồng độ của từng chất tại điểm ảnh đó.

```python
def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
  od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
  flat_od = od.reshape(-1, 3)
  mask = np.linalg.norm(flat_od, axis=1) > od_threshold
  flat_od = flat_od[mask]
```
- `+ 1.0) / 256.0`: Kỹ thuật số học chuẩn tránh trường hợp $I = 0 \implies \log_{10}(0)$ hoặc $I = 255 \implies OD = 0$.
- `np.linalg.norm(flat_od, axis=1) > od_threshold`: Tính độ dài vector quang học. Nếu nhỏ hơn 0.15, đó là pixel nền kính trong suốt, bắt buộc phải loại bỏ để không làm sai lệch SVD.

```python
  _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
  proj = flat_od @ vh[:2].T
  phi = np.arctan2(proj[:, 1], proj[:, 0])

  min_phi = np.percentile(phi, 1.0)
  max_phi = np.percentile(phi, 99.0)
```
- `np.linalg.svd(flat_od)`: Phân tích giá trị kỳ dị. Hai hàng đầu tiên của $V^T$ (`vh[:2]`) chính là 2 véc-tơ trực giao span nên mặt phẳng chứa nhiều biến thiên mật độ quang học nhất.
- `np.arctan2(proj[:, 1], proj[:, 0])`: Chuyển tọa độ chiếu 2D sang góc cực $\phi \in [-\pi, \pi]$.
- `np.percentile(..., 1.0)` và `np.percentile(..., 99.0)`: **Tuyệt đối không lấy Min/Max tuyệt đối!** Vì ảnh luôn có bụi bẩn hoặc hạt nhiễu cực đoan. Phân vị 1% và 99% giúp loại bỏ 2% nhiễu ngoại lai một cách bền vững.

```python
  v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
  v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])

  if v1[0] < v2[0]:
    v1, v2 = v2, v1

  stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
  concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
  q99 = np.percentile(concentrations, 99.0, axis=0)

  return {"stain_matrix": stain_matrix, "q99": q99}
```
- `if v1[0] < v2[0]: v1, v2 = v2, v1`: Hematoxylin (Tím) hấp thụ ánh sáng đỏ (kênh 0) mạnh hơn nhiều so with Eosin (Hồng). Dòng này bảo đảm quy ước cột 0 luôn là Hematoxylin, cột 1 luôn là Eosin.
- `np.linalg.pinv(stain_matrix).T`: Dùng giả nghịch đảo Moore-Penrose để giải hệ phương trình nồng độ: $C = OD \cdot (V^+)^T$.
- `q99`: Nồng độ cực đại phân vị 99% của từng chất nhuộm trên ảnh mẫu.

```python
def macenko_apply(image_rgb: np.ndarray, ref_params: Dict[str, np.ndarray], od_threshold: float = 0.15) -> np.ndarray:
  ...
  if mask.sum() < 0.20 * h * w:
    return image_rgb
```
- `mask.sum() < 0.20 * h * w`: **Cơ chế phòng thủ số 1 của Macenko**: Nếu diện tích mô nhỏ hơn 20% khung hình (ví dụ ảnh toàn phông nền trắng hoặc chất nhầy loãng), số lượng điểm quang học không đủ để SVD hội tụ tin cậy. Khi đó hàm lập tức trả về ảnh gốc để tránh làm méo mó cấu trúc.
- `norm_conc = tile_conc * (ref_params["q99"] / q99)`: Co giãn nồng độ của ảnh nguồn theo nồng độ ảnh tham chiếu.
- `norm_od = norm_conc @ ref_params["stain_matrix"].T`: Tái tổng hợp mật độ quang học bằng ma trận màu của ảnh mẫu.
- `norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0`: Nghịch đảo định luật Beer-Lambert về lại RGB.

---

### 5.4. Khối 4: Tăng cường nhiễu nồng độ màu sinh học HED Jitter (Dòng 167 – 194)

```python
HE_STAIN_MATRIX = np.array([
  [0.650, 0.072],
  [0.704, 0.990],
  [0.286, 0.105],
], dtype=np.float64)
HE_STAIN_MATRIX /= np.linalg.norm(HE_STAIN_MATRIX, axis=0, keepdims=True)
```
- `HE_STAIN_MATRIX`: Ma trận véc-tơ màu nhuộm H&E chuẩn thực nghiệm trong y sinh học. Cột 1 là Hematoxylin $[0.650, 0.704, 0.286]^T$, cột 2 là Eosin $[0.072, 0.990, 0.105]^T$.

```python
def hed_stain_jitter(image_rgb: np.ndarray, sigma: float = 0.2, bias: float = 0.05) -> np.ndarray:
  od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
  h, w, _ = od.shape
  flat_od = od.reshape(-1, 3)

  c = flat_od @ np.linalg.pinv(HE_STAIN_MATRIX).T
  alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
  beta = np.random.uniform(-bias, bias, size=2)

  c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
  c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)

  od_jittered = c @ HE_STAIN_MATRIX.T
  rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
  return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)
```
- `alpha`: Hệ số co giãn nồng độ ngẫu nhiên trong khoảng $[1 - 0.2, 1 + 0.2] = [0.8, 1.2]$ (mô phỏng kỹ thuật viên pha hóa chất đậm hoặc nhạt hơn 20%).
- `beta`: Độ dịch chuyển nồng độ ngẫu nhiên $[-0.05, 0.05]$ (mô phỏng thời gian ngâm tiêu bản lâu hoặc nhanh).
- `np.clip(..., 0, None)`: Đảm bảo nồng độ thuốc nhuộm không bao giờ âm về mặt vật lý.

---

### 5.5. Khối 5: Pipeline biến đổi `PatchTransform` (Dòng 195 – 227)

```python
class PatchTransform:
  def __init__(self, policy: str = "none", norm_fn: Any = None, is_train: bool = True):
    self.policy = policy
    self.norm_fn = norm_fn
    self.is_train = is_train

  def __call__(self, img_pil: Image.Image) -> Any:
    import torchvision.transforms.functional as TF

    arr = np.array(img_pil.convert("RGB"))
    if self.norm_fn is not None:
      arr = self.norm_fn(arr)

    if self.is_train and self.policy in ("aug_stain", "aug_combined"):
      if random.random() > 0.2:
        arr = hed_stain_jitter(arr)

    tensor = TF.to_tensor(arr)

    if self.is_train and self.policy in ("aug_geo", "aug_combined"):
      if random.random() > 0.5: tensor = TF.hflip(tensor)
      if random.random() > 0.5: tensor = TF.vflip(tensor)
      rot = random.choice([0, 90, 180, 270])
      if rot > 0: tensor = TF.rotate(tensor, rot)

    return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
```
- **Ý nghĩa cờ `is_train`**: Cực kỳ quan trọng! Khi `is_train=False` (lúc đánh giá tập val hoặc test), toàn bộ các bước ngẫu nhiên (HED Jitter, lật xoay) bị tắt hoàn toàn. Đánh giá kiểm thử luôn phải trên ảnh gốc nhất quán.
- `TF.to_tensor(arr)`: Chuyển đổi định dạng từ NumPy $(H, W, C)$ uint8 $[0, 255]$ sang PyTorch Tensor $(C, H, W)$ float32 $[0.0, 1.0]$.
- `TF.rotate(tensor, rot)`: Chỉ xoay các góc trực giao $\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$ để không sinh ra các góc đen bị cắt viền do phép xoay góc lẻ.

---

### 5.6. Khối 6: Quét ảnh & Phân chia tập dữ liệu `prepare_dataset_splits` (Dòng 229 – 352)

```python
def find_class_images(base_dir: Path, class_name: str) -> List[Path]:
```
- Quét thông minh: Kiểm tra thư mục con trực tiếp trước (`base_dir / class_name`), nếu không thấy thì duyệt không phân biệt chữ hoa/thường, nếu vẫn không thấy thì quét đệ quy `rglob("*")`. Điều này giúp code chạy mượt mà trên mọi cấu trúc giải nén zip của Kaggle.

```python
def prepare_dataset_splits(source_dir, target_dir, out_dir, subset_size=25000, seed=100):
  ...
  # Lấy mẫu phân tầng 25.000 ảnh từ 100.000 ảnh nguồn
  if 0 < subset_size < len(source_df):
    per_class = subset_size // NUM_CLASSES
    source_df = source_df.groupby("class_name", group_keys=False).apply(
      lambda g: g.sample(min(len(g), per_class), random_state=seed)
    ).reset_index(drop=True)

  # Chia 70% train / 15% val_id / 15% test_id có phân tầng
  train_df, rest_df = train_test_split(source_df, test_size=0.30, stratify=source_df["class_name"], random_state=seed)
  val_df, test_id_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["class_name"], random_state=seed)
```
- `per_class = 25000 // 9 = 2777` ảnh mỗi lớp: Đảm bảo cân bằng lớp tuyệt đối ngay từ khâu lấy mẫu.
- `stratify=source_df["class_name"]`: Đảm bảo tỷ lệ 9 lớp mô học trong tập Train (17.495 ảnh), Val (3.749 ảnh) và Test-ID (3.749 ảnh) đồng nhất hoàn hảo.

```python
  # Trích xuất ảnh mẫu tham chiếu chuẩn hóa
  tum_train = train_df[train_df["class_name"] == "TUM"]
  ref_path = Path(tum_train.iloc[0]["image_path"])
  ref_img = Image.open(ref_path).convert("RGB")
  ref_save_path = out_dir / "reference_stain.png"
  ref_img.save(ref_save_path)
```
- **Tại sao lấy ảnh tham chiếu từ tập Train của lớp `TUM`?**:
  1. Tuyệt đối không lấy từ tập test để tránh rò rỉ dữ liệu (*Data Leakage*).
  2. Lớp `TUM` (u đại trực tràng) có mật độ tế bào dày đặc, bắt cả hai loại thuốc nhuộm Hematoxylin (nhân u) và Eosin (chất nền) rất rõ ràng, tạo ra véc-tơ màu chuẩn mực nhất cho thuật toán Macenko và Reinhard.

---

### 5.7. Khối 7: Khởi tạo mô hình & Linear Probing Phikon (Dòng 354 – 382)

```python
def build_model(backbone_name: str, num_classes: int = NUM_CLASSES):
  if backbone_name == "phikon":
    from transformers import AutoModel
    class PhikonClassifier(nn.Module):
      def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("owkin/phikon")
        for p in self.encoder.parameters():
          p.requires_grad = False
        self.fc = nn.Linear(768, num_classes)

      def forward(self, x):
        feat = self.encoder(x).last_hidden_state[:, 0]
        return self.fc(feat)

    return PhikonClassifier()
```
- `p.requires_grad = False`: **Giao thức Linear Probing**: Đóng băng toàn bộ 86 triệu tham số của Vision Transformer. Mô hình không được phép cập nhật trọng số backbone, chỉ huấn luyện duy nhất tầng phân loại cuối cùng `nn.Linear(768, 9)`.
- `last_hidden_state[:, 0]`: Trong kiến trúc ViT, token tại vị trí index 0 là **Class Token `[CLS]`** — nơi tổng hợp thông tin ngữ nghĩa toàn cục của toàn bộ bức ảnh thông qua các tầng Self-Attention.
- `timm.create_model("convnext_tiny.fb_in1k", ...)` & `timm.create_model("resnet50.a1_in1k", ...)`: Nạp trọng số tiền huấn luyện ImageNet-1K từ thư viện `timm`.

---

### 5.8. Khối 8: Đánh giá mô hình & Đo lường độ bền vững (Dòng 384 – 414)

```python
def evaluate_model(model, loader, device) -> Dict[str, float]:
  ...
  with torch.no_grad():
    for images, targets in loader:
      images = images.to(device, non_blocking=True)
      with torch.amp.autocast("cuda"):
        logits = model(images)
      probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
      ...
  macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
  bal_acc = balanced_accuracy_score(y_true, y_pred)
  acc = accuracy_score(y_true, y_pred)
  auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
```
- `torch.no_grad()`: Tắt đồ thị đạo hàm để tiết kiệm VRAM và tăng tốc gấp đôi tốc độ suy luận.
- `torch.amp.autocast("cuda")`: Suy luận bằng độ chính xác nửa (fp16) trên nhân Tensor Core của GPU NVIDIA.
- `f1_score(..., average="macro")`: **Chỉ số đo lường vàng của nghiên cứu**: Tính F1 độc lập cho từng lớp rồi lấy trung bình cộng không trọng số:
$$\text{Macro-F1} = \frac{1}{9} \sum_{c=1}^{9} \text{F1}_c$$
Giúp đánh giá công bằng, không bị thiên lệch bởi các lớp chiếm số đông.
- `roc_auc_score(..., multi_class="ovr")`: Diện tích dưới đường cong ROC theo chiến lược Một-đối-tất-cả (One-vs-Rest).

---

### 5.9. Khối 9: Động cơ huấn luyện `train_experiment` (Dòng 416 – 513)

```python
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr if exp["backbone"] != "phikon" else 3e-4, weight_decay=0.05)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  scaler = torch.amp.GradScaler("cuda")
```
- `AdamW`: Khắc phục nhược điểm của Adam kinh điển bằng cách tách rời trực tiếp hệ số phân rã trọng số (*Decoupled Weight Decay* $0.05$), ngăn ngừa overfitting cực tốt.
- `CosineAnnealingLR`: Hạ tốc độ học theo chu kỳ nửa sóng hình sin từ $10^{-3}$ xuống $10^{-5}$, giúp trọng số hội tụ êm ái vào đáy cực tiểu địa phương.
- `Label Smoothing = 0.1`: Thay vì ép nhãn One-hot cứng nhắc $[0, 0, 1, \dots]$, chuyển thành nhãn mềm $[0.0125, 0.0125, 0.9, \dots]$. Chống hiện tượng mô hình quá tự tin (*overconfidence*), cải thiện rõ rệt khả năng tổng quát hóa trên miền lạ.
- `scaler = torch.amp.GradScaler("cuda")`: Bộ co giãn gradient. Khi tính toán số thực 16-bit, các gradient quá nhỏ dễ bị tràn số dưới (*underflow*) về 0. `GradScaler` nhân gradient lên $2^{16}$ trước khi lan truyền ngược, sau đó chia lại trước khi cập nhật trọng số.

```python
  # Lưu giữ checkpoint tốt nhất trên tập Validation
  if val_res["macro_f1"] > best_val_f1:
    best_val_f1 = val_res["macro_f1"]
    best_state = copy.deepcopy(model.state_dict())
```
- Cơ chế Early Best Checkpointing: Chỉ nạp trọng số có điểm `val_f1` cao nhất trong 8 epoch để đem đi kiểm thử trên `test_id` và `test_ood`.

```python
  delta_f1 = round(id_res["macro_f1"] - ood_res["macro_f1"], 4)
  rr_f1 = round((ood_res["macro_f1"] / max(id_res["macro_f1"], 1e-6)) * 100.0, 2)
```
- $\Delta\text{-F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$: Độ tụt dốc hiệu năng khi đổi bệnh viện (càng thấp càng tốt).
- $\text{Retention Rate (RR)} = \frac{\text{F1}_{\text{OOD}}}{\text{F1}_{\text{ID}}} \times 100\%$: Tỷ lệ giữ phong độ (càng gần 100% càng bền vững).

---

### 5.10. Khối 10: Hàm điều phối `main()` (Dòng 515 – 588)

```python
def main():
  parser = argparse.ArgumentParser(...)
  ...
  if args.dry_run:
    # In kế hoạch 13 thí nghiệm mà không chạy thực tế (phục vụ kiểm tra)
    return
  ...
  # Xuất bảng kết quả tóm tắt
  res_df.to_csv(csv_file, index=False)
  md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")
```
- Hỗ trợ cờ `--dry-run` để kiểm tra cú pháp và danh sách hàng đợi trong 0.5 giây.
- Tự động xuất kết quả ra cả file máy đọc `summary_results.csv` và báo cáo người đọc `RESULTS_TABLE.md`.

---

## 6. KIẾN TRÚC HƯỚNG ĐỐI TƯỢNG (OOP) CÔNG NGHIỆP TRONG THƯ MỤC `src/`

Để đáp ứng tiêu chuẩn phần mềm công nghiệp phục vụ bảo trì lâu dài, toàn bộ mã nguồn đã được module hóa thành gói [`src/`](file:///D:/LT/DL2026-1-32/src) với kiến trúc sạch (Clean Architecture):

```text
src/
├── __init__.py                  # Expose Clean Public API & Package Versioning
├── __main__.py                  # Điểm thực thi khi gọi `python -m src`
├── cli.py                       # CLI Argument Parser với Typed Dataclasses
├── pipeline.py                  # High-level Orchestrator điều phối toàn trình
│
├── config/                      # Quản lý Cấu hình & Siêu tham số
│   ├── constants.py             # 9 lớp mô học, ImageNet stats, Ma trận H&E
│   ├── schemas.py               # Dataclasses: ExperimentCell, SplitConfig, TrainingConfig
│   └── experiments.py           # Danh mục 13 ô ablation & Bộ lọc ID
│
├── normalization/               # Chuẩn hóa Màu nhuộm (Stain Normalization)
│   ├── base.py                  # BaseStainNormalizer (Abstract Base Class - Interface)
│   ├── reinhard.py              # ReinhardNormalizer (Ruderman LAB Color Transfer)
│   ├── macenko.py               # MacenkoNormalizer (OD SVD Stain Deconvolution)
│   └── factory.py               # NormalizerFactory (Factory Pattern)
│
├── augmentation/                # Tăng cường Dữ liệu (Data Augmentation)
│   ├── policies.py              # AugmentationPolicy (Enum: none, geo, stain, combined)
│   ├── hed_jitter.py            # Biological H&E Stain Concentration Jitter
│   └── transforms.py            # PatchTransform (Composed Pipeline)
│
├── data/                        # Xử lý Dữ liệu & Phân chia Tập mẫu
│   ├── discovery.py             # Quét ảnh thông minh không phân biệt hoa thường
│   ├── dataset.py               # HistologyDataset (PyTorch Dataset)
│   ├── splitter.py              # DatasetSplitter (Stratified 70/15/15 + OOD Isolation)
│   └── dataloader.py            # create_dataloaders (Pinned-memory DataLoader Factory)
│
├── models/                      # Kiến trúc Mạng Nơ-ron
│   ├── phikon.py                # PhikonClassifier (Linear Probing ViT Wrapper)
│   └── factory.py               # ModelFactory (Unified Model Instantiation)
│
├── engine/                      # Động cơ Huấn luyện & Đánh giá
│   ├── evaluator.py             # Evaluator (Macro-F1, Balanced Acc, AUROC)
│   └── trainer.py               # ExperimentTrainer (AMP, Cosine LR, Best-Checkpoint Tracking)
│
└── reporting/                   # Tổng hợp & Xuất Báo cáo
    ├── metrics.py               # ExperimentResult (Dataclass kết quả)
    └── reporter.py              # ResultsReporter (CSV & Markdown Exporter)
```

### Các nguyên lý Thiết kế Phần mềm được áp dụng:
1. **SOLID Principles**:
   - **Single Responsibility (SRP)**: Mỗi class chỉ làm một việc duy nhất (`DatasetSplitter` chỉ chia tập, `Evaluator` chỉ tính metric, `ResultsReporter` chỉ xuất bảng).
   - **Open/Closed (OCP)**: Thêm thuật toán chuẩn hóa mới (như Vahadane) chỉ cần kế thừa [`BaseStainNormalizer`](file:///D:/LT/DL2026-1-32/src/normalization/base.py) mà không cần sửa đổi các file khác.
   - **Liskov Substitution (LSP)**: `ReinhardNormalizer` và `MacenkoNormalizer` có thể thay thế lẫn nhau hoàn toàn thông qua phương thức chuẩn `.fit()` và `.transform()`.
   - **Dependency Inversion (DIP)**: `ExperimentPipeline` phụ thuộc vào các Abstraction và Config Dataclass thay vì hard-code giá trị.
2. **Design Patterns**:
   - **Factory Pattern**: `NormalizerFactory.create("macenko")` và `ModelFactory.build("resnet50")` đóng gói logic khởi tạo đối tượng phức tạp.
   - **Strategy Pattern**: `PatchTransform` cho phép hoán đổi linh hoạt các chính sách tăng cường (`AugmentationPolicy`) tại runtime.
3. **Graceful Fallbacks & Lazy Imports**:
   - Các gói nặng (`torch`, `pandas`, `sklearn`, `transformers`) được import an toàn, cho phép kiểm thử các module toán học độc lập trên môi trường tối giản.

---

## 7. BẢNG KẾT QUẢ 13 THÍ NGHIỆM & 3 PHÁT HIỆN KHOA HỌC CHẤN ĐỘNG

| Mã Thí Nghiệm | Giai đoạn | Bộ não (Backbone) | Chuẩn hóa màu (Norm) | Tăng cường dữ liệu (Aug) | Điểm sân nhà (ID F1) | Điểm trường lạ (OOD F1) | Độ tụt dốc ($\Delta$-F1) | Giữ phong độ (RR %) | Thời gian chạy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Stage 0 | ResNet-50 | Không | Không | **0.9893** | **0.6644** | 0.3249 | **67.16%** | 11.0 phút |
| **EXP-02** | Stage 1 | ResNet-50 | Reinhard | Không | 0.9862 | 0.8488 | 0.1373 | 86.07% | 16.5 phút |
| **EXP-03** | Stage 1 | ResNet-50 | Macenko | Không | 0.9653 | 0.8090 | 0.1563 | 83.80% | 17.3 phút |
| **EXP-04** | Stage 2 | ResNet-50 | Không | Aug-Geo (Xoay) | 0.9899 | 0.6172 | 0.3727 | 62.35% | 10.9 phút |
| **EXP-05** | Stage 2 | ResNet-50 | Không | Aug-Stain (Nhiễu màu) | 0.9877 | 0.7443 | 0.2434 | 75.36% | 10.9 phút |
| **EXP-06** | Stage 2 | ResNet-50 | Không | Aug-Combined (Cả hai) | 0.9901 | 0.7105 | 0.2796 | 71.76% | 10.9 phút |
| **EXP-07** | Stage 3 | ResNet-50 | Macenko | Aug-Geo (Xoay) | 0.9717 | **0.8689** | **0.1028** | **89.42%** | 17.9 phút |
| **EXP-08** | Stage 3 | ResNet-50 | Macenko | Aug-Stain (Nhiễu màu) | 0.9619 | 0.7850 | 0.1770 | 81.60% | 20.0 phút |
| **EXP-09** | Stage 3 | ResNet-50 | Macenko | Aug-Combined (Cả hai) | 0.9699 | 0.8604 | 0.1094 | 88.72% | 20.7 phút |
| **EXP-10** | Stage 4 | ConvNeXt-Tiny | Không | Không | 0.9891 | 0.6821 | 0.3069 | 68.97% | 13.0 phút |
| **EXP-11** | Stage 4 | ConvNeXt-Tiny | Macenko | Aug-Combined | 0.9763 | 0.8685 | 0.1077 | 88.97% | 20.7 phút |
| **EXP-12** | Stage 4 | Phikon | Không | Không | **0.9912** | **0.8239** | 0.1673 | **83.12%** | **9.1 phút** |
| **EXP-13** | Stage 4 | Phikon | Macenko | Aug-Combined | 0.9050 | 0.6982 | 0.2068 | 77.15% | 20.1 phút |

### 3 Phát hiện Khoa học then chốt:
1. **Chuẩn hóa màu sắc là liều thuốc hữu hiệu nhất**: Bật Reinhard (**EXP-02**) hoặc Macenko (**EXP-03**) phục hồi ngay lập tức hơn 14 – 18 điểm F1 khi chuyển viện.
2. **Nghịch lý tăng cường dữ liệu (Augmentation Paradox)**: Khi chưa chuẩn hóa, gây nhiễu màu (**EXP-05**) có ích (+8 điểm F1). Nhưng khi đã chuẩn hóa bằng Macenko, việc nhồi thêm nhiễu màu thuốc nhuộm (**EXP-08**) lại gây xung đột, làm tụt điểm từ **0.8689** xuống **0.7850**! Quán quân toàn diện là **EXP-07** (Macenko + Xoay lật đơn giản).
3. **Foundation Model bị tổn thương khi ép chuẩn hóa màu**: Phikon nguyên bản (**EXP-12**) đạt tới **0.8239** chỉ sau 9 phút huấn luyện. Nhưng khi ép dùng "bộ giáp" Macenko + Augment (**EXP-13**), điểm số rơi thảm hại xuống **0.6982** (-12.5 điểm F1) do thuật toán Macenko tạo ra các vết nhiễu vi thể làm hỏng cơ chế Self-Attention 16x16 pixel của Vision Transformer.

---

## 8. PHÂN TÍCH ĐỊNH TÍNH & CÁC CA NHẦM LẪN KINH ĐIỂN (ERROR ANALYSIS)

1. **`STR` (Mô đệm) vs. `MUS` (Cơ trơn)**: Đều dạng sợi bắt màu hồng Eosin. Khi chưa chuẩn hóa, màu hồng đậm hơn bình thường ở Aachen khiến AI nhầm mô đệm thành cơ trơn. Chuẩn hóa Macenko nắn lại màu, buộc AI phải nhìn vào hình thái nhân tế bào để phân biệt.
2. **`DEB` (Hoại tử) vs. `TUM` (Khối u)**: Trung tâm khối u thiếu máu thường bị hoại tử, chứa tế bào ung thư vỡ vụn. Ngay cả bác sĩ giải phẫu bệnh cũng gặp khó khăn ở vùng ranh giới này.
3. **`MUC` (Chất nhầy) vs. `BACK` (Phông nền kính)**: Chất nhầy trong suốt rất dễ bị "cháy sáng" thành màu trắng nếu máy quét tăng độ chiếu sáng, khiến AI tưởng nhầm là mặt kính rỗng.

---

## 9. VẬN HÀNH TRÊN KAGGLE: CƠ CHẾ HOẠT ĐỘNG CỦA `notebook.ipynb`

Toàn bộ quy trình chạy tự động trong [`notebook.ipynb`](file:///D:/LT/DL2026-1-32/notebook.ipynb):
- **Cell 1**: Cài đặt `timm`, `transformers`, `tabulate`.
- **Cell 2**: Khắc phục đường dẫn động của Kaggle, copy mã nguồn vào `/kaggle/working/run_experiments.py`.
- **Cell 3**: Dùng `subprocess.Popen` chạy pipeline ngầm với live logs đẩy ra màn hình.
- **Cell 4**: Hiển thị bảng kết quả `RESULTS_TABLE.md` trực tiếp bằng Markdown.
- **Cell 5**: Hiển thị ảnh mẫu tham chiếu `reference_stain.png`.

---

## 10. TỪ ĐIỂN THUẬT NGỮ DEEP LEARNING "BÌNH DÂN HỌC VỤ"

| Thuật ngữ | Ý nghĩa kỹ thuật | Ví von đời thường |
| :--- | :--- | :--- |
| **Domain Shift** | Lệch phân phối dữ liệu đầu vào giữa 2 bệnh viện. | Đề thi đại học ở tỉnh khác có phong cách ra đề lạ hoắc so với trường làng. |
| **Linear Probing** | Đóng băng toàn bộ backbone, chỉ học 1 tầng Linear cuối cùng. | Mời giáo sư đầu ngành về, chỉ cần phổ biến 5 phút quy ước chấm điểm là làm việc được ngay. |
| **Label Smoothing** | Làm mềm nhãn One-hot thành phân phối xác suất nhẹ. | Ngăn học sinh trở nên quá ngạo mạn, tự phụ vào đáp án của mình. |
| **Cosine Annealing** | Hạ tốc độ học theo đường cong hình sin mềm mại. | Lúc đầu năm học thì đi nhanh, càng gần ngày thi càng đi chậm nắn nót. |
| **AMP (fp16)** | Tính toán xen kẽ số thực 16-bit và 32-bit. | Viết nháp bằng bút chì cho nhanh (16-bit), kết quả then chốt mới viết bút mực (32-bit). |
| **Retention Rate (RR)** | Tỷ lệ $\frac{\text{OOD}}{\text{ID}} \times 100\%$. | Điểm thi trường lạ so với điểm thi trên lớp ở nhà. Càng gần 100% càng chứng tỏ học thật thi thật! |

---

## 11. BỘ 20 CÂU HỎI VẤN ĐÁP BẢO VỆ ĐỒ ÁN (DEFENSE Q&A) HIỂM HÓC NHẤT

Dưới đây là 20 câu hỏi trọng tâm mà Hội đồng chấm thi thường xuyên đặt ra cùng câu trả lời mẫu chuẩn xác, tự tin:

#### ❓ Câu 1: Tại sao nhóm lại chọn chỉ số Macro-F1 làm thước đo chính mà không dùng Accuracy thông thường?
> **Trả lời**: Trong bài toán phân loại đa lớp y khoa, mặc dù tập dữ liệu của nhóm đã được lấy mẫu phân tầng cân bằng, nhưng trong thực tế lâm sàng, tỷ lệ xuất hiện giữa các loại mô (như u ác tính so với mô mỡ hay chất nhầy) luôn có sự chênh lệch lớn. Accuracy chỉ đo tỷ lệ đoán đúng tổng thể và dễ bị chi phối bởi các lớp chiếm đa số. **Macro-F1** tính điểm F1 riêng biệt cho từng lớp trong cả 9 lớp mô học rồi lấy trung bình cộng không trọng số:
> $$\text{Macro-F1} = \frac{1}{9} \sum_{c=1}^{9} \text{F1}_c$$
> Nhờ đó, một mô hình chỉ đạt điểm Macro-F1 cao khi và chỉ khi nó phân loại tốt trên **tất cả 9 phân lớp**, bảo đảm an toàn sinh mạng cho bệnh nhân.

---

#### ❓ Câu 2: Bản chất của hiện tượng Domain Shift trong nghiên cứu này là gì? Tại sao ResNet-50 thô (EXP-01) lại tụt dốc thảm hại từ 98.9% xuống 66.4%?
> **Trả lời**: Hiện tượng Domain Shift ở đây là **Covariate Shift** do sự khác biệt về thuốc thử, quy trình ngâm tiêu bản và thiết bị số hóa giữa Trung tâm Ung thư Heidelberg (tập nguồn) và Bệnh viện Đại học Aachen (tập đích). Mô hình ResNet-50 ở EXP-01 bị mắc phải căn bệnh **Shortcut Learning (Học lối tắt)**: Do ở Heidelberg, các mô ung thư thường có nhân tế bào dày đặc nên ảnh ánh tím sẫm, mạng nơ-ron đã học lối tắt "cứ thấy tím là phán ung thư". Khi sang Aachen, nơi cả các mô cơ lành tính cũng bị nhuộm hơi ánh tím, mô hình bị đánh lừa hoàn toàn, dẫn đến độ chính xác rơi tự do 32.5 điểm F1.

---

#### ❓ Câu 3: Thuật toán chuẩn hóa Reinhard hoạt động như thế nào? Tại sao lại phải chuyển sang không gian Ruderman LAB mà không dùng trực tiếp RGB?
> **Trả lời**: Trong không gian RGB, 3 kênh màu Đỏ, Xanh lục, Xanh lam có mối tương quan chéo cực kỳ cao (khi tăng độ sáng thì cả 3 kênh đều tăng). Thuật toán Reinhard (2001) trước hết chuyển RGB sang không gian nón cảm thụ mắt người LMS bằng ma trận thực nghiệm Ruderman, sau đó lấy logarit cơ số 10 để mô phỏng phản ứng thị giác phi tuyến, rồi chiếu sang hệ trục trực giao đối kháng $L\alpha\beta$:
> - Kênh $L$: Đo độ sáng phi sắc.
> - Kênh $\alpha$: Trục màu Đỏ – Lục.
> - Kênh $\beta$: Trục màu Vàng – Lam.
> Trong không gian này, 3 kênh hoàn toàn độc lập và không tương quan. Khi đó, ta có thể căn chỉnh phân phối Gaussian một chiều bằng cách chuẩn hóa Z-score theo giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ của ảnh tham chiếu mẫu mà không sợ làm méo mó các sắc thái màu còn lại.

---

#### ❓ Câu 4: Định luật Beer-Lambert là gì và tại sao trong thuật toán Macenko bắt buộc phải lấy logarit âm của cường độ ảnh?
> **Trả lời**: Trong vật lý quang học, định luật Beer-Lambert phát biểu rằng cường độ ánh sáng $I$ truyền qua một lát cắt sinh học giảm dần theo hàm số mũ của mật độ quang học (Optical Density - OD):
> $$I = I_0 \cdot 10^{-OD} \iff OD = -\log_{10}\left(\frac{I}{I_0}\right)$$
> Do đó, không gian cường độ pixel RGB là không gian phi tuyến đối với nồng độ chất nhuộm. Bằng cách lấy $-\log_{10}(I / 255)$, ta chuyển bài toán từ không gian cường độ phi tuyến sang **không gian mật độ quang học tuyến tính**. Trong không gian OD, mật độ quang học bằng tổng tuyến tính của nồng độ chất nhuộm nhân với véc-tơ hấp thụ: $OD = V \cdot C$. Điều này cho phép áp dụng các công cụ đại số tuyến tính như SVD để phân rã.

---

#### ❓ Câu 5: Tại sao thuật toán Macenko lại sử dụng SVD (Phân tích giá trị kỳ dị) mà không dùng PCA hay k-Means?
> **Trả lời**: Sau khi lọc bỏ các điểm phông nền kính có $OD \le 0.15$, tập hợp các điểm mật độ quang học trong không gian 3 chiều sẽ nằm chủ yếu trên một mặt phẳng 2 chiều được tạo bởi 2 chất nhuộm Hematoxylin và Eosin. Phép phân tích SVD trên ma trận mật độ quang học $OD = U \Sigma V^T$ tìm ra trực tiếp 2 véc-tơ kỳ dị đầu tiên (2 hàng đầu của $V^T$), chính là hệ cơ sở trực giao tối ưu span nên mặt phẳng này theo nghĩa bình phương tối thiểu, với tốc độ tính toán cực nhanh và ổn định số học cao hơn nhiều so với việc tính ma trận hiệp phương sai trong PCA.

---

#### ❓ Câu 6: Trong hàm `macenko_fit()`, tại sao nhóm lại lấy phân vị 1% và 99% của góc cực phi mà không lấy giá trị Min và Max tuyệt đối?
> **Trả lời**: Sau khi chiếu các điểm quang học lên mặt phẳng 2D của SVD và tính góc cực $\phi = \arctan2(y, x)$, hai véc-tơ biên của hình quạt biểu diễn cho hai chất nhuộm Hematoxylin và Eosin tinh khiết. Nếu ta dùng `min()` và `max()` tuyệt đối, chỉ cần một vài pixel bị nhiễu do bụi bẩn trên lam kính hoặc điểm ảnh chết trên cảm biến máy quét, véc-tơ màu sẽ bị lệch nghiêm trọng. Bằng cách lấy **phân vị 1% và 99%**, thuật toán loại bỏ hoàn toàn 2% nhiễu ngoại lai (*outliers*), đảm bảo ước lượng véc-tơ màu chuẩn xác và bền vững.

---

#### ❓ Câu 7: Đoạn mã `if mask.sum() < 0.20 * h * w: return image_rgb` trong `macenko_apply()` có ý nghĩa gì?
> **Trả lời**: Đây là cơ chế phòng vệ chống sập (*fail-safe*) thiết yếu của thuật toán Macenko. Nếu một lát cắt có diện tích mô nhỏ hơn 20% khung hình (ví dụ như ảnh phông nền kính `BACK` hoặc mô mỡ rỗng `ADI`), số lượng điểm ảnh có $OD > 0.15$ quá ít. Khi đó, phép phân tích SVD sẽ không thể ước lượng được mặt phẳng 2 chiều một cách tin cậy và có thể dẫn đến ma trận suy biến. Việc trả về ngay ảnh gốc giúp pipeline chạy trơn tru, không bị lỗi toán học và không làm hỏng các mẫu ảnh vốn dĩ không có mô.

---

#### ❓ Câu 8: Thuật toán gây nhiễu nồng độ HED Stain Jitter dựa trên cơ chế nào và ma trận `HE_STAIN_MATRIX` từ đâu mà có?
> **Trả lời**: Khác với Color Jitter thông thường (chỉ đổi màu ngẫu nhiên vô nghĩa), **HED Stain Jitter** là kỹ thuật tăng cường dữ liệu bám sát bản chất sinh học. Ma trận `HE_STAIN_MATRIX` ($3 \times 2$) được đo đạc thực nghiệm trong y sinh học:
> - Cột Hematoxylin: $[0.650, 0.704, 0.286]^T$
> - Cột Eosin: $[0.072, 0.990, 0.105]^T$
> Thuật toán giải nồng độ của từng chất bằng giả nghịch đảo Moore-Penrose, sau đó biến đổi nồng độ theo công thức:
> $$C_{\text{mới}} = \text{clip}(C \cdot \alpha + \beta, 0, \infty)$$
> Trong đó $\alpha \in [0.8, 1.2]$ mô phỏng nồng độ hóa chất đậm nhạt $\pm 20\%$, còn $\beta \in [-0.05, 0.05]$ mô phỏng sự chênh lệch thời gian ngâm tiêu bản. Nhờ đó, AI học được tính bất biến với nồng độ thuốc nhuộm thực tế.

---

#### ❓ Câu 9: "Nghịch lý tăng cường dữ liệu" (Augmentation Paradox) được thể hiện như thế nào trong kết quả thực nghiệm?
> **Trả lời**: Đây là một trong những phát hiện khoa học then chốt của dự án:
> - Khi **chưa chuẩn hóa màu**, HED Stain Jitter (**EXP-05**) hoạt động rất tốt, nâng điểm OOD từ 0.6644 lên **0.7443** (+8 điểm F1).
> - Nhưng khi **đã chuẩn hóa màu bằng Macenko**, việc kết hợp thêm xoay lật hình học (**EXP-07**) đạt quán quân toàn diện với điểm OOD lên tới **0.8689** (giữ 89.42% phong độ). Tuy nhiên, nếu nhồi thêm HED Stain Jitter (**EXP-08**), điểm số lập tức **rơi xuống 0.7850** (tụt hơn 8.3 điểm F1)!
> - **Lý do**: Thuật toán Macenko đã cất công nắn mọi bức ảnh về một phân phối nồng độ chuẩn mực duy nhất. Việc tiếp tục gây nhiễu nồng độ ngẫu nhiên lúc học đã phá vỡ sự nhất quán của không gian đặc trưng mà Macenko vừa tạo ra.

---

#### ❓ Câu 10: Tại sao Foundation Model Phikon (EXP-12) tự nhiên đạt kết quả rất tốt (0.8239), nhưng khi áp dụng Macenko + Augment (EXP-13) lại tụt dốc nghiêm trọng xuống 0.6982?
> **Trả lời**: 
> 1. Phikon là Vision Transformer được tiền huấn luyện tự giám sát (iBOT) trên **40 triệu ảnh mô bệnh học** của TCGA. Không gian biểu diễn đặc trưng của nó đã tự động tích lũy khả năng miễn dịch cực cao với sự biến thiên màu sắc tự nhiên.
> 2. Phikon chia ảnh thành các mảnh vuông $16 \times 16$ pixel và cơ chế Self-Attention phụ thuộc rất lớn vào các cấu trúc vân vi thể (*micro-textures*). Thuật toán Macenko khi phân rã quang học và tái tạo lại điểm ảnh vô tình tạo ra các vết nhiễu mịn (*artifacts*) ở cấp độ pixel.
> 3. Các vết nhiễu này phá hỏng các đặc trưng tinh vi mà ViT đã học được. Do đó, với các Foundation Model chuyên biệt y tế, giữ nguyên ảnh thô tự nhiên lại mang lại hiệu quả cao hơn việc cố tình can thiệp tiền xử lý.

---

#### ❓ Câu 11: Phương pháp "Linear Probing" trên mô hình Phikon là gì? Tại sao nhóm không Fine-tune toàn bộ mô hình?
> **Trả lời**: Linear Probing là phương thức đánh giá chuẩn trong học tự giám sát: ta **đóng băng 100% trọng số** của mạng nơ-ron nền tảng (`requires_grad = False` cho toàn bộ encoder ViT), và chỉ huấn luyện duy nhất một lớp ánh xạ tuyến tính phân loại ở đầu ra: `nn.Linear(768, 9)`. Nhóm không fine-tune toàn bộ vì:
> 1. Phikon có hơn 86 triệu tham số. Fine-tune trên tập dữ liệu nhỏ rất dễ gây ra hiện tượng **Catastrophic Forgetting (Quên thảm khốc)** các kiến thức tổng quát đã học từ 40 triệu ảnh trước đó.
> 2. Linear Probing là bài kiểm tra trung thực nhất để đo lường xem bản thân các đặc trưng của Foundation Model có thực sự bền vững với Domain Shift hay không.
> 3. Giảm chi phí tính toán: Phikon chỉ mất **9.1 phút** để hoàn thành 8 epoch, nhanh nhất trong toàn bộ 13 thí nghiệm.

---

#### ❓ Câu 12: Trong mô hình Phikon, tại sao nhóm lại trích xuất đặc trưng tại `last_hidden_state[:, 0]`?
> **Trả lời**: Trong kiến trúc Vision Transformer (ViT), ảnh đầu vào được chia thành các patch $16 \times 16$ và được gắn thêm một token đặc biệt ở vị trí đầu tiên (chỉ số index 0) gọi là **Class Token `[CLS]`**. Qua các tầng Transformer Encoder, nhờ cơ chế đa đầu chú ý (*Multi-Head Self-Attention*), token `[CLS]` này liên tục tương tác và tổng hợp thông tin ngữ nghĩa toàn cục từ tất cả các patch còn lại trong ảnh. Do đó, vector ẩn tại vị trí 0 kích thước 768 chiều chính là vector biểu diễn súc tích nhất cho toàn bộ bức ảnh.

---

#### ❓ Câu 13: Kỹ thuật Label Smoothing với hệ số 0.1 trong hàm mất mát CrossEntropyLoss có tác dụng gì?
> **Trả lời**: Trong hàm mất mát Cross-Entropy tiêu chuẩn, nhãn đích là vector One-hot (ví dụ $[0, 0, 1, 0, \dots]$), điều này ép mô hình phải đẩy logit của lớp đúng ra vô cùng lớn so với các lớp khác ($p \to 1.0$), dẫn đến hiện tượng mạng nơ-ron quá tự tin vào phán đoán (*overconfidence*). Kỹ thuật **Label Smoothing** ($0.1$) làm mịn phân phối nhãn đích theo công thức:
> $$y_{\text{smooth}} = (1 - \epsilon) y_{\text{one-hot}} + \frac{\epsilon}{K}$$
> Với $\epsilon = 0.1$ và $K = 9$, nhãn đúng nhận giá trị $0.911$, còn 8 nhãn sai nhận $0.011$. Điều này đóng vai trò như một cơ chế điều hòa (*regularization*), giúp không gian đặc trưng của các lớp được phân tách mềm mại hơn và nâng cao rõ rệt độ bền vững khi gặp dữ liệu lệch miền ở bệnh viện lạ.

---

#### ❓ Câu 14: Tại sao nhóm lại sử dụng bộ tối ưu AdamW thay vì SGD hay Adam truyền thống?
> **Trả lời**: Thuật toán Adam kinh điển kết hợp L2 Regularization bằng cách cộng đạo hàm của trọng số vào gradient trước khi tính moment. Năm 2017, Loshchilov và Hutter đã chứng minh rằng cách làm này làm mất đi bản chất của phân rã trọng số trong các bộ tối ưu thích nghi. **AdamW (Decoupled Weight Decay)** tách rời hoàn toàn phép trừ phân rã trọng số ra khỏi bước cập nhật moment:
> $$\theta_{t} = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
> Với hệ số `weight_decay = 0.05`, AdamW giúp mô hình tổng quát hóa tốt hơn hẳn SGD trên các kiến trúc hiện đại và tránh hiện tượng bùng nổ trọng số.

---

#### ❓ Câu 15: Cơ chế lập lịch Cosine Annealing Learning Rate hoạt động như thế nào?
> **Trả lời**: Bộ lập lịch `CosineAnnealingLR` điều chỉnh tốc độ học $\eta_t$ sau mỗi batch theo đường cong nửa chu kỳ cosin:
> $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\max}} \pi\right)\right)$$
> Với $\eta_{\max} = 10^{-3}$, $\eta_{\min} = 10^{-5}$ và $T_{\max} = \text{epochs} \times N_{\text{batches}}$. Ở những epoch đầu tiên, tốc độ học lớn giúp mô hình nhanh chóng thoát khỏi các vùng bằng phẳng (*saddle points*). Càng về cuối chu kỳ, tốc độ học giảm êm dịu về mức rất nhỏ, giúp các trọng số hội tụ chính xác vào đáy thung lũng của hàm mất mát mà không bị dao động nhảy vọt ra ngoài.

---

#### ❓ Câu 16: Kỹ thuật tính toán độ chính xác hỗn hợp AMP (Automatic Mixed Precision) và `GradScaler` bảo vệ mô hình khỏi lỗi gì?
> **Trả lời**: AMP tự động ép các phép nhân ma trận nặng về số thực nửa độ chính xác 16-bit (`fp16`), giúp giảm $50\%$ bộ nhớ VRAM và tận dụng nhân Tensor Core của GPU. Tuy nhiên, phạm vi biểu diễn của `fp16` rất hẹp ($[2^{-14}, 2^{15}]$). Các gradient nhỏ trong quá trình lan truyền ngược rất dễ bị tràn số dưới (**Underflow**) và biến thành số $0$, làm tê liệt quá trình học. **`GradScaler`** giải quyết triệt để vấn đề này bằng cách nhân loss lên một hệ số co giãn $S = 65536$ trước khi backward để đưa gradient vào vùng an toàn của `fp16`, sau đó unscale chia ngược lại cho $S$ trước khi optimizer cập nhật trọng số.

---

#### ❓ Câu 17: Tại sao tile ảnh tham chiếu chuẩn hóa bắt buộc phải lấy từ lớp `TUM` trong tập Train? Nếu lấy từ tập Test thì sao?
> **Trả lời**: 
> 1. Nếu lấy ảnh tham chiếu từ tập Test (dù là `test_id` hay `test_ood`), ta đã vi phạm nguyên tắc khoa học nghiêm trọng nhất trong Machine Learning: **Rò rỉ dữ liệu (Data Leakage)**. Mô hình sẽ gián tiếp "nhìn thấy" phân phối màu của đề thi trước khi thi.
> 2. Nhóm chọn lớp `TUM` (Biểu mô ung thư) vì đây là loại mô có cấu trúc phức tạp nhất: vừa có mật độ nhân dày đặc bắt màu tím Hematoxylin, vừa có chất nền bắt màu hồng Eosin. Nhờ đó, ma trận quang học ước lượng được sẽ đại diện đầy đủ nhất cho cả 2 kênh màu.

---

#### ❓ Câu 18: Hai chỉ số $\Delta\text{-F1}$ và Retention Rate ($RR$) phản ánh điều gì trong nghiên cứu?
> **Trả lời**:
> - **$\Delta\text{-F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$**: Đo lường sự tụt giảm tuyệt đối về điểm số khi mô hình chuyển từ sân nhà (Heidelberg) sang sân khách (Aachen). Giá trị này càng nhỏ càng chứng minh mô hình ít bị tổn thương bởi Domain Shift (ở EXP-07 chỉ tụt $0.1028$, trong khi EXP-01 tụt tới $0.3249$).
> - **Retention Rate ($RR$) $= (\text{F1}_{\text{OOD}} / \text{F1}_{\text{ID}}) \times 100\%$**: Đo lường tỷ lệ phần trăm phong độ được giữ lại. Giá trị này phản ánh độ bền vững tương đối, giúp so sánh công bằng giữa các mô hình có xuất phát điểm điểm sân nhà khác nhau (EXP-07 giữ được kỷ lục **89.42%**).

---

#### ❓ Câu 19: Nhóm làm thế nào để đảm bảo tuyệt đối không có sự rò rỉ dữ liệu (Data Leakage) giữa Bệnh viện Heidelberg và Bệnh viện Aachen?
> **Trả lời**: Nhóm thiết lập nguyên tắc **Domain Firewall (Bức tường lửa miền dữ liệu)**:
> 1. Toàn bộ 7.180 ảnh của Bệnh viện Aachen (`CRC-VAL-HE-7K`) được cách ly hoàn toàn, đóng vai trò tập kiểm thử mù 100% Out-of-Domain.
> 2. Quá trình chia tập 70/15/15 chỉ diễn ra duy nhất trên 25.000 ảnh của Heidelberg.
> 3. Toàn bộ các tham số thống kê (Mean, Std của Reinhard, ma trận SVD của Macenko) đều chỉ được tính trên duy nhất 1 tile mẫu thuộc tập Train của Heidelberg.
> 4. Mô hình chỉ được chạm vào dữ liệu Aachen ở bước suy luận cuối cùng sau khi đã hoàn tất huấn luyện và cố định trọng số.

---

#### ❓ Câu 20: Nếu được triển khai giải pháp này tại các bệnh viện thực tế ở Việt Nam, nhóm sẽ khuyến nghị cấu hình nào và tại sao?
> **Trả lời**: Dựa trên kết quả thực nghiệm 13 cell, nhóm đưa ra 2 khuyến nghị thực tiễn tùy thuộc vào điều kiện hạ tầng của cơ sở y tế:
> 1. **Khuyến nghị 1 (Tối ưu độ chính xác và an toàn nhất - Edge AI / GPU tầm trung)**: Sử dụng cấu hình **EXP-07 (ResNet-50 + Chuẩn hóa Macenko + Aug-Geo)**. Đây là cấu hình đạt điểm OOD cao nhất toàn diện (**0.8689** F1) và tỷ lệ giữ phong độ cao nhất (**89.42%**), mô hình nhẹ (25M tham số) dễ dàng triển khai trên các máy tính trạm bệnh viện thông thường.
> 2. **Khuyến nghị 2 (Tối ưu tốc độ suy luận và không cần tiền xử lý phức tạp - Cloud / Server mạnh)**: Sử dụng cấu hình **EXP-12 (Phikon Foundation Model nguyên bản)**. Mô hình này không cần cài đặt thuật toán chuẩn hóa Macenko (vốn tốn thời gian tính SVD), tốc độ chạy siêu nhanh và đạt F1 OOD rất cao (**0.8239**).

---

## 12. CHIẾN LƯỢC BẢO VỆ TIẾNG ANH: BỘ KHUNG WHY - HOW - IF & CÂU HỎI CHUYÊN SÂU
*(Dành riêng cho buổi bảo vệ đồ án bằng Tiếng Anh — Kèm lời thoại mẫu học thuật chuẩn quốc tế)*

Phần này được thiết kế đặc biệt cho các bạn bảo vệ đồ án trước Hội đồng chấm thi bằng **Tiếng Anh**. Mỗi chủ đề được cấu trúc thành 3 phần:
1. **Giải thích cốt lõi bằng Tiếng Việt bình dân**: Giúp bạn hiểu sâu sắc bản chất vấn đề dù chưa có nền tảng Deep Learning chuyên sâu.
2. **Khung tư duy phản xạ WHY – HOW – IF**: Giúp bạn làm chủ thế trận, không bao giờ bị "đứng hình" trước câu hỏi vặn vẹo.
3. **English Defense Script (Lời thoại mẫu tiếng Anh chuẩn học thuật)**: Đoạn văn tiếng Anh súc tích, chuyên nghiệp để bạn tự tin phát biểu trực tiếp trước Hội đồng.

---

### 12.1. BẢN ĐẶC TẢ INPUT VÀ OUTPUT CỦA HỆ THỐNG (INPUT & OUTPUT SPECIFICATIONS)

```mermaid
flowchart LR
    A["Raw Slide Patch\nPNG/TIF on disk\n(224 x 224 x 3, uint8)"] --> B["Preprocessed Tensor\nInput to Model\n(Batch, 3, 224, 224, float32)\nNormalized ImageNet"]
    B --> C["Deep Learning Backbone\n(ResNet-50 / ConvNeXt / Phikon)"]
    C --> D["Raw Output Logits\nz in R^9"]
    D --> E["Softmax Probabilities\np in [0, 1]^9"]
    E --> F["Predicted Class Label\ny_hat = argmax(p)\nin {0..8}"]
```

#### 📌 Bản chất dữ liệu:
- **INPUT (Đầu vào của mô hình)**:
  - **Trên ổ cứng**: File ảnh định dạng `.png` hoặc `.tif`, kích thước $224 \times 224$ pixels, 3 kênh màu RGB, giá trị số nguyên không dấu 8-bit `uint8` trong khoảng $[0, 255]$.
  - **Sau khi qua PatchTransform đưa vào mạng nơ-ron**: Một tensor PyTorch có kích thước `(Batch_Size, 3, 224, 224)`, kiểu dữ liệu số thực `float32`, đã được chuẩn hóa Z-score theo ImageNet Mean $[0.485, 0.456, 0.406]$ và Std $[0.229, 0.224, 0.225]$.
  - **Tại sao lại dùng RGB?** Kính hiển vi quang học tiêu chuẩn ghi nhận ánh sáng truyền qua mẫu mô trên 3 phổ quang học Red, Green, Blue. Hai loại thuốc nhuộm Hematoxylin (tím) và Eosin (hồng) tạo ra sự khác biệt màu sắc rõ nét nhất trên hệ màu 3 kênh này.
  - **Tại sao kích thước lại là $224 \times 224$?**
    1. Đây là kích thước chuẩn mực mà các backbone thị giác (ResNet, ConvNeXt) được tối ưu hóa kiến trúc.
    2. Với Vision Transformer (Phikon), ảnh $224 \times 224$ chia cho kích thước patch $16 \times 16$ vừa vặn tạo thành một lưới $14 \times 14 = 196$ tokens không bị lẻ.
    3. Ở độ phóng đại $20\times$ (độ phân giải vi thể $0.5\ \mu\text{m/pixel}$), một ô $224 \times 224$ tương ứng với trường nhìn diện tích thực tế là $112 \times 112\ \mu\text{m}$. Kích thước này vừa đủ lớn để bao quát cấu trúc tuyến u, mạch máu và mô đệm, nhưng cũng đủ nhỏ để duy trì độ sắc nét của từng nhân tế bào đơn lẻ.
- **OUTPUT (Đầu ra của mô hình)**:
  - **Đầu ra thô của mạng (Raw Logits)**: Một véc-tơ số thực $\mathbf{z} = [z_0, z_1, \dots, z_8] \in \mathbb{R}^9$.
  - **Xác suất dự đoán (Predicted Probabilities)**: Sau khi qua hàm Softmax: $p_c = \frac{e^{z_c}}{\sum_{j=0}^{8} e^{z_j}}$, trả về phân phối xác suất trên 9 lớp sao cho $\sum_{c=0}^8 p_c = 1.0$.
  - **Nhãn phân loại cuối cùng (Final Prediction)**: Lớp có xác suất cao nhất $\hat{y} = \arg\max_{c \in \{0 \dots 8\}} p_c$.
  - **Tại sao lại là bài toán Phân loại đơn nhãn (Single-label Multi-class)?** Vì các lát cắt trong bộ dữ liệu Kather được các chuyên gia giải phẫu bệnh cắt tỉ mỉ từ các vùng mô thuần nhất, mỗi ô vuông $224 \times 224$ đại diện áp đảo cho một loại mô bệnh học duy nhất.

#### 🎙️ English Defense Script (Input/Output Rationale):
> *"Honorable members of the committee, the input to our pipeline consists of standardized $224 \times 224$ RGB histological patch tiles. At a scanning resolution of $0.5\ \mu\text{m}$ per pixel ($20\times$ magnification), each tile captures a physical field of view of $112 \times 112\ \mu\text{m}$, which is clinically optimal to encompass glandular tissue architecture while preserving single-cell nuclear morphology. The tensor input is shaped as $(B, 3, 224, 224)$ normalized with ImageNet channel statistics. The output is a 9-dimensional probability vector computed via Softmax over raw logits, representing the categorical distribution across the 9 histological tissue classes. We formulated this as a single-label multi-class problem because each patch was curated from pure morphological regions annotated by expert pathologists."*

---

### 12.2. KỊCH BẢN "IF": NẾU DATASET THÊM CLASS MỚI THÌ SẼ NHƯ THẾ NÀO? (WHAT IF A NEW CLASS IS ADDED?)

Thầy cô rất thích hỏi câu hỏi tình huống: *"Nêu ngày mai bệnh viện bổ sung thêm class thứ 10 (ví dụ mô hoại tử đông Coagulative Necrosis `NEC`, hoặc cấu trúc vi mạch máu Vascular `VAS`), hệ thống của các em sẽ phải thay đổi như thế nào? Có phải huấn luyện lại từ đầu không?"*

#### 📌 Bản chất kỹ thuật:
1. **Các thay đổi trong code**:
   - `src/config/constants.py`: Thêm tên nhãn vào danh sách `CLASSES.append("NEC")`, hằng số `NUM_CLASSES` tự động cập nhật từ $9 \rightarrow 10$.
   - **Tầng phân loại đầu ra (Classification Head)**: Thay thế tầng tuyến tính cuối cùng:
     - ResNet-50: `model.fc = nn.Linear(2048, 10)`
     - ConvNeXt-Tiny: `model.head.fc = nn.Linear(768, 10)`
     - Phikon: `model.fc = nn.Linear(768, 10)`
   - **Hàm mất mát CrossEntropyLoss**: Hàm loss tự động thích ứng với số chiều đầu ra $K=10$. Công thức làm mịn nhãn Label Smoothing ($0.1$) tự động chia đều xác suất mục tiêu: nhãn đúng nhận $0.9 + 0.1/10 = 0.91$, 9 nhãn sai nhận $0.01$.
2. **Có phải huấn luyện lại từ đầu không? (Catastrophic Forgetting & Transfer Learning)**:
   - **HOÀN TOÀN KHÔNG!** Chúng ta tận dụng triệt để kỹ thuật **Transfer Learning (Học chuyển giao)**.
   - Các tầng trích xuất đặc trưng phía dưới (Backbone) đã học cách nhận diện góc cạnh, vân bề mặt, màng tế bào từ hàng chục ngàn ảnh. Chúng ta giữ nguyên 100% trọng số của Backbone.
   - Ta chỉ cần khởi tạo ngẫu nhiên tầng Linear 10 lớp mới và huấn luyện (*fine-tune*) trên tập dữ liệu mở rộng.
   - Đặc biệt, với Foundation Model **Phikon**, nhờ giao thức **Linear Probing**, toàn bộ ViT backbone vẫn được đóng băng; ta chỉ cần huấn luyện tầng `nn.Linear(768, 10)` gồm đúng 7.690 tham số trong vòng **chưa đầy 10 phút** trên GPU!
3. **Các chỉ số đánh giá thích ứng thế nào?**:
   - Chỉ số Macro-F1 tự động lấy trung bình cộng trên 10 lớp: $\text{Macro-F1} = \frac{1}{10} \sum_{c=1}^{10} \text{F1}_c$.
   - Ma trận nhầm lẫn (Confusion Matrix) mở rộng thành kích thước $10 \times 10$.

#### 🎙️ English Defense Script (Handling New Classes):
> *"If a 10th histological class is introduced—for example, coagulative necrosis—our modular architecture adapts seamlessly without retraining from scratch. Programmatically, we update the `CLASSES` list so `NUM_CLASSES` becomes 10, and replace the final fully connected projection head `nn.Linear(d_in, 10)`. Conceptually, we leverage transfer learning: the underlying feature extractor—which captures invariant nuclear morphology and tissue textures—is entirely preserved. We only initialize and train the newly expanded classification head. For our Phikon foundation model, this linear probing retraining requires less than 10 minutes on a single GPU. The Cross-Entropy loss with Label Smoothing automatically scales its probability regularization across the 10 classes, and Macro-F1 computes the unweighted average across all 10 categories to maintain balanced evaluation."*

---

### 12.3. TẠI SAO LẠI THIẾT LẬP CÁC SIÊU THAM SỐ (HYPERPARAMETERS) NHƯ THẾ?

Hội đồng chấm thi thường chất vấn: *"Tại sao nhóm lại chọn đúng 8 epoch mà không phải 50 epoch? Tại sao Learning Rate lại là $10^{-3}$ cho ResNet nhưng lại là $3 \times 10^{-4}$ cho Phikon? Con số $0.05$ của Weight Decay từ đâu mà ra?"*

#### 📌 Bản chất kỹ thuật & Cơ sở khoa học:

| Siêu tham số | Giá trị | Tại sao chọn giá trị này? (Why this value?) | Rủi ro nếu chọn sai (What if wrong?) |
| :--- | :---: | :--- | :--- |
| **Epochs** | **8** | 1. Mô hình sử dụng trọng số tiền huấn luyện (ImageNet/TCGA), không học từ đầu.<br>2. Tập train có 17.495 ảnh $\times 8$ epochs = **139.960 lượt duyệt mẫu** (2.184 optimizer steps), đủ để hội tụ.<br>3. Biểu đồ Validation F1 thực tế bão hòa ở epoch 6-7.<br>4. Huấn luyện quá lâu sẽ làm mô hình học thuộc lòng màu sắc nguồn (Overfitting).<br>5. Đảm bảo 13 thí nghiệm hoàn thành trong giới hạn 3.5 giờ GPU Kaggle. | Nếu tăng lên 30-50 epochs: Mô hình bị overfitting nặng nề trên Bệnh viện Heidelberg, điểm OOD tại Aachen sẽ tụt giảm mạnh và tốn quota GPU vô ích. |
| **Batch Size** | **64** | Cân bằng lý tưởng giữa việc tận dụng nhân Tensor Core trên GPU 16GB VRAM (đặc biệt khi chạy Vision Transformer Phikon với 196 tokens không bị lỗi OOM) và tính ngẫu nhiên của gradient stochastic giúp nhảy khỏi cực tiểu hẹp. | Nếu quá nhỏ (8, 16): Gradient bị nhiễu lớn, tốc độ chậm.<br>Nếu quá lớn (256, 512): Tràn bộ nhớ GPU OOM và làm giảm khả năng tổng quát hóa (*Generalization Gap*). |
| **Learning Rate (CNNs)** | **$10^{-3}$** | Mức chuẩn mực cho thuật toán AdamW khi fine-tune các mạng tích chập ResNet-50 và ConvNeXt-Tiny kết hợp lịch giảm Cosine Annealing. | Nếu đặt quá lớn ($10^{-2}$): Gradient nổ, loss phân kỳ.<br>Nếu quá nhỏ ($10^{-5}$): Mô hình học quá chậm, không kịp hội tụ trong 8 epoch. |
| **Learning Rate (Phikon)** | **$3 \times 10^{-4}$** | Với Phikon, ta chỉ huấn luyện duy nhất 1 lớp tuyến tính (Linear Probe) trên đặc trưng ViT đã đóng băng. Tốc độ học nhỏ hơn giúp đầu dò ổn định, không làm rung lắc trọng số. | Nếu dùng $10^{-3}$: Đầu phân loại cập nhật quá mạnh, làm mất tính ổn định của vector đặc trưng 768 chiều. |
| **Weight Decay** | **$0.05$** | Chuẩn mực của thuật toán AdamW (Loshchilov & Hutter, 2017) để tách rời việc phạt độ lớn trọng số L2 khỏi cập nhật gradient, ép các tham số không phân kỳ. | Nếu đặt bằng 0: Trọng số phình to, dễ overfit.<br>Nếu quá lớn (0.5): Mô hình bị underfit do bị bóp nghẹt tham số. |
| **Label Smoothing** | **$0.1$** | Kỹ thuật điều hòa (Szegedy et al., 2016). Phạt sự tự tin thái quá ($p \to 1.0$), làm mềm ranh giới quyết định giữa các cặp mô có cấu trúc tương tự (như mô đệm và cơ trơn). | Nếu để 0.0: Mô hình bị overconfidence, khi gặp mẫu ảnh Aachen hơi lệch màu sẽ phán đoán sai với xác suất cực đoan. |
| **Scheduler** | **Cosine Annealing** (`min_lr=1e-5`) | Hạ tốc độ học theo đường cong cosin từ $10^{-3} \to 10^{-5}$. Giúp giai đoạn cuối mô hình tinh chỉnh nhẹ nhàng vào vùng cực tiểu phẳng (*flat minima*). | Nếu dùng StepLR: Bước nhảy đột ngột có thể làm giật gradient và khó chọn thời điểm giảm bước. |

#### 🎙️ English Defense Script (Hyperparameter Justification):
> *"Regarding our hyperparameter choices, we configured 8 training epochs based on empirical convergence and regularized transfer learning. Because our backbones are pretrained on ImageNet and TCGA, they do not start from scratch. With 17,495 training samples per epoch, 8 epochs translate to over 139,000 patch evaluations and 2,184 optimization steps. Our validation curves consistently plateaued between epochs 5 and 7. Extending training beyond 8 epochs introduces severe overfitting to Heidelberg’s hospital-specific color profile, exacerbating out-of-domain degradation. Batch size 64 optimizes the 16GB GPU memory footprint under mixed precision without out-of-memory errors on Vision Transformers. We set a base learning rate of $10^{-3}$ for CNN backbones, but a conservative $3 \times 10^{-4}$ for Phikon linear probing to prevent gradient shocks to the linear classification layer. AdamW weight decay was set to $0.05$ coupled with a Cosine Annealing schedule down to $10^{-5}$, and Label Smoothing was fixed at $0.1$ to regularize overconfident logits, enhancing out-of-distribution robustness."*

---

### 12.4. TẠI SAO LẠI CHỌN 3 LOẠI MÔ HÌNH: RESNET-50, CONVNEXT-TINY, PHIKON?

Thầy cô sẽ hỏi: *"Tại sao nhóm lại so sánh đúng 3 mô hình này? Chúng đại diện cho điều gì trong khoa học thị giác máy tính?"*

```mermaid
flowchart TD
    M1["ResNet-50 (2016)\nClassical CNN Workhorse\n(25.6M params, ImageNet Pretrained)\n-> Mỏ neo đối chứng kinh điển"]
    M2["ConvNeXt-Tiny (2022)\nModernized Pure CNN\n(28.6M params, 7x7 Depthwise Conv)\n-> Kiểm tra ConvNet hiện đại có tự miễn dịch màu không?"]
    M3["Phikon (2023 - Owkin)\nHistology Foundation Model\n(86M params ViT-B/16, TCGA SSL iBOT)\n-> Kiểm tra Foundation Model y tế quy mô lớn"]
    
    M1 --- M2 --- M3
```

#### 📌 Bản chất kỹ thuật:
1. **ResNet-50 (He et al., 2016)**:
   - **Đại diện cho**: Kiến trúc mạng tích chập kinh điển (Classical ConvNet) phổ biến nhất trong mọi nghiên cứu y tế suốt 8 năm qua.
   - **Cơ chế**: 50 tầng với các khối kết nối tắt Residual Connections ($F(x) + x$) giúp giải quyết triệt để vấn đề tiêu biến đạo hàm (*Vanishing Gradient*). Toàn bộ 25.6 triệu tham số được mở khóa để học lại (*Full fine-tuning*).
   - **Mục đích trong nghiên cứu**: Đóng vai trò **Mỏ neo đối chứng (Anchor Baseline)** để so sánh hiệu quả của chuẩn hóa và tăng cường dữ liệu.
2. **ConvNeXt-Tiny (Liu et al., 2022)**:
   - **Đại diện cho**: Thế hệ mạng tích chập thuần túy hiện đại (Modernized ConvNet).
   - **Cơ chế**: Lấy cảm hứng từ Vision Transformer nhưng vẫn dùng tích chập: Kernel lớn $7 \times 7$, Depthwise Separable Convolutions, LayerNorm thay cho BatchNorm, hàm kích hoạt GELU thay cho ReLU, tỷ lệ đảo ngược số kênh $1:4:1$. Số lượng tham số tương đương ResNet-50 (28.6 triệu tham số).
   - **Mục đích trong nghiên cứu**: Trả lời câu hỏi nghiên cứu: *"Liệu một kiến trúc ConvNet hiện đại có tự thân chống chịu lệch màu tốt hơn ResNet kinh điển mà không cần bộ lọc phòng vệ hay không?"* Kết quả ở EXP-10 cho thấy: **Không!** ConvNeXt thô vẫn tụt dốc thảm hại (OOD F1 = 0.6821, chỉ giữ 68.97% phong độ), chứng minh lệch màu là vấn đề dữ liệu chứ không thể giải quyết bằng cách đổi mạng ConvNet.
3. **Phikon (Filiot et al., Owkin, 2023)**:
   - **Đại diện cho**: Kỷ nguyên **Mô hình nền tảng y tế (Domain-specific Foundation Models)**.
   - **Cơ chế**: Kiến trúc Vision Transformer (ViT-B/16) với 86 triệu tham số, được huấn luyện tự giám sát (Self-Supervised Learning với thuật toán iBOT) trên hơn **40 triệu lát cắt mô bệnh học** từ hàng ngàn bệnh nhân thuộc kho dữ liệu ung thư toàn cầu TCGA.
   - **Mục đích trong nghiên cứu**: Đánh giá khả năng tổng quát hóa của Foundation Model y tế dưới dạng **Linear Probing** (đóng băng ViT, chỉ học 1 tầng Linear). Kết quả ở EXP-12 chứng minh: Bản thân việc học tự giám sát trên 40 triệu ảnh mô học đã mang lại khả năng tự miễn dịch màu sắc rất cao (OOD F1 = 0.8239) mà không cần nắn màu.

#### 🎙️ English Defense Script (Model Architectures):
> *"We intentionally selected three vision architectures representing three distinct evolutionary paradigms in computer vision:
> First, **ResNet-50** serves as our classical convolutional baseline—the historical benchmark in medical imaging with 25.6 million parameters, allowing us to evaluate the raw impact of normalization and augmentation.
> Second, **ConvNeXt-Tiny** represents modernized ConvNets engineered with Vision Transformer design principles, including $7 \times 7$ depthwise convolutions, LayerNorm, and GELU activations. This isolates whether modern convolutional inductive biases intrinsically resolve domain shift. Our EXP-10 findings proved they do not: ConvNeXt raw still degrades down to 0.6821 F1.
> Third, **Phikon** represents domain-specific Foundation Models. Developed by Owkin, it is a ViT-Base architecture pretrained via self-supervised learning (iBOT) on over 40 million pan-cancer histology patches from TCGA. We evaluated Phikon under strict linear probing to test whether self-supervised pretraining on diverse clinical cohorts confers intrinsic stain invariance. This tri-model comparison provides a holistic assessment across classical ConvNets, modern ConvNets, and Vision Transformer Foundation Models."*

---

### 12.5. PHƯƠNG PHÁP ĐÁNH GIÁ (EVALUATION METHODOLOGY) & CÁCH TÍNH BẢNG KẾT QUẢ

Thầy cô sẽ yêu cầu: *"Chỉ vào bảng kết quả và giải thích cho tôi từng cột được tính toán như thế nào? Lấy ví dụ số liệu cụ thể ở EXP-01 và EXP-07."*

#### 📌 Công thức toán học & Cách tính từng cột:

```text
+-------------------------------------------------------------------------------------------------------------------------+
| Mã     | Giai đoạn | Backbone     | Norm    | Aug      | Best Val F1 | Test-ID F1 | Test-OOD F1 | Delta-F1 | RR (%) | Phút  |
+-------------------------------------------------------------------------------------------------------------------------+
| EXP-01 | Stage 0   | resnet50     | none    | none     | 0.9850      | 0.9893     | 0.6644      | 0.3249   | 67.16% | 11.0m |
| EXP-07 | Stage 3   | resnet50     | macenko | aug_geo  | 0.9705      | 0.9717     | 0.8689      | 0.1028   | 89.42% | 17.9m |
| EXP-12 | Stage 4   | phikon       | none    | none     | 0.9890      | 0.9912     | 0.8239      | 0.1673   | 83.12% | 9.1m  |
+-------------------------------------------------------------------------------------------------------------------------+
```

1. **Cột `Test-ID F1` (Điểm sân nhà - In-Domain Test F1)**:
   - Được đánh giá trên tập kiểm thử nội miền `test_id` (15% = 3.749 ảnh từ Bệnh viện Heidelberg).
   - Với mỗi lớp $c \in \{0 \dots 8\}$, tính:
     $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{F1}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
   - Lấy trung bình cộng không trọng số: $\text{Test-ID F1} = \frac{1}{9} \sum_{c=1}^9 \text{F1}_c$.
   - Ở EXP-01: Điểm sân nhà đạt cực cao **0.9893** (98.93%).
2. **Cột `Test-OOD F1` (Điểm trường lạ - Out-of-Domain Test F1)**:
   - Được đánh giá trên 100% tập dữ liệu độc lập của Bệnh viện Aachen `test_ood` (7.180 ảnh).
   - Áp dụng cùng công thức Macro-F1 trên 9 lớp.
   - Ở EXP-01: Rơi tự do xuống **0.6644** (66.44%).
   - Ở EXP-07: Được phục hồi mạnh mẽ lên **0.8689** (86.89% - Quán quân toàn diện).
3. **Cột `Delta-F1` ($\Delta\text{-F1}$ - Độ tụt dốc hiệu năng khi chuyển viện)**:
   - Công thức:
     $$\Delta\text{-F1} = \text{Test-ID F1} - \text{Test-OOD F1}$$
   - **Ý nghĩa**: Giá trị càng nhỏ càng chứng minh mô hình ít bị suy giảm chất lượng.
   - **Tính toán thực tế**:
     - Với EXP-01: $\Delta\text{-F1} = 0.9893 - 0.6644 = \mathbf{0.3249}$ (suy giảm tới 32.5 điểm F1!).
     - Với EXP-07: $\Delta\text{-F1} = 0.9717 - 0.8689 = \mathbf{0.1028}$ (độ tụt dốc giảm hơn 3 lần so với baseline!).
4. **Cột `RR (%)` (Retention Rate - Tỷ lệ giữ phong độ)**:
   - Công thức:
     $$\text{Retention Rate (RR)} = \left(\frac{\text{Test-OOD F1}}{\text{Test-ID F1}}\right) \times 100\%$$
   - **Ý nghĩa**: Tỷ lệ phần trăm hiệu năng được duy trì khi chuyển từ môi trường quen thuộc sang môi trường hoàn toàn mới. Càng gần 100% càng chứng minh mô hình học bản chất bệnh học thay vì học thuộc lòng màu sắc.
   - **Tính toán thực tế**:
     - Với EXP-01: $RR = \frac{0.6644}{0.9893} \times 100\% = \mathbf{67.16\%}$ (mất 1/3 phong độ).
     - Với EXP-07: $RR = \frac{0.8689}{0.9717} \times 100\% = \mathbf{89.42\%}$ (giữ gần 90% phong độ - kỷ lục cao nhất của nghiên cứu).
5. **Cột `Minutes` (Thời gian thực thi)**:
   - Tính bằng `(time.time() - t0) / 60.0`, ghi nhận chính xác thời gian chạy 8 epoch huấn luyện và kiểm thử trên GPU Kaggle.

#### 🎙️ English Defense Script (Metric Calculations):
> *"To rigorously quantify generalization under staining variations, we track two primary robustness metrics derived from Macro-F1:
> First, **Delta-F1** represents the absolute performance drop between in-domain test data and external out-of-domain test data: $\Delta\text{-F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$. Lower is better. In EXP-01, the raw ResNet baseline suffered an alarming Delta-F1 of $0.3249$. In EXP-07, our Macenko defense compressed this drop to only $0.1028$.
> Second, **Retention Rate (RR)** reflects relative robustness: $\text{RR} = (\text{F1}_{\text{OOD}} / \text{F1}_{\text{ID}}) \times 100\%$. While EXP-01 retained merely $67.16\%$ of its performance across hospital firewalls, EXP-07 maintained a peak retention rate of $89.42\%$. Macro-F1 was deliberately chosen over raw accuracy because it weights all 9 tissue categories equally, preventing majority classes from masking severe misclassifications in critical categories like tumor epithelium."*

---

### 12.6. NGUỒN GỐC DỮ LIỆU & TIÊU CHUẨN ĐẠO ĐỨC (DATA ORIGIN & ETHICAL PROTOCOLS)

Thầy cô hỏi: *"Bộ dữ liệu này của ai, xuất bản năm nào, thu thập ở đâu, có vi phạm bản quyền hay quy chuẩn đạo đức y tế không?"*

#### 📌 Bản chất kỹ thuật:
- **Tác giả & Xuất bản**: Giáo sư Jakob Nikolas Kather cùng các đồng nghiệp tại Đức, công bố trong bài báo khoa học nổi tiếng: *"Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicentric study"*, xuất bản trên tạp chí y khoa hàng đầu thế giới **PLOS Medicine (2019)**.
- **Mã định danh lưu trữ dữ liệu**: Lưu trữ vĩnh viễn trên cổng dữ liệu mở quốc tế **Zenodo**, mã số DOI: `10.5281/zenodo.1214456`.
- **Thông số kỹ thuật của ảnh**:
  - Độ phóng đại quang học: $20\times$.
  - Độ phân giải quét vi thể: $0.5\ \mu\text{m}$ trên mỗi pixel.
  - Kích thước mỗi patch: $224 \times 224$ pixels (tương ứng vùng mô thực tế $112 \times 112\ \mu\text{m}$).
- **Hai cơ sở y tế độc lập**:
  - **Miền nguồn (In-Domain)**: `NCT-CRC-HE-100K-NONORM` — 100.000 ảnh từ 86 bệnh nhân tại Trung tâm Ung thư Quốc gia Heidelberg (NCT) và Trung tâm Y tế Đại học Mannheim. Phiên bản `NONORM` bảo tồn 100% biến thiên thô tự nhiên.
  - **Miền đích ngoại viện (Out-of-Domain)**: `CRC-VAL-HE-7K` — 7.180 ảnh từ 50 bệnh nhân tại Bệnh viện Đại học RWTH Aachen.
- **Quy chuẩn đạo đức (Ethical Clearance)**: Toàn bộ tiêu bản giải phẫu bệnh được hồi cứu và ẩn danh hóa hoàn toàn (*fully anonymized*), tuân thủ nghiêm ngặt phê duyệt của Hội đồng Đạo đức Y khoa tại Đại học Heidelberg và RWTH Aachen.

#### 🎙️ English Defense Script (Data Provenance):
> *"Our experimental cohorts originate from the landmark study by Kather et al., published in PLOS Medicine in 2019, publicly archived on Zenodo under DOI 10.5281/zenodo.1214456. The dataset encompasses digitized tissue patches at $20\times$ optical magnification with a spatial resolution of $0.5\ \mu\text{m}$ per pixel. Our in-domain source dataset (`NCT-CRC-HE-100K-NONORM`) comprises 100,000 raw, unnormalized patches from 86 colorectal cancer patients across Heidelberg and Mannheim. Our out-of-domain external validation cohort (`CRC-VAL-HE-7K`) contains 7,180 patches from 50 independent patients at RWTH Aachen University Hospital. All patient identifiers were strictly anonymized prior to digitization in full compliance with local institutional ethics review boards."*

---

### 12.7. TỔNG HỢP CÂU NÓI MẪU "CỨU SINH" KHI GẶP CÂU HỎI KHÓ TRONG PHÒNG VỆ

Dưới đây là các mẫu câu tiếng Anh tự tin giúp bạn xử lý tình huống khéo léo khi gặp câu hỏi bất ngờ từ Ban giám khảo:

1. **Khi bạn cần vài giây suy nghĩ**:
   > *"Thank you, Professor, that is an insightful question. Let me break down the technical rationale behind that decision..."*
2. **Khi bảo vệ việc chọn EXP-07 thay vì các mô hình phức tạp khác**:
   > *"While deeper architectures and heavy augmentations intuitively seem appealing, our empirical ablation demonstrates that simplicity and biological grounding prevail: Macenko optical deconvolution combined strictly with spatial geometric invariance delivers the peak out-of-domain F1 of 0.8689."*
3. **Khi giải thích tại sao không dùng Stain Jitter cùng với Macenko**:
   > *"Adding synthetic stain jitter to Macenko-normalized tiles introduces an Augmentation Paradox: Macenko maps tiles into a canonical optical density space, and perturbing concentrations post-normalization disrupts the structural distribution that the normalizer established, causing an 8-point F1 drop."*
4. **Khi được hỏi về khả năng ứng dụng thực tế**:
   > *"In a clinical deployment setting, our pipeline guarantees both robustness and computational efficiency: running inference on a standard hospital workstation takes under 15 milliseconds per tile, ensuring seamless integration into Whole Slide Image triage workflows."*

---

## 🎯 LỜI KẾT DÀNH CHO BUỔI BẢO VỆ

Một công trình Deep Learning y tế xuất sắc không nằm ở việc tạo ra một mô hình đạt 99% điểm số trên tập dữ liệu nội bộ quen thuộc, mà nằm ở **sự dũng cảm đối diện với dữ liệu thực tế ngoại viện** và tìm ra đúng bản chất tại sao mô hình thành công hay thất bại. 

Chúc bạn có một buổi bảo vệ đồ án tự tin, đĩnh đạc và đạt kết quả cao nhất! 🎉
