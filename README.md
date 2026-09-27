# Robust Histopathology Image Classification under Staining Variations

A clean, reproducible 13-cell ablation study evaluating what makes deep learning classifiers robust to clinical histopathology staining variations.

* **In-Domain (Source)**: `NCT-CRC-HE-100K-NONORM` (100,000 raw unnormalized H&E colorectal cancer tiles, Heidelberg/Mannheim).
* **Out-of-Domain (Target)**: `CRC-VAL-HE-7K` (7,180 tiles from 50 independent patients at RWTH Aachen; strictly firewall-protected).

---

## The 13-Cell Ablation Matrix

| Exp ID | Stage | Backbone | Normalization | Augmentation | Research Question |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-01** | Stage 0 | ResNet-50 | None | None | **Primary Baseline**: Out-of-the-box performance under raw staining. |
| **EXP-02** | Stage 1 | ResNet-50 | Reinhard | None | Does statistical LAB color transfer improve out-of-domain generalization? |
| **EXP-03** | Stage 1 | ResNet-50 | Macenko | None | Does optical density H&E deconvolution outperform statistical transfer? |
| **EXP-04** | Stage 2 | ResNet-50 | None | Aug-Geo | What portion of robustness is attributable purely to spatial invariance? |
| **EXP-05** | Stage 2 | ResNet-50 | None | Aug-Stain | Does synthetic HED stain jitter match or exceed explicit normalization? |
| **EXP-06** | Stage 2 | ResNet-50 | None | Aug-Combined | Are spatial and stain augmentations additive when used without normalization? |
| **EXP-07** | Stage 3 | ResNet-50 | Macenko | Aug-Geo | Does combining Macenko with spatial augmentation yield gains? |
| **EXP-08** | Stage 3 | ResNet-50 | Macenko | Aug-Stain | Does stain jitter complement or conflict with normalized tiles? |
| **EXP-09** | Stage 3 | ResNet-50 | Macenko | Aug-Combined | **Defended ResNet Baseline**: Peak performance of ResNet under full defense. |
| **EXP-10** | Stage 4 | ConvNeXt-Tiny | None | None | Is modern ConvNet architecture intrinsically more robust than ResNet? |
| **EXP-11** | Stage 4 | ConvNeXt-Tiny | Macenko | Aug-Combined | Does ConvNeXt-Tiny benefit from the combined defense policy? |
| **EXP-12** | Stage 4 | Phikon | None | None | Does large-scale histology SSL pretraining (TCGA iBOT) provide intrinsic stain invariance? |
| **EXP-13** | Stage 4 | Phikon | Macenko | Aug-Combined | Can foundation model representations be further boosted by defenses? |

---

## Quickstart

### 1. Kaggle Execution (1 Click)
All 13 experiments train in **~3.5 hours** on a single Kaggle GPU P100:
1. Upload [`notebook.ipynb`](notebook.ipynb) to Kaggle via **File -> Import Notebook**.
2. Set **Accelerator** and **Internet**: `On`.
3. Attach the two dataset and codebase inputs (`nct-crc-he-100k-nonorm`, `crc-val-he-7k` and `histo-robust-code`).
4. Click **Run All** (or **Save & Run All**).
5. The final markdown table is automatically exported to `results/RESULTS_TABLE.md`.

### 2. Local Verification (< 1 second)
```bash
uv run python tests/run_all.py          # Verifies pipeline, normalization, and transforms
uv run python run_experiments.py --dry-run   # Prints execution plan
```

### 3. Build Kaggle Code Zip (Suggest read more on docs/KAGGLE_GUIDE.md)
```bash
uv run python scripts/make_zips.py      # Produces dist/histo-robust-code.zip (7.2 KB)
```

---

## Repository Structure

```text
├── README.md               # 1-page project overview & 13-cell matrix
├── run_experiments.py      # Complete standalone training & evaluation pipeline (~500 lines)
├── notebook.ipynb          # 5-cell Kaggle execution notebook
├── pyproject.toml          # Minimal environment dependencies (managed by uv)
├── docs/                   # Scientific documentation (exactly 4 files)
│   ├── PLAN.md             # Research problem, 13-cell matrix & evaluation math
│   ├── RESULTS.md          # 13-cell master results tracking table & hypotheses
│   ├── dataset_card.md     # 9 classes, split rules, and domain firewall
│   └── kaggle_guide.md     # 1-click execution guide for Kaggle GPU
├── scripts/
│   └── make_zips.py        # Kaggle archive packager (packs only run_experiments.py)
└── tests/
    ├── test_pipeline.py    # Offline unit tests
    └── run_all.py          # Instant test runner
```