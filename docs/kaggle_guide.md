# Kaggle Execution Guide: 13-Cell Ablation Study

A concise, step-by-step guide to running the entire 13-cell histopathology stain robustness study on Kaggle's free GPU tier.

---

## 1. Prerequisites

1. **Kaggle Account**: Phone-verified (required to access free GPU accelerators and Internet).
2. **Datasets** (Zenodo Record [1214456](https://doi.org/10.5281/zenodo.1214456), CC-BY 4.0):
   - `NCT-CRC-HE-100K-NONORM.zip` (Source domain: 100,000 raw H&E patches, 9 classes).
   - `CRC-VAL-HE-7K.zip` (Target domain: 7,180 external validation patches, 50 patients).
3. **Code Archive**:
   - Run `python scripts/make_zips.py` to generate `dist/histo-robust-code.zip` (7.2 KB containing `run_experiments.py`).

---

## 2. Dataset Setup on Kaggle

In Kaggle, navigate to **Datasets** → **New Dataset** for each input:

| Dataset Title (Slug) | File to Upload | Purpose |
| :--- | :--- | :--- |
| `histo-robust-code` | `dist/histo-robust-code.zip` | Standalone experiment pipeline script. |
| `nct-crc-he-100k-nonorm` | `dist/NCT-CRC-HE-100K-NONORM.zip` | In-domain training & validation data. |
| `crc-val-he-7k` | `dist/CRC-VAL-HE-7K.zip` | Strictly firewalled out-of-domain evaluation. |

*Set visibility to **Private**.*

---

## 3. Notebook Configuration

1. In Kaggle, click **Code** → **New Notebook**.
2. Go to **File** → **Import Notebook** and upload [`notebook.ipynb`](../notebook.ipynb).
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: `GPU T4 x2`.
   - **Internet**: `On` (required to download ImageNet weights and `owkin/phikon`).
   - **Persistence**: `Files only`.
4. Click **Add Input** (or **+ Add Data**), select **Your Datasets**, and attach all three datasets:
   - `histo-robust-code`
   - `nct-crc-he-100k-nonorm`
   - `crc-val-he-7k`

---

## 4. Execution & Expected Timeline

Click **Run All** (or **Save Version** → **Save & Run All (Commit)** to run in the background).

### Notebook Cells Overview

| Cell | Action | Runtime |
| :---: | :--- | :---: |
| **Cell 1** | Installs dependencies (`timm`, `transformers`, `tabulate`) & asserts GPU availability. | ~30s |
| **Cell 2** | Automatically discovers dataset directories and copies `run_experiments.py` to `/kaggle/working/`. | ~5s |
| **Cell 3** | Executes all 13 ablation experiments sequentially with live streaming logs. | ~3.5h |
| **Cell 4** | Displays the final Markdown ablation summary table and verifies CSV export. | <5s |

### Progression Across Stages

```text
EXP-01           (Stage 0: Raw ResNet-50 Baseline)
EXP-02 -> EXP-03 (Stage 1: Reinhard vs. Macenko Color Normalization)
EXP-04 -> EXP-06 (Stage 2: Spatial, HED Stain Jitter, Combined Augmentation)
EXP-07 -> EXP-09 (Stage 3: Normalization x Augmentation Defense Interaction)
EXP-10 -> EXP-11 (Stage 4: Modern ConvNeXt-Tiny Raw vs. Defended)
EXP-12 -> EXP-13 (Stage 4: Phikon Pathology Foundation Model Raw vs. Defended)
```

---

## 5. Artifact Retrieval

Once Cell 4 finishes, inspect or download the following files from the Kaggle **Output** tab (`/kaggle/working/results/`):

- `RESULTS_TABLE.md`: The formatted Markdown table ready to paste directly into your project report or [`docs/RESULTS.md`](RESULTS.md).
- `summary_results.csv`: Complete numerical records (`exp_id`, `stage`, `backbone`, `norm`, `aug`, `val_f1`, `test_id_f1`, `test_ood_f1`, `delta_f1`, `rr_f1`, `runtime`).
