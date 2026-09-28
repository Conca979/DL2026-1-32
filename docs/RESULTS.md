# Experimental Results: Robust Histopathology Classification under Staining Variations

## 1. Overview & Evaluation Protocol

This document tracks the empirical findings of the **13-cell ablation study** evaluating stain robustness in digital histopathology.

- **In-Domain (Source Domain)**: `NCT-CRC-HE-100K-NONORM` (100,000 raw patches from Heidelberg & Mannheim, Germany).
- **Out-of-Domain (Target Domain)**: `CRC-VAL-HE-7K` (7,180 patches from 50 independent patients at University Hospital RWTH Aachen, Germany; strictly domain-firewalled).
- **Primary Metric**: **Macro-averaged F1 Score (`Macro-F1`)** across all 9 colorectal tissue classes.
- **Robustness Quantifiers**:
  - **Stain Drop**: `Delta_F1 = F1_ID - F1_OOD` (lower is better; 0 indicates perfect stain invariance).
  - **Retention Rate**: `RR_F1 = (F1_OOD / F1_ID) * 100%` (higher is better; 100% indicates full retention of diagnostic capability across hospitals).

---

## 2. Master Results Table (13-Cell Matrix)

> [!NOTE]
> The table below reflects the experimental schema executed by `run_experiments.py`. Final empirical numbers are written automatically to `results/RESULTS_TABLE.md` and `results/summary_results.csv` after the run.

| Exp ID | Stage | Backbone | Normalization | Augmentation | Val F1 | ID F1 | OOD F1 | Delta-F1 (Drop) | Retention Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Stage 0 | ResNet-50 | None | None | — | — | — | — | — | Pending Run |
| **EXP-02** | Stage 1 | ResNet-50 | Reinhard | None | — | — | — | — | — | Pending Run |
| **EXP-03** | Stage 1 | ResNet-50 | Macenko | None | — | — | — | — | — | Pending Run |
| **EXP-04** | Stage 2 | ResNet-50 | None | Aug-Geo | — | — | — | — | — | Pending Run |
| **EXP-05** | Stage 2 | ResNet-50 | None | Aug-Stain | — | — | — | — | — | Pending Run |
| **EXP-06** | Stage 2 | ResNet-50 | None | Aug-Combined | — | — | — | — | — | Pending Run |
| **EXP-07** | Stage 3 | ResNet-50 | Macenko | Aug-Geo | — | — | — | — | — | Pending Run |
| **EXP-08** | Stage 3 | ResNet-50 | Macenko | Aug-Stain | — | — | — | — | — | Pending Run |
| **EXP-09** | Stage 3 | ResNet-50 | Macenko | Aug-Combined | — | — | — | — | — | Pending Run |
| **EXP-10** | Stage 4 | ConvNeXt-Tiny | None | None | — | — | — | — | — | Pending Run |
| **EXP-11** | Stage 4 | ConvNeXt-Tiny | Macenko | Aug-Combined | — | — | — | — | — | Pending Run |
| **EXP-12** | Stage 4 | Phikon | None | None | — | — | — | — | — | Pending Run |
| **EXP-13** | Stage 4 | Phikon | Macenko | Aug-Combined | — | — | — | — | — | Pending Run |

### Legend
- **Aug-Geo**: Spatial geometric invariance (random horizontal/vertical flips, 90° rotations).
- **Aug-Stain**: Biologically-grounded H&E stain concentration perturbation via optical density deconvolution.
- **Aug-Combined**: Composition of geometric transforms and H&E stain jitter.
- **Phikon**: Foundation model for pathology (ViT-B/16 pretrained on ~40M TCGA patches via iBOT), evaluated via frozen-feature linear probing.

---

## 3. Scientific Hypotheses & Expected Dynamics

```text
+--------------------------------------------------------------------------+
|                        STAGE-WISE RESEARCH QUESTIONS                     |
+--------------------------------------------------------------------------+
| Stage 0 (EXP-01):                                                        |
|   Baseline out-of-the-box performance. Establishes the magnitude of      |
|   clinical stain shift (typically a 10-25% drop from ID to OOD).         |
|                                                                          |
| Stage 1 (EXP-02 vs EXP-03):                                              |
|   Physical deconvolution (Macenko) vs statistical transfer (Reinhard).   |
|   Macenko isolates actual dye absorption vectors, typically yielding     |
|   superior boundary preservation in dark lymphocyte clusters.            |
|                                                                          |
| Stage 2 (EXP-04, 05, 06):                                                |
|   Can synthetic stain jitter match explicit physical normalization?      |
|   Does combining spatial and color augmentations provide synergy?        |
|                                                                          |
| Stage 3 (EXP-07, 08, 09):                                                |
|   Defense interaction. Does stain jitter conflict with pre-normalized    |
|   tiles, or does it confer additive robustness?                          |
|                                                                          |
| Stage 4 (EXP-10, 11, 12, 13):                                            |
|   Modern architectures & Foundation Models.                              |
|   - Does ConvNeXt-Tiny's 7x7 depthwise convolutions resist stain shifts? |
|   - Does Phikon's massive TCGA pretraining possess intrinsic stain       |
|     invariance even without test-time normalization?                     |
+--------------------------------------------------------------------------+
```

---

## 4. Nine Histological Classes (Target Domain Distribution)

| Class Code | Tissue Description | Out-of-Domain Count (`CRC-VAL-HE-7K`) | Clinical Diagnostic Role |
| :--- | :--- | :---: | :--- |
| `ADI` | Adipose | 1,338 | Background fatty tissue; high fat vacuoles. |
| `BACK` | Background | 1,056 | Glass slide / mounting medium; pure optical transmission. |
| `DEB` | Debris | 339 | Necrosis, hemorrhages; high morphological variation. |
| `LYM` | Lymphocytes | 638 | Immune infiltration; dark hyperchromatic round nuclei. |
| `MUC` | Mucus | 617 | Extracellular mucin pools; faint amphophilic staining. |
| `MUS` | Smooth Muscle | 592 | Colonic muscularis propria; eosinophilic fiber bundles. |
| `NORM` | Normal Mucosa | 876 | Non-neoplastic colon crypts with goblet cells. |
| `STR` | Stroma | 421 | Desmoplastic cancer-associated stroma. |
| `TUM` | Colorectal Carcinoma | 1,303 | Primary tumor epithelium; dysplastic glandular architecture. |
| **Total** | | **7,180** | **50 Independent Patients** |

---

## 5. Instructions for Syncing Results from Kaggle

When `notebook.ipynb` finishes executing on Kaggle:
1. Open the **Output** tab in the Kaggle notebook.
2. Locate `/kaggle/working/results/RESULTS_TABLE.md`.
3. Copy the populated Markdown table into Section 2 of this file.
4. Download `/kaggle/working/results/summary_results.csv` into `results/` for downstream plotting or statistical analysis.
