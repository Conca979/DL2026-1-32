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
> Tthe table below reflects the experimental schema, template executed by `run_experiments.py`. Final empirical numbers will be pasted on `results/RESULTS_TABLE.md` and `results/summary_results.csv` after the run.

# Final Ablation Results

| exp_id   | stage   | backbone      | norm     | aug          |   best_val_f1 |   test_id_f1 |   test_ood_f1 |   delta_f1 |   rr_f1 |   test_id_acc |   test_ood_acc |   minutes |
|:------|:-----|:------|:-------|:--------|----------:|--------:|--------:|-------:|-------:|-------:|------:|-------:|
| EXP-01   | Stage 0 | resnet50      | none     | none         |        0.9917 |       0.9893 |        0.6644 |     0.3249 |   67.16 |        0.9893 |         0.7318 |     11.01 |
| EXP-02   | Stage 1 | resnet50      | reinhard | none         |        0.9872 |       0.9862 |        0.8488 |     0.1373 |   86.07 |        0.9861 |         0.8915 |     16.53 |
| EXP-03   | Stage 1 | resnet50      | macenko  | none         |        0.9693 |       0.9653 |        0.809  |     0.1563 |   83.8  |        0.9653 |         0.846  |     17.27 |
| EXP-04   | Stage 2 | resnet50      | none     | aug_geo      |        0.9915 |       0.9899 |        0.6172 |     0.3727 |   62.35 |        0.9899 |         0.6955 |     10.85 |
| EXP-05   | Stage 2 | resnet50      | none     | aug_stain    |        0.9859 |       0.9877 |        0.7443 |     0.2434 |   75.36 |        0.9877 |         0.8035 |     10.88 |
| EXP-06   | Stage 2 | resnet50      | none     | aug_combined |        0.988  |       0.9901 |        0.7105 |     0.2796 |   71.76 |        0.9901 |         0.7734 |     10.94 |
| EXP-07   | Stage 3 | resnet50      | macenko  | aug_geo      |        0.9749 |       0.9717 |        0.8689 |     0.1028 |   89.42 |        0.9717 |         0.9    |     17.91 |
| EXP-08   | Stage 3 | resnet50      | macenko  | aug_stain    |        0.9651 |       0.9619 |        0.785  |     0.177  |   81.6  |        0.9619 |         0.8276 |     20.04 |
| EXP-09   | Stage 3 | resnet50      | macenko  | aug_combined |        0.972  |       0.9699 |        0.8604 |     0.1094 |   88.72 |        0.9699 |         0.8921 |     20.65 |
| EXP-10   | Stage 4 | convnext_tiny | none     | none         |        0.9896 |       0.9891 |        0.6821 |     0.3069 |   68.97 |        0.9891 |         0.7558 |     12.98 |
| EXP-11   | Stage 4 | convnext_tiny | macenko  | aug_combined |        0.9757 |       0.9763 |        0.8685 |     0.1077 |   88.97 |        0.9763 |         0.8997 |     20.7  |
| EXP-12   | Stage 4 | phikon        | none     | none         |        0.9915 |       0.9912 |        0.8239 |     0.1673 |   83.12 |        0.9912 |         0.8623 |      9.08 |
| EXP-13   | Stage 4 | phikon        | macenko  | aug_combined |        0.9018 |       0.905  |        0.6982 |     0.2068 |   77.15 |        0.9045 |         0.7398 |     20.11 |


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
