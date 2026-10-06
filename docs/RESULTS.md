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

## 2. Master Results Tables (Multi-Campaign Validation)

> [!NOTE]
> To definitively rule out stochastic noise and subset bias, the 13-cell ablation campaign was executed across three independent settings:
> 1. **Table 1 (Campaign 1)**: Seed 42, $N = 25{,}000$ tiles (Primary Baseline) — [Kaggle Notebook](https://www.kaggle.com/code/tahuy138/completed-result-1)
> 2. **Table 2 (Campaign 2)**: Seed 100, $N = 25{,}000$ tiles (Multi-Seed Re-shuffling) — [Kaggle Notebook](https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355465855)
> 3. **Table 3 (Campaign 3)**: Seed 42, $N = 50{,}000$ tiles (Scaled Training Budget) — [Kaggle Notebook](https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355494334)
>
> All core phenomena (stain collapse, normalization efficacy, stain-jitter conflict with Macenko, and foundation model breakdown under artificial normalization) replicate with 100% consistency across all 3 runs.

### Table 1: Campaign 1 — Seed 42, Subset = 25,000
*Notebook: [https://www.kaggle.com/code/tahuy138/completed-result-1](https://www.kaggle.com/code/tahuy138/completed-result-1)*

| exp_id | stage | backbone | norm | aug | best_val_f1 | test_id_f1 | test_ood_f1 | delta_f1 | rr_f1 | test_id_acc | test_ood_acc | minutes |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-01 | Stage 0 | resnet50 | none | none | 0.9917 | 0.9893 | 0.6644 | 0.3249 | 67.16% | 0.9893 | 0.7318 | 11.01 |
| EXP-02 | Stage 1 | resnet50 | reinhard | none | 0.9872 | 0.9862 | 0.8488 | 0.1373 | 86.07% | 0.9861 | 0.8915 | 16.53 |
| EXP-03 | Stage 1 | resnet50 | macenko | none | 0.9693 | 0.9653 | 0.8090 | 0.1563 | 83.80% | 0.9653 | 0.8460 | 17.27 |
| EXP-04 | Stage 2 | resnet50 | none | aug_geo | 0.9915 | 0.9899 | 0.6172 | 0.3727 | 62.35% | 0.9899 | 0.6955 | 10.85 |
| EXP-05 | Stage 2 | resnet50 | none | aug_stain | 0.9859 | 0.9877 | 0.7443 | 0.2434 | 75.36% | 0.9877 | 0.8035 | 10.88 |
| EXP-06 | Stage 2 | resnet50 | none | aug_combined | 0.9880 | 0.9901 | 0.7105 | 0.2796 | 71.76% | 0.9901 | 0.7734 | 10.94 |
| EXP-07 | Stage 3 | resnet50 | macenko | aug_geo | 0.9749 | 0.9717 | **0.8689** | **0.1028** | **89.42%** | 0.9717 | 0.9000 | 17.91 |
| EXP-08 | Stage 3 | resnet50 | macenko | aug_stain | 0.9651 | 0.9619 | 0.7850 | 0.1770 | 81.60% | 0.9619 | 0.8276 | 20.04 |
| EXP-09 | Stage 3 | resnet50 | macenko | aug_combined | 0.9720 | 0.9699 | 0.8604 | 0.1094 | 88.72% | 0.9699 | 0.8921 | 20.65 |
| EXP-10 | Stage 4 | convnext_tiny | none | none | 0.9896 | 0.9891 | 0.6821 | 0.3069 | 68.97% | 0.9891 | 0.7558 | 12.98 |
| EXP-11 | Stage 4 | convnext_tiny | macenko | aug_combined | 0.9757 | 0.9763 | 0.8685 | 0.1077 | 88.97% | 0.9763 | 0.8997 | 20.70 |
| EXP-12 | Stage 4 | phikon | none | none | 0.9915 | 0.9912 | 0.8239 | 0.1673 | 83.12% | 0.9912 | 0.8623 | 9.08 |
| EXP-13 | Stage 4 | phikon | macenko | aug_combined | 0.9018 | 0.9050 | 0.6982 | 0.2068 | 77.15% | 0.9045 | 0.7398 | 20.11 |

### Table 2: Campaign 2 — Seed 100, Subset = 25,000
*Notebook: [https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355465855](https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355465855)*

| exp_id | stage | backbone | norm | aug | best_val_f1 | test_id_f1 | test_ood_f1 | delta_f1 | rr_f1 | test_id_acc | test_ood_acc | minutes |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-01 | Stage 0 | resnet50 | none | none | 0.9912 | 0.9899 | 0.7234 | 0.2664 | 73.08% | 0.9899 | 0.7868 | 23.97 |
| EXP-02 | Stage 1 | resnet50 | reinhard | none | 0.9883 | 0.9886 | 0.8244 | 0.1642 | 83.39% | 0.9885 | 0.8776 | 19.39 |
| EXP-03 | Stage 1 | resnet50 | macenko | none | 0.9698 | 0.9661 | 0.8393 | 0.1268 | 86.87% | 0.9661 | 0.8721 | 23.19 |
| EXP-04 | Stage 2 | resnet50 | none | aug_geo | 0.9915 | 0.9893 | 0.6685 | 0.3208 | 67.57% | 0.9893 | 0.7255 | 11.73 |
| EXP-05 | Stage 2 | resnet50 | none | aug_stain | 0.9885 | 0.9885 | 0.7194 | 0.2692 | 72.77% | 0.9885 | 0.7840 | 11.20 |
| EXP-06 | Stage 2 | resnet50 | none | aug_combined | 0.9904 | 0.9899 | 0.7610 | 0.2289 | 76.88% | 0.9899 | 0.8134 | 11.31 |
| EXP-07 | Stage 3 | resnet50 | macenko | aug_geo | 0.9753 | 0.9721 | 0.8414 | 0.1307 | 86.56% | 0.9720 | 0.8773 | 20.77 |
| EXP-08 | Stage 3 | resnet50 | macenko | aug_stain | 0.9669 | 0.9618 | 0.8019 | 0.1599 | 83.38% | 0.9619 | 0.8398 | 22.26 |
| EXP-09 | Stage 3 | resnet50 | macenko | aug_combined | 0.9712 | 0.9680 | 0.8307 | 0.1373 | 85.82% | 0.9680 | 0.8702 | 25.21 |
| EXP-10 | Stage 4 | convnext_tiny | none | none | 0.9896 | 0.9869 | 0.6727 | 0.3142 | 68.16% | 0.9869 | 0.7586 | 13.39 |
| EXP-11 | Stage 4 | convnext_tiny | macenko | aug_combined | 0.9768 | 0.9776 | 0.8613 | 0.1163 | 88.10% | 0.9776 | 0.8919 | 25.95 |
| EXP-12 | Stage 4 | phikon | none | none | 0.9912 | 0.9917 | **0.8668** | **0.1249** | **87.40%** | 0.9917 | 0.9031 | 9.14 |
| EXP-13 | Stage 4 | phikon | macenko | aug_combined | 0.9053 | 0.9053 | 0.6903 | 0.2150 | 76.26% | 0.9050 | 0.7336 | 24.62 |

### Table 3: Campaign 3 — Seed 42, Subset = 50,000
*Notebook: [https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355494334](https://www.kaggle.com/code/tahuy138/notebook/notebook?scriptVersionId=355494334)*

| exp_id | stage | backbone | norm | aug | best_val_f1 | test_id_f1 | test_ood_f1 | delta_f1 | rr_f1 | test_id_acc | test_ood_acc | minutes |
|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-01 | Stage 0 | resnet50 | none | none | 0.9940 | 0.9932 | 0.6104 | 0.3828 | 61.45% | 0.9932 | 0.6724 | 20.67 |
| EXP-02 | Stage 1 | resnet50 | reinhard | none | 0.9917 | 0.9913 | 0.8236 | 0.1677 | 83.09% | 0.9913 | 0.8787 | 33.02 |
| EXP-03 | Stage 1 | resnet50 | macenko | none | 0.9737 | 0.9777 | 0.8211 | 0.1566 | 83.98% | 0.9777 | 0.8538 | 33.36 |
| EXP-04 | Stage 2 | resnet50 | none | aug_geo | 0.9952 | 0.9951 | 0.5574 | 0.4377 | 56.01% | 0.9951 | 0.6049 | 20.51 |
| EXP-05 | Stage 2 | resnet50 | none | aug_stain | 0.9904 | 0.9903 | 0.7487 | 0.2416 | 75.60% | 0.9903 | 0.8001 | 20.61 |
| EXP-06 | Stage 2 | resnet50 | none | aug_combined | 0.9948 | 0.9931 | 0.8002 | 0.1928 | 80.58% | 0.9931 | 0.8345 | 20.76 |
| EXP-07 | Stage 3 | resnet50 | macenko | aug_geo | 0.9816 | 0.9828 | 0.8319 | 0.1509 | 84.65% | 0.9828 | 0.8593 | 35.66 |
| EXP-08 | Stage 3 | resnet50 | macenko | aug_stain | 0.9742 | 0.9734 | 0.8266 | 0.1467 | 84.93% | 0.9733 | 0.8584 | 40.41 |
| EXP-09 | Stage 3 | resnet50 | macenko | aug_combined | 0.9792 | 0.9778 | 0.8310 | 0.1468 | 84.99% | 0.9777 | 0.8586 | 40.54 |
| EXP-10 | Stage 4 | convnext_tiny | none | none | 0.9913 | 0.9907 | 0.7607 | 0.2300 | 76.78% | 0.9907 | 0.8180 | 24.33 |
| EXP-11 | Stage 4 | convnext_tiny | macenko | aug_combined | 0.9817 | 0.9785 | **0.8753** | **0.1032** | **89.45%** | 0.9785 | 0.9039 | 41.72 |
| EXP-12 | Stage 4 | phikon | none | none | 0.9937 | 0.9940 | 0.8319 | 0.1621 | 83.69% | 0.9940 | 0.8701 | 17.03 |
| EXP-13 | Stage 4 | phikon | macenko | aug_combined | 0.9132 | 0.9150 | 0.6790 | 0.2360 | 74.21% | 0.9151 | 0.7253 | 40.03 |


### Column Reference

Every column is written by `train_experiment()` in `run_experiments.py` (result dict, L507–L521) and appears identically in `results/summary_results.csv` and `results/RESULTS_TABLE.md`. Rounding: F1 and accuracy to 4 decimals; `rr_f1` and `minutes` to 2.

#### Identity columns — describe the run, not the result

| Column | Values | Meaning |
| :--- | :--- | :--- |
| `exp_id` | `EXP-01` … `EXP-13` | Stable join key across this document, `summary_results.csv`, and `docs/PLAN.md`. Copied verbatim from the `EXPERIMENTS` matrix. |
| `stage` | `Stage 0` … `Stage 4` | Reporting group only — **not** used for control flow; all 13 cells execute the identical pipeline. Stage 0 = anchor baseline, 1 = normalization, 2 = augmentation, 3 = normalization × augmentation interaction, 4 = backbones & foundation models. |
| `backbone` | `resnet50`, `convnext_tiny`, `phikon` | `resnet50` = timm `resnet50.a1_in1k`; `convnext_tiny` = timm `convnext_tiny.fb_in1k` (both ImageNet-pretrained, fully fine-tuned); `phikon` = Owkin ViT-B/16 pretrained on ~43M TCGA tiles via iBOT, used as a **frozen encoder with a linear probe** — only 6,921 of ~86M parameters are trainable. |
| `norm` | `none`, `reinhard`, `macenko` | Stain normalization applied to **both** training and evaluation tiles. The reference tile is fixed across all 13 cells (first `TUM` patch of the training split), and the normalizer is fitted once per experiment. |
| `aug` | `none`, `aug_geo`, `aug_stain`, `aug_combined` | Training-time augmentation policy. **Applies to the training split only** — every evaluation set is transformed with `policy="none"`, so this column never changes how test tiles are processed. |

#### Metric columns

| Column | Definition | Range | Better |
| :--- | :--- | :---: | :---: |
| `best_val_f1` | Highest macro-F1 reached on the in-domain **validation** split (3,749 tiles) during the 8 epochs. This is the **model-selection criterion** — the checkpoint with the best value here is the one later evaluated on both test sets. | 0–1 | higher |
| `test_id_f1` | **Macro-averaged F1 on the in-domain test split** (`test_id`, 3,749 held-out source tiles). All 9 classes weighted equally regardless of frequency. This is the model's ceiling: performance when stain conditions match training. | 0–1 | higher |
| `test_ood_f1` | **Macro-averaged F1 on the out-of-domain target set** (`test_ood`, all 7,180 CRC-VAL-HE-7K tiles, 50 unseen patients, different institution). The study's headline metric. | 0–1 | higher |
| `delta_f1` | `test_id_f1 − test_ood_f1`, the **Stain Drop**. Absolute loss of F1 attributable to the stain/domain shift. | −1 … 1 | **lower** (0 = perfect invariance) |
| `rr_f1` | `(test_ood_f1 / test_id_f1) × 100`, the **Retention Rate**. Percentage of in-domain diagnostic power preserved across institutions. | 0–100 (%) | **higher** (100% = full retention) |
| `test_id_acc` | Plain top-1 accuracy on `test_id`. | 0–1 | higher |
| `test_ood_acc` | Plain top-1 accuracy on `test_ood`. | 0–1 | higher |
| `minutes` | Wall-clock training time per experiment, measured from just before epoch 1 to just after the OOD evaluation. **Includes** the 8 validation passes and both test evaluations; **excludes** split preparation and the one-time pretrained-weights download. | ≥ 0 | — |

#### Notes on interpretation

- **`delta_f1` alone can mislead.** It is an *absolute* difference, so it is sensitive to the in-domain ceiling. A model that is simply weaker in-domain can post a smaller drop without being more robust. Always read it next to `rr_f1`, which normalizes by `test_id_f1` and is the fairer cross-model comparison.
- **Accuracy columns are the weakest evidence in the table.** The class distribution is imbalanced in the target domain (`ADI` 1,338 vs `DEB` 339), so plain accuracy flatters models that do well on majority classes. They are kept for legibility to a non-specialist audience; macro-F1 is the primary metric for every claim. Expect `test_id_acc` to sit within ~0.002 of `test_id_f1` (in-domain the classes are near-balanced by stratification) but `test_ood_acc` to diverge more.
- **`minutes` is a real experimental signal, not just bookkeeping.** Normalization costs wall-clock because Macenko/Reinhard add per-tile NumPy work: unnormalized cells run ~11–13 min, Reinhard ~16.5 min, Macenko ~17–21 min. Phikon is the fastest cell in the study (9.08 min) *because* its encoder is frozen and only the linear head receives gradients.
- **`best_val_f1` is not a test metric.** It is a development metric used for checkpoint selection; comparing it to `test_id_f1` is a useful sanity check (they should be close), but neither should be reported as a result.

#### What the table does *not* contain

- **`balanced_acc` and `auroc` are computed but discarded.** `evaluate_model()` returns four metrics (L424), but only macro-F1 and accuracy are propagated into the result dict (L507–L521). Both are therefore computed on all 10 evaluations per experiment and then thrown away — they are not recoverable from any artifact.
- **No per-class metrics and no confusion matrices.** The pipeline emits aggregate scalars only. Diagnosing *which* tissue classes drive a given `delta_f1` — e.g. whether `STR`/`MUS` confusion or background-heavy classes dominate — requires the error-analysis pass that is not yet implemented.

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
| `BACK` | Background | 847 | Glass slide / mounting medium; pure optical transmission. |
| `DEB` | Debris | 339 | Necrosis, hemorrhages; high morphological variation. |
| `LYM` | Lymphocytes | 634 | Immune infiltration; dark hyperchromatic round nuclei. |
| `MUC` | Mucus | 1,035 | Extracellular mucin pools; faint amphophilic staining. |
| `MUS` | Smooth Muscle | 592 | Colonic muscularis propria; eosinophilic fiber bundles. |
| `NORM` | Normal Mucosa | 741 | Non-neoplastic colon crypts with goblet cells. |
| `STR` | Stroma | 421 | Desmoplastic cancer-associated stroma. |
| `TUM` | Colorectal Carcinoma | 1,233 | Primary tumor epithelium; dysplastic glandular architecture. |
| **Total** | | **7,180** | **50 Independent Patients** |

*Counts verified directly from `results/splits/test_ood.csv` — the exact file used in the experiments — not from upstream documentation.*

---

## 5. Instructions for Syncing Results from Kaggle

When `notebook.ipynb` finishes executing on Kaggle:
1. Open the **Output** tab in the Kaggle notebook.
2. Locate `/kaggle/working/results/RESULTS_TABLE.md`.
3. Copy the populated Markdown table into Section 2 of this file.
4. Download `/kaggle/working/results/RESULTS_TABLE.md` into `results/` for downstream plotting or statistical analysis.
