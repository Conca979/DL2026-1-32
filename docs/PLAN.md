# Project Plan: Robust Histopathology Image Classification under Staining Variations

## 1. Overview

### Problem Statement
Digital histopathology utilizes Hematoxylin and Eosin (H&E) staining to reveal cellular and tissue architecture for diagnostic and prognostic assessment. In clinical practice, whole-slide images (WSIs) exhibit profound appearance and color discrepancies across different pathology laboratories. These variations stem from differences in:
- Tissue processing and section thickness (microtome precision)
- Staining protocols, reagent batches, dye concentrations, and incubation times
- Whole-slide imaging (WSI) scanner hardware (charge-coupled device sensors, optical lenses, illumination spectra, and internal color calibration profiles)

Deep neural networks trained on histology data from a single hospital or scanner frequently suffer from shortcut learning: they latch onto site-specific staining signatures (color histograms and hue biases) rather than invariant biological morphology (nuclear pleomorphism, chromatin distribution, glandular architecture). Consequently, when deployed on slides from unseen medical centers, model accuracy often drops precipitously.

### Project Goals
1. **Classify histopathology images** robustly under conditions where staining and image appearance vary across clinical centers and scanner platforms.
2. **Investigate the individual and synergistic effects** of:
   - **Color normalization algorithms** (algorithmic color harmonization to a canonical reference)
   - **Data augmentation policies** (spatial geometric vs. biologically-grounded stain color jitter)
   - **Pretrained vision backbones** (classical ResNet-50 vs. modern ConvNeXt-Tiny architectures)
   on out-of-distribution (OOD) classification robustness.

---

## 2. Dataset Decision
### Final Selection & Justification

**Selected Benchmark**: **`NCT-CRC-HE-100K-NONORM` (Source Domain)** paired with **`CRC-VAL-HE-7K` (Target / OOD Domain)**.

#### Justification:
1. **Native 224x224 Resolution**: Standard ImageNet models (ResNet, ConvNeXt) are natively pretrained on 224x224 tiles. Using native 224x224 patches eliminates interpolation artifacts and blurred nuclear boundaries caused by upscaling 96x96 images.
2. **Granular Multi-Class Evaluation**: Unlike binary tumor detection, 9-class tissue categorization tests fine-grained morphological discrimination across diverse histological entities:
   - Hematoxylin-dense nuclei (`LYM`)
   - Eosinophilic fibrillar structures (`MUS` vs. `STR`)
   - Pale amorphous material (`MUC`, `BACK`, `ADI`)
   - Malignant epithelium (`TUM`)
   This reveals whether color normalization inadvertently degrades diagnostic boundaries between specific tissue classes (e.g., confusing stroma with muscle).
3. **Pristine Domain Boundary**: The source domain (`NCT-CRC-HE-100K-NONORM`) and target domain (`CRC-VAL-HE-7K`) are completely non-overlapping cohorts from different German academic medical centers (Heidelberg/Mannheim vs. Aachen). `CRC-VAL-HE-7K` serves as a true out-of-distribution (OOD) test set.

---

## 3. Experiment Design

To dissect how preprocessing, augmentation, and model architectures influence staining robustness, we organize experiments across five targeted axes:

### Axis A: Anchor Baseline
- Generic ImageNet pretrained backbone (`ResNet-50`).
- No color normalization (raw RGB).
- No data augmentation (standard center evaluation / deterministic normalization).

### Axis B: Color Normalization Ablation
Holding the backbone (`ResNet-50`) and augmentation (`None`) fixed, compare color harmonization strategies:
1. **None**: Raw RGB images with unnormalized hospital stain variation.
2. **Reinhard Normalization**: Statistical color transfer in CIELAB space matching mean and standard deviation of L*, a*, b* channels to a canonical reference patch.
3. **Macenko Normalization**: Optical Density (OD) deconvolution using the Beer-Lambert law to extract individual Hematoxylin and Eosin stain vectors and normalize stain concentrations against a canonical reference patch.

### Axis C: Augmentation Policy Ablation
Holding the backbone (`ResNet-50`) and normalization (`None`) fixed, compare invariance mechanisms:
1. **None**: Deterministic input.
2. **Geometric-Only (`Aug-Geo`)**: Invariance to spatial orientation (Random Horizontal/Vertical Flips, Random 90-degree rotations).
3. **Stain/Color-Focused (`Aug-Stain`)**: Biologically-grounded Hematoxylin-Eosin-DAB (HED) color jitter that deconvolves the image into H and E stain concentration maps and applies independent stochastic scaling and shifts, plus subtle HSV perturbations.
4. **Combined (`Aug-Combined`)**: Spatial geometric transformations combined with HED stain jitter.

### Axis D: Normalization x Augmentation Interaction
Evaluate whether explicit color normalization and stochastic stain augmentation are complementary or redundant:
- Combine the *expected* strongest normalization method (Macenko) with each augmentation policy (`Aug-Geo`, `Aug-Stain`, `Aug-Combined`). Stage 3 was designed on the assumption that Macenko would outperform Reinhard; the completed run contradicted that assumption — which is itself a reported finding (see `docs/RESULTS.md`).

---

### Staged Ablation Matrix

The 13 experiments below provide direct, controlled comparisons where each variable is isolated against the baseline. Stage 4 extends beyond the original 11-cell design with a histology-specific foundation model (Phikon), contrasted against the ImageNet-pretrained backbones.

| Exp ID | Stage | Backbone | Weights Source | Normalization | Augmentation Policy | Primary Research Question Addressed | Comparator / Baseline |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **EXP-01** | Stage 0 | ResNet-50 | ImageNet-1k | None | None | **Primary Baseline**: Out-of-the-box in-domain and OOD performance under raw staining. | Anchor Baseline |
| **EXP-02** | Stage 1 | ResNet-50 | ImageNet-1k | Reinhard | None | Does statistical Lab color transfer improve out-of-domain generalization? | vs. EXP-01 |
| **EXP-03** | Stage 1 | ResNet-50 | ImageNet-1k | Macenko | None | Does optical density H&E deconvolution outperform statistical color transfer? | vs. EXP-01, EXP-02 |
| **EXP-04** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Geo | What portion of robustness is attributable solely to spatial invariance? | vs. EXP-01 |
| **EXP-05** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Stain | Does synthetic HED stain jitter match or exceed explicit stain normalization? | vs. EXP-01, EXP-03 |
| **EXP-06** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Combined | Are spatial and stain augmentations additive when used without normalization? | vs. EXP-04, EXP-05 |
| **EXP-07** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Geo | Does combining Macenko normalization with spatial augmentation yield gains? | vs. EXP-03, EXP-04 |
| **EXP-08** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Stain | Does stain jitter add value to normalized tiles, or do they conflict? | vs. EXP-03, EXP-05 |
| **EXP-09** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Combined | **Defended ResNet Baseline**: Peak performance of ResNet under full defense. | vs. EXP-01, EXP-06 |
| **EXP-10** | Stage 4 | ConvNeXt-T | ImageNet-1k | None | None | Is modern ConvNet architecture intrinsically more robust to stain shift than ResNet? | vs. EXP-01 |
| **EXP-11** | Stage 4 | ConvNeXt-T | ImageNet-1k | Macenko | Aug-Combined | Does ConvNeXt-Tiny benefit from the combined normalization/augmentation policy? | vs. EXP-09, EXP-10 |
| **EXP-12** | Stage 4 | Phikon (ViT-B/16) | TCGA, iBOT SSL | None | None | Does large-scale histology self-supervised pretraining provide intrinsic stain invariance? | vs. EXP-01, EXP-10 |
| **EXP-13** | Stage 4 | Phikon (ViT-B/16) | TCGA, iBOT SSL | Macenko | Aug-Combined | Can a frozen foundation-model encoder benefit from the combined defense policy? | vs. EXP-12, EXP-11 |

---

## 4. Evaluation Protocol

### 1. Data Split & Partitioning Strategy

To strictly avoid data leakage and prevent optimistic bias:
- **Source Domain (`NCT-CRC-HE-100K-NONORM`)**:
  - Partitioned into **Train (70%)**, **In-Domain Validation (`Val-ID`, 15%)**, and **In-Domain Test (`Test-ID`, 15%)**.
  - Partitioning is stratified across the 9 classes with a fixed random seed (`seed = 42`).
  - **Val-ID** is used exclusively for checkpoint selection: the model is scored on Val-ID after every epoch, and the weights achieving the best macro-F1 are retained for the final evaluation.
  - **Test-ID** is evaluated once at the end of training to measure in-domain performance.
- **Target Domain (`CRC-VAL-HE-7K`)**:
  - 100% of the 7,180 patches from 50 independent Aachen patients form the **Out-of-Domain Test (`Test-OOD`)**.
  - **Strict Firewall**: `CRC-VAL-HE-7K` is never used for training, normalization reference selection, hyperparameter tuning, or early stopping.

### 2. Evaluation Metrics

Because medical test sets often exhibit natural class imbalances (in `CRC-VAL-HE-7K`, `ADI` has 1,338 samples while `DEB` has only 339), raw classification accuracy is biased toward majority classes.

1. **Primary Metric**: **Macro-averaged F1 Score (`Macro-F1`)**
   - Calculates the harmonic mean of precision and recall for each of the 9 classes independently, then averages across classes.
   - Treats all tissue types equally, directly penalizing models that misclassify clinically critical minority classes.
2. **Secondary Metric 1**: **Balanced Accuracy (`Bal-Acc`)**
   - Arithmetic mean of recall across all 9 classes (equivalent to macro-averaged sensitivity).
   - *Reporting status: computed on every evaluation, but never propagated to the results table.*
3. **Secondary Metric 2**: **Overall Top-1 Accuracy (`Acc`)**
   - Standard benchmark metric for comparison with existing literature. Reported for both test splits.
4. **Secondary Metric 3**: **Macro One-vs-Rest AUROC (`Macro-AUROC`)**
   - Evaluates the ranking quality of predicted softmax probabilities across classes.
   - *Reporting status: computed on every evaluation, but never propagated to the results table.*
5. **Per-Class Analysis** — **not yet implemented**:
   - 9x9 confusion matrices (normalized by true class rows) and per-class precision/recall would diagnose specific tissue confusions (e.g. `STR` vs. `MUS`).
   - The shipped pipeline emits aggregate scalars only; producing these artifacts requires the error-analysis pass described in §5.

### 3. Robustness Quantification

Robustness to staining variations is explicitly quantified via two primary mathematical metrics:

```text
1. Absolute Performance Drop:
   Delta_F1 = F1_ID - F1_OOD

2. Relative Retention Rate (Robustness Ratio):
   RR_F1 = (F1_OOD / F1_ID) * 100%
```

- **`Delta_F1`**: Lower is better. A model with `Delta_F1 = 0` demonstrates perfect stain invariance across institutions.
- **`RR_F1`**: Higher is better. Quantifies the percentage of in-domain diagnostic power preserved when encountering external laboratory stains.

---

## 5. Repository Structure

The delivered implementation is deliberately compact: data preparation, stain
normalization, augmentation, training, and evaluation all live in a **single
standalone script**, `run_experiments.py`. The modular `src/histo_robust/` package
layout and the `configs/*.yaml` experiment registry described in earlier drafts of
this plan were **never built**; this section documents what actually ships.

```text
<repo root>/
├── README.md                    # overview, 13-cell matrix, how to run
├── DATA.md                      # dataset source, version, split protocol, preprocessing
├── run_experiments.py           # the complete pipeline (prep → train → evaluate)
├── notebook.ipynb               # Kaggle execution notebook
│
├── results/                     # committed run artifacts
│   ├── summary_results.csv      # 13-cell metrics table (machine-readable)
│   ├── RESULTS_TABLE.md         # same table in Markdown
│   └── splits/
│       ├── train.csv            # 17,495 rows
│       ├── val_id.csv           #  3,749 rows
│       ├── test_id.csv          #  3,749 rows
│       ├── test_ood.csv         #  7,180 rows
│       └── reference_stain.png  # canonical stain reference tile
│
├── docs/
│   ├── README.md                # documentation index
│   ├── PLAN.md                  # this document
│   ├── RESULTS.md               # results table + per-column reference
│   ├── CODE_WALKTHROUGH.md      # line-by-line explanation of the pipeline
│   ├── dataset_card.md          # 9 classes, split rules, domain firewall
│   ├── kaggle_guide.md          # 1-click execution guide
│   └── exam_requirement.md      # course rubric
│
└── scripts/
    └── make_zips.py             # packages run_experiments.py for the Kaggle dataset
