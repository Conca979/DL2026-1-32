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
   - **Pretrained vision backbones** (generic ImageNet CNNs vs. modern vision architectures vs. histopathology-specific foundation models)
   on out-of-distribution (OOD) classification robustness.

---

## 2. Dataset Decision

### Final Selection & Justification

**Selected Benchmark**: **`NCT-CRC-HE-100K-NONORM` (Source Domain)** paired with **`CRC-VAL-HE-7K` (Target / OOD Domain)**.

#### Justification:
1. **Native 224x224 Resolution**: Standard ImageNet models (ResNet, ConvNeXt) and modern computational pathology foundation models (Phikon, CTransPath, UNI) are natively pretrained on 224x224 tiles.
2. **Granular Multi-Class Evaluation**: Unlike binary tumor detection, 9-class tissue categorization tests fine-grained morphological discrimination across diverse histological entities:
   - Hematoxylin-dense nuclei (`LYM`)
   - Eosinophilic fibrillar structures (`MUS` vs. `STR`)
   - Pale amorphous material (`MUC`, `BACK`, `ADI`)
   - Malignant epithelium (`TUM`)
   This reveals whether color normalization inadvertently degrades diagnostic boundaries between specific tissue classes (e.g., confusing stroma with muscle).
3. **Pristine Domain Boundary**: The source domain (`NCT-CRC-HE-100K-NONORM`) and target domain (`CRC-VAL-HE-7K`) are completely non-overlapping cohorts from different German academic medical centers (Heidelberg/Mannheim vs. Aachen). `CRC-VAL-HE-7K` serves as a true out-of-distribution (OOD) test set.
4. **Friction-Free Direct Access**: Both sets are available and publically free to download.
5. **Single-GPU Compute Feasibility**: The dataset size (~1.5 GB total) can be processed on a single GPU workstation within hours, supporting a complete 13-cell ablation matrix without compromises.

---

## 3. Experiment Design

To dissect how preprocessing, augmentation, and model architectures influence staining robustness, we organize experiments across four targeted axes:

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
- Combine the top-performing normalization method (Macenko) with each augmentation policy (`Aug-Geo`, `Aug-Stain`, `Aug-Combined`).

### Axis E: Pretrained Vision Model Comparison
Evaluate three distinct backbone architectures under both the raw baseline and the best defense configuration:
1. **ResNet-50** (`timm: resnet50.a1_in1k`): Traditional clinical deep learning benchmark (residual convolutional network, 25.6M parameters).
2. **ConvNeXt-Tiny** (`timm: convnext_tiny.fb_in1k`): Modern convolutional architecture incorporating Vision Transformer design choices (7x7 depthwise convolutions, inverted bottleneck, LayerNorm, 28.6M parameters).
3. **Phikon** (`owkin/phikon` via Hugging Face `transformers`): Foundation model specifically pretrained on >40 million histopathology tiles from TCGA using self-supervised iBOT (Vision Transformer ViT-B/16 architecture, 86M parameters).

---

### Staged Ablation Matrix

The 13 experiments below provide direct, controlled comparisons where each variable is isolated against the baseline.

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
| **EXP-12** | Stage 4 | Phikon | Histopathology (TCGA) | None | None | Does large-scale histology SSL pretraining provide intrinsic stain invariance without defenses? | vs. EXP-01, EXP-10 |
| **EXP-13** | Stage 4 | Phikon | Histopathology (TCGA) | Macenko | Aug-Combined | Can foundation model representations be further boosted by stain normalization and augmentation? | vs. EXP-09, EXP-11, EXP-12 |

---

## 4. Evaluation Protocol

### 1. Data Split & Partitioning Strategy

To strictly avoid data leakage and prevent optimistic bias:
- **Source Domain (`NCT-CRC-HE-100K-NONORM`)**:
  - Partitioned into **Train (70%)**, **In-Domain Validation (`Val-ID`, 15%)**, and **In-Domain Test (`Test-ID`, 15%)**.
  - Partitioning is stratified across the 9 classes with a fixed random seed (`seed = 42`).
  - **Val-ID** is used exclusively for learning rate scheduling, early stopping, and selecting the optimal checkpoint (`best_checkpoint.pt`).
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
3. **Secondary Metric 2**: **Overall Top-1 Accuracy (`Acc`)**
   - Standard benchmark metric for comparison with existing literature.
4. **Secondary Metric 3**: **Macro One-vs-Rest AUROC (`Macro-AUROC`)**
   - Evaluates the ranking quality of predicted softmax probabilities across classes.
5. **Per-Class Analysis**:
   - Complete 9x9 confusion matrices (normalized by true class rows) saved as CSV and heatmap plots to diagnose specific tissue confusions (e.g., `STR` vs. `MUS`).

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

## 5. Tech Stack Decision

### Core Libraries & Environment

| Component | Library & Version | Rationale & Selection Justification |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.10 or 3.11 | Modern typing, stability, and maximum ecosystem compatibility. |
| **Deep Learning Framework** | `torch >= 2.2.0`, `torchvision >= 0.17.0` | Industry standard; native Automatic Mixed Precision (AMP `fp16`) for fast training and low VRAM footprint on single GPU. |
| **Standard Vision Backbones** | `timm >= 0.9.16` | Clean, battle-tested implementations of `resnet50.a1_in1k` and `convnext_tiny.fb_in1k` with verified ImageNet weights and unified classifier replacement. |
| **Pathology Foundation Model** | `transformers >= 4.38.0` | Official Hugging Face loader for `owkin/phikon` (`AutoImageProcessor`, `AutoModel`). Ungated, open-weights. |
| **Stain Normalization** | `torchstain >= 0.2.2`, `scikit-image >= 0.22.0`, `opencv-python-headless >= 4.9.0` | PyTorch/GPU implementations of Macenko and Reinhard algorithms; OpenCV for color space conversions and optical density masking. |
| **Data Augmentation** | `albumentations >= 1.4.0` | High-throughput C++ backend for spatial augmentations; custom PyTorch transform for HED stain deconvolution jitter. |
| **Metrics & Evaluation** | `scikit-learn >= 1.4.0` | Robust, standardized implementations of macro F1, balanced accuracy, and OvR AUROC. |
| **Data & Tabulation** | `pandas >= 2.2.0`, `numpy >= 1.26.0` | Stratified dataset splitting and structured results tabulation. |

### Compute Constraints & Optimization (1 GPU Feasibility)
- **Automatic Mixed Precision (AMP)**: `torch.cuda.amp.autocast()` enabled across all runs, cutting VRAM by ~50% and accelerating Tensor Core operations.
- **Batch Size & Gradient Accumulation**:
  - ResNet-50 / ConvNeXt-Tiny: Batch size 64.
  - Phikon (ViT-B/16): Batch size 32 with 2 gradient accumulation steps (effective batch size 64) to prevent Out-Of-Memory (OOM) on 8GB-12GB GPUs.
- **Pre-computed Normalization Cache**: To eliminate CPU bottleneck during training, an optional offline preprocessing step caches Reinhard- and Macenko-normalized patches to disk, increasing training throughput by 5-10x.

---

## 6. Repository Structure

The project is a standalone repository: everything below lives at the repository
root, and the codebase zip handed to Kaggle is built from this root (verified by
`scripts/make_zips.py`). `docs/` was added after the original plan and `6b` below
lists it.

```text
<repo root>/
├── README.md
├── pyproject.toml            # package + extras (train / normalize / dev / all)
├── uv.lock                   # pinned local environment
├── .python-version           # 3.10
├── docs/                     # all documentation (index: docs/README.md)
├── configs/
│   ├── base_config.yaml
│   ├── experiments_registry.json
│   └── experiments/
│       ├── exp01_baseline_resnet50.yaml
│       ├── exp02_norm_reinhard_resnet50.yaml
│       ├── exp03_norm_macenko_resnet50.yaml
│       ├── exp04_aug_geo_resnet50.yaml
│       ├── exp05_aug_stain_resnet50.yaml
│       ├── exp06_aug_combined_resnet50.yaml
│       ├── exp07_interaction_macenko_geo_resnet50.yaml
│       ├── exp08_interaction_macenko_stain_resnet50.yaml
│       ├── exp09_interaction_macenko_combined_resnet50.yaml
│       ├── exp10_backbone_convnext_raw.yaml
│       ├── exp11_backbone_convnext_best.yaml
│       ├── exp12_backbone_phikon_raw.yaml
│       └── exp13_backbone_phikon_best.yaml
├── data/
│   ├── raw/
│   │   ├── NCT-CRC-HE-100K-NONORM/
│   │   └── CRC-VAL-HE-7K/
│   ├── processed/
│   │   ├── splits/
│   │   │   ├── train.csv
│   │   │   ├── val_id.csv
│   │   │   ├── test_id.csv
│   │   │   └── test_ood.csv
│   │   └── templates/
│   │       └── reference_stain.png
│   └── scripts/
│       ├── download_data.sh
│       ├── download_data.py
│       └── prepare_splits.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── datamodule.py
│   ├── normalization/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── reinhard.py
│   │   ├── macenko.py
│   │   └── normalizer_factory.py
│   ├── augmentation/
│   │   ├── __init__.py
│   │   ├── geometric.py
│   │   ├── stain_jitter.py
│   │   └── policy_factory.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py
│   │   └── classifier.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── evaluator.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── seed.py
│       ├── metrics.py
│       └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_all_ablations.py
│   └── preprocess_normalize.py
│
└── kaggle/                         # notebooks for training in kaggle notebook
    ├── notebook_01_run_all_ablations.ipynb
    └── notebook_02_results_tally.ipynb
```

---

## 7. Risks & Mitigations

### Risk 1: Macenko Deconvolution Instability on Background and Adipose Patches
- **Description**: Patches containing mostly transparent slide glass (`BACK`), large lipid droplets (`ADI`), or necrosis have near-zero optical density. Singular Value Decomposition (SVD) on low-density optical matrices can fail to converge or yield near-zero stain vectors, causing division by zero or NaN artifacts.
- **Mitigation**: Implement a luminance tissue mask filter:
  - If a patch contains >85% background pixels (luminance > 220 in all RGB channels), bypass SVD deconvolution and return the raw tile.
  - Wrap Macenko deconvolution in a `try/except` block that automatically catches non-convergence and falls back to Reinhard normalization or raw RGB, incrementing a logged fallback counter.

### Risk 2: Online Normalization DataLoader Latency
- **Description**: Performing SVD-based optical density deconvolution on-the-fly for every training image in PyTorch `DataLoader` workers creates a major CPU bottleneck, starving the GPU and extending training time significantly.
- **Mitigation**: Provide an offline pre-normalization utility (`scripts/preprocess_normalize.py`). Pre-normalizing the training and validation splits once into cached disk directories (`data/processed/cache_macenko/` and `data/processed/cache_reinhard/`) allows high-throughput native tensor loading at full GPU saturation.

### Risk 3: Target Domain Data Leakage / Premature Optimization
- **Description**: If early stopping or hyperparameter choices are inadvertently guided by the out-of-domain test set (`CRC-VAL-HE-7K`), reported robustness numbers become invalid.
- **Mitigation**: Enforce an architectural firewall in code: the `Trainer` class only has access to `train_loader` and `val_id_loader`. Model checkpoints are saved exclusively on peak `val_id_macro_f1`. `CRC-VAL-HE-7K` is loaded only inside `scripts/evaluate.py` during post-training audit.

### Risk 4: Target Set Class Imbalance Masking Clinical Failures
- **Description**: In `CRC-VAL-HE-7K`, the majority classes (`ADI`, `TUM`) comprise >35% of all samples, while `DEB` and `STR` are scarce. A model could achieve high nominal accuracy while failing on difficult diagnostic classes.
- **Mitigation**: Mandate `Macro-F1` and `Balanced Accuracy` as the primary evaluation criteria in all summary tables. Generate and save normalized 9x9 confusion matrices for every test run to visually track class-specific confusions.

### Risk 5: Foundation Model (Phikon) VRAM Limits on Consumer GPUs
- **Description**: Phikon is a 86M-parameter Vision Transformer. Backpropagation with full gradient tracking may trigger CUDA Out-of-Memory (OOM) errors on 8GB-12GB GPUs.
- **Mitigation**:
  1. Default to PyTorch AMP (`fp16`).
  2. Reduce per-device batch size to 32 with 2 gradient accumulation steps.
  3. Support linear probing / head fine-tuning mode (frozen ViT backbone, training only the classification head) via a simple config toggle (`freeze_backbone: true`).

---