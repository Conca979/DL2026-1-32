# Histopathology Staining Robustness Package (`src`)

A modular, production-ready, object-oriented Python package for evaluating the robustness of deep learning classifiers against clinical histopathology staining variations.

---

## 🏛️ Architecture Overview

The codebase is organized following SOLID principles, separation of concerns, and factory patterns:

```text
src/
├── __init__.py                  # Top-level API exports and versioning
├── __main__.py                  # CLI entrypoint for `python -m src`
├── cli.py                       # Argument parsing and CLI dispatcher
├── pipeline.py                  # High-level ExperimentPipeline orchestrator
│
├── config/                      # Configuration, schemas, and ablation matrix
│   ├── constants.py             # Tissue classes, ImageNet stats, file formats
│   ├── schemas.py               # Strongly-typed dataclasses (ExperimentCell, Configs)
│   └── experiments.py           # Predefined 13-cell ablation matrix & filters
│
├── normalization/               # Color & Stain Normalization
│   ├── base.py                  # BaseStainNormalizer (Abstract Base Class)
│   ├── reinhard.py              # Reinhard LAB color transfer (Ruderman space)
│   ├── macenko.py               # Macenko optical density SVD deconvolution
│   └── factory.py               # NormalizerFactory
│
├── augmentation/                # Data Augmentation Pipelines
│   ├── policies.py              # AugmentationPolicy enum
│   ├── hed_jitter.py            # Biological H&E stain concentration scaling & shifting
│   └── transforms.py            # PatchTransform (composed normalization + aug + tensor)
│
├── data/                        # Dataset & Split Management
│   ├── discovery.py             # File scanner with case-insensitive search
│   ├── dataset.py               # HistologyDataset (PyTorch Dataset)
│   ├── splitter.py              # DatasetSplitter (70/15/15 stratified + OOD cohort)
│   └── dataloader.py            # DataLoader generation factory
│
├── models/                      # Deep Learning Backbones
│   ├── phikon.py                # PhikonClassifier (TCGA SSL ViT linear probe)
│   └── factory.py               # ModelFactory (ResNet-50, ConvNeXt-Tiny, Phikon)
│
├── engine/                      # Training & Evaluation Engine
│   ├── evaluator.py             # Evaluator (Macro-F1, Balanced Acc, Acc, AUROC)
│   └── trainer.py               # ExperimentTrainer (AMP, Cosine LR, best-checkpoint tracking)
│
└── reporting/                   # Metrics and Artifact Export
    ├── metrics.py               # ExperimentResult dataclass
    └── reporter.py              # ResultsReporter (CSV & Markdown summary tables)
```

---

## 🧩 Module Details

### 1. `src.config`
- **[`constants.py`](file:///D:/LT/DL2026-1-32/src/config/constants.py)**: Defines the 9 classes (`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`), class index lookups, ImageNet mean/std, and canonical H&E stain absorption vectors.
- **[`schemas.py`](file:///D:/LT/DL2026-1-32/src/config/schemas.py)**: Strong typing for configs via dataclasses: `ExperimentCell`, `SplitConfig`, `TrainingConfig`, `PipelineConfig`.
- **[`experiments.py`](file:///D:/LT/DL2026-1-32/src/config/experiments.py)**: The 13 canonical ablation cells (EXP-01 through EXP-13) and query/filter helpers.

### 2. `src.normalization`
- **[`BaseStainNormalizer`](file:///D:/LT/DL2026-1-32/src/normalization/base.py)**: Unified interface with `fit(ref_rgb)` and `transform(img_rgb)`.
- **[`ReinhardNormalizer`](file:///D:/LT/DL2026-1-32/src/normalization/reinhard.py)**: Converts RGB to Ruderman LAB space, matches channel means and standard deviations, and converts back to RGB.
- **[`MacenkoNormalizer`](file:///D:/LT/DL2026-1-32/src/normalization/macenko.py)**: Converts to optical density (OD), estimates stain vectors via SVD projection, deconvolves concentrations, and scales to canonical reference.
- **[`NormalizerFactory`](file:///D:/LT/DL2026-1-32/src/normalization/factory.py)**: Creates and initializes normalizers dynamically based on experiment configuration.

### 3. `src.augmentation`
- **[`AugmentationPolicy`](file:///D:/LT/DL2026-1-32/src/augmentation/policies.py)**: Enum of policies (`none`, `aug_geo`, `aug_stain`, `aug_combined`).
- **[`hed_stain_jitter`](file:///D:/LT/DL2026-1-32/src/augmentation/hed_jitter.py)**: Physically grounded H&E concentration jitter ($\alpha \in [1-\sigma, 1+\sigma]$, $\beta \in [-\text{bias}, \text{bias}]$).
- **[`PatchTransform`](file:///D:/LT/DL2026-1-32/src/augmentation/transforms.py)**: Composable pipeline executing normalization $\to$ stain jitter $\to$ tensor conversion $\to$ geometric flips/rotations $\to$ ImageNet standardization.

### 4. `src.data`
- **[`find_class_images`](file:///D:/LT/DL2026-1-32/src/data/discovery.py)**: Robust image file discovery across case variations and nested folder hierarchies.
- **[`DatasetSplitter`](file:///D:/LT/DL2026-1-32/src/data/splitter.py)**: Stratified 70/15/15 source split + OOD target cohort isolation; canonical tumor reference tile extraction.
- **[`HistologyDataset`](file:///D:/LT/DL2026-1-32/src/data/dataset.py)**: PyTorch `Dataset` wrapper.
- **[`create_dataloaders`](file:///D:/LT/DL2026-1-32/src/data/dataloader.py)**: Builds pinned-memory DataLoaders for train, val_id, test_id, and test_ood.

### 5. `src.models`
- **[`PhikonClassifier`](file:///D:/LT/DL2026-1-32/src/models/phikon.py)**: Linear probing classifier leveraging the frozen 768-dim `[CLS]` token from `owkin/phikon`.
- **[`ModelFactory`](file:///D:/LT/DL2026-1-32/src/models/factory.py)**: Factory providing unified construction for `resnet50`, `convnext_tiny`, and `phikon`.

### 6. `src.engine`
- **[`Evaluator`](file:///D:/LT/DL2026-1-32/src/engine/evaluator.py)**: Mixed-precision evaluation calculating Macro-F1, Balanced Accuracy, Accuracy, and AUROC.
- **[`ExperimentTrainer`](file:///D:/LT/DL2026-1-32/src/engine/trainer.py)**: Complete training loop with AdamW, Cosine Annealing, Label Smoothing (0.1), AMP GradScaler, and best checkpoint restoration. Computes in-domain vs. OOD robustness ($\Delta$-F1, Robustness Ratio RR).

### 7. `src.reporting`
- **[`ResultsReporter`](file:///D:/LT/DL2026-1-32/src/reporting/reporter.py)**: Outputs formatted summary tables in both CSV (`summary_results.csv`) and Markdown (`RESULTS_TABLE.md`).

---

## 🚀 Usage Examples

### 1. Command Line Interface (CLI)

```bash
# Print planned execution matrix (Dry-Run)
python -m src --dry-run

# Run only specific experiments
python -m src --experiments EXP-01 EXP-07 EXP-13 --epochs 8 --batch-size 64

# Full pipeline run
python -m src --data-root data/raw/NCT-CRC-HE-100K-NONORM \
              --target-root data/raw/CRC-VAL-HE-7K \
              --out-dir results \
              --subset 25000 \
              --epochs 8
```

### 2. Python Programmatic API

```python
from src import (
    ExperimentPipeline,
    PipelineConfig,
    SplitConfig,
    TrainingConfig,
    NormalizerFactory,
)

# 1. Using high-level pipeline
config = PipelineConfig(
    split=SplitConfig(subset_size=5000),
    training=TrainingConfig(epochs=5, batch_size=32),
    experiment_ids=["EXP-01", "EXP-07"],
)
pipeline = ExperimentPipeline(config)
pipeline.run()

# 2. Or using individual components directly
normalizer = NormalizerFactory.create("macenko")
normalizer.fit(reference_tile_rgb)
normalized_tile = normalizer.transform(sample_tile_rgb)
```
