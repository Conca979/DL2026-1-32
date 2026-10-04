# DATA.md — Datasets, Splits, and Preprocessing

This document describes every dataset used in this project, how the data is split, the preprocessing applied, and the exact commands needed to reproduce the data used in the reported experiments.

It supersedes the former `docs/dataset_card.md`.

---

## 1. Official Dataset Source

| Field | Value |
| :--- | :--- |
| **Dataset title** | *100,000 histological images of human colorectal cancer and healthy tissue* |
| **Official URL** | https://doi.org/10.5281/zenodo.1214456 |
| **Zenodo record** | https://zenodo.org/records/1214456 |
| **Version** | **v0.1** (published 2018-04-07; record last modified 2020-01-24) |
| **Record metadata verified** | 2026-10-04 |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Creators** | Jakob Nikolas Kather, Niels Halama (NCT Heidelberg); Alexander Marx (University Medical Center Mannheim) |
| **Primary reference** | Kather et al., *Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study*, PLOS Medicine, 2019 |

### Files downloaded from the record

| File | Size | MD5 checksum |
| :--- | ---: | :--- |
| `NCT-CRC-HE-100K-NONORM.zip` | 11.7 GB | `035777cf327776a71a05c95da6d6325f` |
| `CRC-VAL-HE-7K.zip` | 800.3 MB | `2fd1651b4f94ebd818ebf90ad2b6ce06` |

> The record also hosts `NCT-CRC-HE-100K.zip` (11.7 GB), the **Macenko-normalized** variant. This project uses the **`NONORM`** variant exclusively — retaining raw, unnormalized stain variation is the entire point of the study, and using the pre-normalized variant would confound every normalization experiment.

### Provenance notes

- `NCT-CRC-HE-100K-NONORM` — 100,000 non-overlapping patches extracted from **N = 86** FFPE slides (NCT Biobank and UMM pathology archive), covering primary colorectal cancer and CRC liver metastases; normal classes supplemented from non-tumorous gastrectomy specimens. No color normalization applied.
- `CRC-VAL-HE-7K` — 7,180 patches from **N = 50** colorectal adenocarcinoma patients with **no patient overlap** with the 100K set. This patient-level independence is what makes it a valid out-of-domain test set.
- The record notes that tile selection was stochastic, so the `NONORM` patches are not pixel-identical to the normalized set, and that classes are only roughly balanced — hence this project's use of macro-averaged F1 rather than accuracy.

---

## 2. The Nine Histological Classes

Both datasets use the same nine mutually exclusive classes, stored in subfolders named by the official acronyms. **The class index below is the training label** (`label_idx` in the split CSVs) and must not be reordered.

| Index | Code | Class | Morphology | Target count (`CRC-VAL-HE-7K`) |
| :---: | :--- | :--- | :--- | ---: |
| 0 | `ADI` | Adipose | Large empty lipid droplets, thin peripheral cytoplasm, eccentric nuclei | 1,338 |
| 1 | `BACK` | Background | Transparent glass-slide areas, no tissue | 847 |
| 2 | `DEB` | Debris | Necrotic material, hemorrhage, cell fragments, cautery artifact | 339 |
| 3 | `LYM` | Lymphocytes | Dense aggregates of small, dark, round hyperchromatic nuclei | 634 |
| 4 | `MUC` | Mucus | Pools of amorphous, pale basophilic extracellular mucin | 1,035 |
| 5 | `MUS` | Smooth muscle | Elongated eosinophilic spindle cells with cigar-shaped nuclei | 592 |
| 6 | `NORM` | Normal mucosa | Healthy non-dysplastic crypts, orderly columnar epithelium | 741 |
| 7 | `STR` | Stroma | Desmoplastic fibrous stroma, collagen matrix, vascular elements | 421 |
| 8 | `TUM` | Tumor epithelium | Crowded malignant glands with atypia, nuclear enlargement, hyperchromasia | 1,233 |
| | | **Total** | | **7,180** |

*Target counts above were counted directly from `results/splits/test_ood.csv` — they are exact, not estimates.*

Per-class counts for the 100,000-patch source set are approximately ADI ≈ 10.4k, BACK ≈ 10.6k, DEB ≈ 11.5k, LYM ≈ 11.6k, MUC ≈ 8.9k, MUS ≈ 13.5k, NORM ≈ 8.8k, STR ≈ 10.4k, TUM ≈ 14.3k. **These are upstream approximations and have not been re-verified in this repository.**

---

## 3. Data Split

The design separates an in-domain source cohort from a strictly held-out external cohort.

```
+----------------------------------------------------------------------------------+
| SOURCE DOMAIN: NCT-CRC-HE-100K-NONORM   (Heidelberg / Mannheim cohort)           |
|                                                                                  |
|   [ Train  70% ]          [ Val-ID  15% ]         [ Test-ID  15% ]               |
|   weight updates          checkpoint selection    in-domain benchmark            |
+----------------------------------------------------------------------------------+
                                    |
                                    |  Delta_F1 = F1_ID - F1_OOD
                                    v
+----------------------------------------------------------------------------------+
| TARGET DOMAIN: CRC-VAL-HE-7K   (RWTH Aachen cohort, 50 unseen patients)          |
|                                                                                  |
|   [ Test-OOD  100% of 7,180 patches ]   strictly held out                        |
+----------------------------------------------------------------------------------+
```

### 3.1 Sizes actually produced

With the default `--subset 25000` budget:

| Split | Patches | Source |
| :--- | ---: | :--- |
| `train` | **17,495** | `results/splits/train.csv` |
| `val_id` | **3,749** | `results/splits/val_id.csv` |
| `test_id` | **3,749** | `results/splits/test_id.csv` |
| `test_ood` | **7,180** | `results/splits/test_ood.csv` (entire target domain) |

In-domain class balance is even by construction — `test_id.csv` holds 416–417 patches per class.

> **How 25,000 becomes 24,993.** The subset budget is divided per class using floor division: `25000 // 9 = 2,777` patches per class → **24,993** patches total. The three source splits are then produced by two stratified `train_test_split` calls (30% off the top, then a 50/50 halving of the remainder), giving 17,495 / 3,749 / 3,749. This is why the numbers differ slightly from the nominal 17,500 / 3,750 / 3,750 quoted in older project notes.

### 3.2 Partitioning rules

1. **Source domain** — stratified across all 9 classes with a fixed `seed = 42`. The split is deterministic: the same input file set always yields the same three CSVs.
2. **No overlap** — `train`, `val_id`, and `test_id` are sampled disjointly; no patch appears in more than one split.
3. **Compute budget** — the standardized 25,000-patch subset is the default so that all 13 experiments complete in a single Kaggle session. Pass `--subset 0` to train on all 100,000 patches.
4. **Strict domain firewall** — `CRC-VAL-HE-7K` is evaluated **once per experiment, post-training**, as `test_ood`. It is never used for training, normalizer fitting, hyperparameter tuning, early stopping, or checkpoint selection. This is enforced structurally: the target directory is read exactly once, at split-preparation time, and no other code path references it.

### 3.3 Canonical stain reference tile

Exactly **one** reference tile is selected for color normalization: the **first `TUM` patch of the training split**, saved as `results/splits/reference_stain.png`. Three constraints are deliberate:

1. It is a **densely stained tumor** tile, which gives the stain estimators strong hematoxylin/eosin signal;
2. It comes **only from the training split**, so it carries no evaluation information;
3. It is **deterministic**, so all 13 experiments normalize toward the identical reference and remain comparable.

### 3.4 Split file schema

All four CSVs share the same columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_path` | string | Absolute path to the patch image |
| `class_name` | string | One of the nine class codes |
| `label_idx` | int | Class index 0–8, matching §2 |
| `domain` | string | `source` for train/val_id/test_id, `target` for test_ood |

---

## 4. Preprocessing

### 4.1 Data preparation (applied once, before training)

1. **Locate** the nine class folders under each dataset root, tolerating case differences and one level of nesting.
2. **Subsample** the source domain to a per-class cap of `subset_size // 9` patches (default 2,777 → 24,993 total).
3. **Split** the source domain 70/15/15, stratified by class, `seed = 42`.
4. **Write** the four split CSVs and the reference tile to `results/splits/`.
5. **No pixel-level preprocessing is applied to the stored data.** Patches are used at their native 224 × 224 px, 0.5 µm/px. Images are converted to RGB and resized by nothing.

### 4.2 Experiment-time transforms (applied per patch, in this order)

These are part of the model input pipeline, not the stored data. They are applied identically at training and evaluation time unless noted.

| Step | Transform | Applies to |
| :--- | :--- | :--- |
| 1 | **Stain normalization** (optional): `reinhard` — statistical mean/std transfer in log-αβ space; `macenko` — optical-density H&E deconvolution with reference stain-vector and 99th-percentile concentration matching | All splits, when the experiment's `norm` is not `none` |
| 2 | **HED stain jitter** (train only, p = 0.8): independent multiplicative (α ∈ [0.8, 1.2]) and additive (β ∈ [−0.05, 0.05]) perturbation of hematoxylin/eosin concentrations | Training split only |
| 3 | **To tensor**: `uint8 [0,255] → float32 [0,1]`, HWC → CHW | All splits |
| 4 | **Geometric augmentation** (train only): horizontal flip p = 0.5, vertical flip p = 0.5, rotation uniformly from {0°, 90°, 180°, 270°} | Training split only |
| 5 | **ImageNet normalization**: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` | All splits |

> Evaluation is **always** performed with the experiment's normalization applied but **no augmentation** — augmentation is a training-time intervention only, so every model is compared on identically-transformed test tiles.

### 4.3 Expected folder layout

```text
data/
└── raw/
    ├── NCT-CRC-HE-100K-NONORM/
    │   ├── ADI/
    │   ├── BACK/
    │   ├── DEB/
    │   ├── LYM/
    │   ├── MUC/
    │   ├── MUS/
    │   ├── NORM/
    │   ├── STR/
    │   └── TUM/
    └── CRC-VAL-HE-7K/
        ├── ADI/
        ├── BACK/
        ├── DEB/
        ├── LYM/
        ├── MUC/
        ├── MUS/
        ├── NORM/
        ├── STR/
        └── TUM/

results/                      # produced by the pipeline, committed to this repo
└── splits/
    ├── train.csv
    ├── val_id.csv
    ├── test_id.csv
    ├── test_ood.csv
    └── reference_stain.png
```

Class folder names are matched **case-insensitively**, and one level of extra nesting (e.g. `<root>/<root>/ADI/`) is tolerated, so archive-extraction differences do not break preparation.

---

## 5. Reproducing the Data

### 5.1 Download

Download the two archives from the Zenodo record in §1 and extract them so the tree matches §4.3.

### 5.2 Generate the splits

The split CSVs are produced by `run_experiments.py` as its first step, before any training begins. The settings that determine the data are `--data-root`, `--target-root`, `--out-dir`, and `--subset`:

```bash
python run_experiments.py \
  --data-root  data/raw/NCT-CRC-HE-100K-NONORM \
  --target-root data/raw/CRC-VAL-HE-7K \
  --out-dir    results \
  --subset     25000
```

Running this command writes `results/splits/{train,val_id,test_id,test_ood}.csv` and `results/splits/reference_stain.png` with the exact sizes in §3.1, then proceeds to train.

> **Note:** there is currently no prepare-only flag — the splits are written at the start of a normal run. To regenerate the CSVs without training, interrupt the run after the `[splits] ...` line is printed, or call `prepare_dataset_splits()` from `run_experiments.py` directly. A dedicated `--prepare-only` flag is planned.

The split is fully determined by (`--subset`, `seed = 42`, and the input file set). Given identical inputs, the four CSVs and the reference tile are reproducible byte-for-byte.

---

## 6. Citation

If you use these datasets, cite the original record:

```bibtex
@dataset{kather2018colorectal,
  author    = {Kather, Jakob Nikolas and Halama, Niels and Marx, Alexander},
  title     = {100,000 histological images of human colorectal cancer and healthy tissue},
  year      = {2018},
  publisher = {Zenodo},
  version   = {v0.1},
  doi       = {10.5281/zenodo.1214456},
  url       = {https://doi.org/10.5281/zenodo.1214456}
}
```

Primary research paper describing the source cohort:

```bibtex
@article{kather2019predicting,
  author  = {Kather, Jakob Nikolas and Krisam, Johannes and Charoentong, Pornpimol and others},
  title   = {Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study},
  journal = {PLOS Medicine},
  year    = {2019},
  volume  = {16},
  number  = {1},
  pages   = {e1002730},
  doi     = {10.1371/journal.pmed.1002730}
}
```

---

## 7. License and Ethical Notes

- Both datasets are released under **CC BY 4.0**, which permits redistribution and reuse with attribution. Attribution is given in §6.
- Source tissue was collected under NCT Heidelberg ethics approval S-207/2005 (tissue bank decisions 2152, 2154) and UMM Ethics Board II approval 2017-806R-MA, with informed consent waived for retrospective anonymized archival analysis.
- This repository redistributes **only derived artifacts** — the split CSV files and the single reference tile — not the source imagery. The 224 × 224 patches remain available solely from the official Zenodo record.
