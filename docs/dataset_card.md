# Dataset Card: Colorectal Cancer Tissue Classification (NCT-CRC-HE-100K-NONORM & CRC-VAL-HE-7K)

> `docs/` index: [`README.md`](README.md) · plan: [`PLAN.md`](PLAN.md) · procedure:
> [`kaggle_guide.md`](kaggle_guide.md) · run register: [`RUN_LOG.md`](RUN_LOG.md) ·
> results: [`RESULTS.md`](RESULTS.md) · reference: [`APPENDICES.md`](APPENDICES.md)

## 1. Dataset Overview

This project investigates model robustness to histopathological staining variations using two paired, publicly available benchmark datasets of Hematoxylin and Eosin (H&E) stained colorectal tissue patches:

- **Source Domain (In-Domain Training & Evaluation)**: `NCT-CRC-HE-100K-NONORM`
- **Target Domain (Out-of-Domain Generalization)**: `CRC-VAL-HE-7K`

Both datasets share identical image dimensions, spatial resolution, and class definitions, but originate from distinct patient cohorts, pathology laboratories, tissue preparation protocols, and digital slide scanners. Critically, `NCT-CRC-HE-100K-NONORM` preserves raw, unnormalized stain variations (unlike the Macenko-normalized standard NCT-CRC-100K set), providing a realistic testbed for evaluating color normalization and stain augmentation.

| Dataset Attribute | Source Domain (`NCT-CRC-HE-100K-NONORM`) | Target Domain (`CRC-VAL-HE-7K`) |
| :--- | :--- | :--- |
| **Total Patches** | 100,000 non-overlapping patches | 7,180 non-overlapping patches |
| **Patch Resolution** | 224 x 224 pixels | 224 x 224 pixels |
| **Microns Per Pixel (MPP)** | 0.50 µm/px (approx. 20x magnification) | 0.50 µm/px (approx. 20x magnification) |
| **Staining Protocol** | Routine clinical H&E (raw, unnormalized) | Routine clinical H&E (external laboratory protocol) |
| **Download link** | [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)![NCT-100K](../playground/images/NCT-100k.png) | [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)![CRC-7k](../playground/images/CRC-7k.png) |

---

## 2. Exact Class Definitions

Both datasets categorize tissue patches into nine mutually exclusive histological classes. Patches are stored in subfolders named after the official class acronyms.

| Class Index | Class Code | Tissue Class Name | Description and Morphological Features | Source Count (`100K-NONORM`) | Target Count (`VAL-7K`) |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **0** | `ADI` | Adipose | Adipose tissue (fat cells); large empty intracellular lipid droplets with thin peripheral cytoplasm and eccentric nuclei. | ~10,407 | 1,338 |
| **1** | `BACK` | Background | Transparent background; glass slide areas with no tissue or slide edges. | ~10,566 | 847 |
| **2** | `DEB` | Debris | Necrotic debris, hemorrhage, cell fragments, mucus debris, and surgical cautery artifacts. | ~11,512 | 339 |
| **3** | `LYM` | Lymphocytes | Dense aggregates of immune cells (lymphocytes, plasma cells); small, dark, round, hyperchromatic hematoxylin-rich nuclei with minimal cytoplasm. | ~11,557 | 634 |
| **4** | `MUC` | Mucus | Pools of extracellular mucin; amorphous, pale basophilic/bluish-gray material with few scattered cells. | ~8,896 | 1,035 |
| **5** | `MUS` | Smooth Muscle | Normal smooth muscle bundles (muscularis mucosae / muscularis propria); elongated, eosinophilic spindle-shaped cells with cigar-shaped nuclei. | ~13,536 | 592 |
| **6** | `NORM` | Normal Mucosa | Healthy, non-dysplastic colon mucosal glands / crypts with orderly columnar epithelium and goblet cells. | ~8,763 | 741 |
| **7** | `STR` | Stroma | Cancer-associated fibrous stroma; desmoplastic fibroblasts, extracellular collagen matrix, and vascular elements surrounding tumor glands. | ~10,446 | 421 |
| **8** | `TUM` | Colorectal Epithelium | Colorectal adenocarcinoma epithelium; irregular, crowded malignant tumor glands with cytological atypia, nuclear enlargement, and hyperchromasia. | ~14,317 | 1,233 |
| **Total** | | | | **100,000** | **7,180** |

---

## 3. Exact Split Strategy

To evaluate generalization to unseen staining conditions without data leakage, the split strategy partitions data at the domain and cohort level:

```
+-----------------------------------------------------------------------------------+
| SOURCE DOMAIN: NCT-CRC-HE-100K-NONORM (Heidelberg / Mannheim Cohort)              |
|                                                                                   |
|  [Train Set: 70%]              [Val-ID: 15%]              [Test-ID: 15%]          |
|  Model parameter update        Early stopping,            In-Domain Performance   |
|  and loss computation          checkpoint selection       Baseline Benchmark      |
+-----------------------------------------------------------------------------------+
                                                                   |
                                                                   v Compare Drop (Delta)
+-----------------------------------------------------------------------------------+
| TARGET DOMAIN: CRC-VAL-HE-7K (RWTH Aachen Cohort, 50 Unseen Patients)             |
|                                                                                   |
|  [Test-OOD Set: 100% of 7,180 patches]                                            |
|  Strictly held-out out-of-domain evaluation. Never used during training/tuning.   |
+-----------------------------------------------------------------------------------+
```

### Partitioning Rules
1. **Source Domain Split (`NCT-CRC-HE-100K-NONORM`)**:
   - **Train Set (70%)**: Used solely for updating neural network weights.
   - **In-Domain Validation Set (`Val-ID`, 15%)**: Used solely for learning rate scheduling, early stopping, and selecting the optimal checkpoint (`best_checkpoint.pt`).
   - **In-Domain Test Set (`Test-ID`, 15%)**: Evaluated once using `best_checkpoint.pt` to establish baseline in-domain performance.
   - **Stratification**: Sampling must be stratified across all 9 classes using a deterministic random seed (`seed = 42`).
2. **Compute-Budget Benchmark Subset (Standardized 25K Subset)**:
   - For rapid ablation iteration on a single consumer GPU, an optional standardized stratified subset of **25,000 patches** from `NCT-CRC-HE-100K-NONORM` is supported:
     - Train: 17,496 patches (1,944 per class)
     - Val-ID: 3,753 patches (417 per class)
     - Test-ID: 3,753 patches (417 per class)
   - Each part is sampled per class from the subset size, so the split is exactly
     70/15/15 per class: `round(25000/9 x 0.70) = 1944`, `round(25000/9 x 0.15) = 417`.
     The 0.03 % difference from the nominal 17,500 / 3,750 / 3,750 comes purely from
     rounding to whole patches per class.
   - The three parts are sampled **disjointly** (train, then val from the remainder,
     then test), so no image can appear in two splits; `prepare_splits.py` re-checks
     this and fails if it ever happens.
   - The preparation script `scripts/prepare_splits.py` supports both `--subset 25000`
     and `--full` (all 100,000 patches). The default configuration uses the 25,000
     subset to ensure all 13 ablation experiments complete within 6-8 hours on a
     single GPU. `tests/splits_test.py` verifies both modes against the real
     per-class counts.
3. **Target Domain (`CRC-VAL-HE-7K`)**:
   - **Out-of-Domain Test Set (`Test-OOD`, 100%, 7,180 patches)**:
   - Evaluated using the exact same `best_checkpoint.pt` selected via `Val-ID`.
   - Under no circumstances may `CRC-VAL-HE-7K` be used for training, normalization reference fitting, early stopping, or hyperparameter selection.
4. **Split Artifacts**:
   - The preparation script generates deterministic CSV files in `data/processed/splits/`:
     - `train.csv` (columns: `image_path,class_name,label_idx,domain`)
     - `val_id.csv` (columns: `image_path,class_name,label_idx,domain`)
     - `test_id.csv` (columns: `image_path,class_name,label_idx,domain`)
     - `test_ood.csv` (columns: `image_path,class_name,label_idx,domain`)

---

## 4. Expected Folder Layout

After downloading, extracting, and running `data/scripts/prepare_splits.py`, the repository data tree must match the following structure:

```text
data/
├── raw/
│   ├── NCT-CRC-HE-100K-NONORM/
│   │   ├── ADI/
│   │   │   ├── ADI-TCGA-AA-3511-01Z-00-DX1.png
│   │   │   └── ...
│   │   ├── BACK/
│   │   ├── DEB/
│   │   ├── LYM/
│   │   ├── MUC/
│   │   ├── MUS/
│   │   ├── NORM/
│   │   ├── STR/
│   │   └── TUM/
│   └── CRC-VAL-HE-7K/
│       ├── ADI/
│       │   ├── ADI-TCGA-A6-2675-01Z-00-DX1.png
│       │   └── ...
│       ├── BACK/
│       ├── DEB/
│       ├── LYM/
│       ├── MUC/
│       ├── MUS/
│       ├── NORM/
│       ├── STR/
│       └── TUM/
└── processed/
    └── splits/
    │   ├── train.csv
    │   ├── val_id.csv
    │   ├── test_id.csv
    │   └── test_ood.csv
    └── templates/
        └── reference_stain.png

```
