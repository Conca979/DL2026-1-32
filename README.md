# Robust Histopathology Image Classification under Staining Variations

**DL2026 — Group 1 · Project 32**

A 13-cell ablation study measuring what actually makes deep learning classifiers robust to clinical histopathology staining variation: color normalization, data augmentation, and pretrained backbone choice.

* **In-Domain (Source)**: `NCT-CRC-HE-100K-NONORM` — 100,000 raw, unnormalized H&E colorectal cancer tiles (Heidelberg / Mannheim).
* **Out-of-Domain (Target)**: `CRC-VAL-HE-7K` — 7,180 tiles from 50 independent patients at RWTH Aachen, strictly firewall-protected.

---

## Headline Results

Macro-F1 on the held-out in-domain test split (`ID`) and the external hospital cohort (`OOD`). `Δ-F1 = ID − OOD` (lower is better); `RR = OOD / ID` (higher is better).

| Configuration | Cell | ID F1 | OOD F1 | Δ-F1 | RR |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Raw baseline (ResNet-50) | EXP-01 | 0.9893 | 0.6644 | 0.3249 | 67.2% |
| Best overall (ResNet-50, Macenko + geometric aug) | EXP-07 | 0.9717 | **0.8689** | **0.1028** | **89.4%** |
| Best foundation model (Phikon, undefended) | EXP-12 | 0.9912 | 0.8239 | 0.1673 | 83.1% |
| Phikon + full defense | EXP-13 | 0.9050 | 0.6982 | 0.2068 | 77.2% |

Three findings drive the study: stain normalization recovers most of the lost robustness (+18 F1 points), augmentation is largely **redundant once normalization is applied**, and a frozen histology foundation encoder — the strongest single model when undefended — is the one configuration that defenses **actively break**. Full numbers in [`results/summary_results.csv`](results/summary_results.csv).

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

## Environment

The project runs entirely on **Kaggle**; there is no local execution path.

| Requirement | Value |
| :--- | :--- |
| **Accelerator** | GPU `T4 x2` |
| **Internet** | **On** — required to fetch ImageNet weights (`timm`) and `owkin/phikon` (`transformers`) |
| **Python / PyTorch** | 3.10+ / 2.x, as supplied by the Kaggle image |
| **Provided by the Kaggle image** | `torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `Pillow` |
| **Installed by the notebook** | `timm`, `transformers`, `tabulate` — `notebook.ipynb` Cell 1 |

No dependency manifest is committed: the only packages that need installing are the three above, and the notebook installs them automatically.

---

## Reproducing the Main Results (read on docs/kaggle_guide.md for more information)

**1. Build the code archive**

```bash
python scripts/make_zips.py      # → dist/histo-robust-code.zip (contains run_experiments.py only)
```

**2. Upload it to Kaggle** as a dataset named `histo-robust-code` (Private visibility is fine).

**3. Attach three inputs** to a new Kaggle notebook, under **Add Input → Your Datasets**:

| Dataset | Role |
| :--- | :--- |
| `histo-robust-code` | the pipeline script built in step 1 |
| `nct-crc-he-100k-nonorm` | in-domain source patches |
| `crc-val-he-7k` | out-of-domain target patches |

**4. Import and run** [`notebook.ipynb`](notebook.ipynb) via **File → Import Notebook**, then **Run All**. The five cells install dependencies, stage the script into `/kaggle/working`, run all 13 experiments sequentially with live logs, and render the final table. Expect **~3.5 hours** on a single P100.

**5. Collect the artifacts** from the notebook's **Output** tab, under `results/`:

- `RESULTS_TABLE.md` — the formatted 13-cell table
- `summary_results.csv` — the same numbers, machine-readable
- `splits/` — the four split CSVs and the canonical stain reference tile

The notebook reproduces the table in this README and in [`docs/RESULTS.md`](docs/RESULTS.md).

**Data details** — official dataset URL and version, class definitions, the 70/15/15 split protocol, the domain firewall, and every preprocessing step — are documented in [`DATA.md`](DATA.md). A click-by-click version of the steps above is in [`docs/kaggle_guide.md`](docs/kaggle_guide.md).

---

## Repository Structure

```text
├── README.md                    # this file
├── DATA.md                      # dataset source, version, split protocol, preprocessing
├── run_experiments.py           # the complete pipeline (data prep → train → evaluate)
├── notebook.ipynb               # Kaggle execution notebook
│
├── results/                     # committed run artifacts
│   ├── summary_results.csv      # 13-cell metrics table (machine-readable)
│   ├── RESULTS_TABLE.md         # the same table in Markdown
│   └── splits/                  # train / val_id / test_id / test_ood splits
│                                #   + reference_stain.png (canonical stain reference)
│
├── docs/
│   ├── README.md                # documentation index
│   ├── PLAN.md                  # research design, 13-cell matrix, evaluation math
│   ├── RESULTS.md               # results table + per-column reference
│   ├── CODE_WALKTHROUGH.md      # line-by-line explanation of the pipeline
│   ├── dataset_card.md          # 9 classes, split rules, domain firewall
│   ├── kaggle_guide.md          # step-by-step Kaggle execution guide
│   └── exam_requirement.md      # course rubric
│
└── scripts/
    └── make_zips.py             # packages run_experiments.py for the Kaggle dataset
```
