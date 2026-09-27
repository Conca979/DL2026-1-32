# RESULTS — Ablation Matrix

Every cell below is **pending**. Nothing in this file may be filled in until the
corresponding run has actually happened; all numbers come from
`results/metrics/summary_results.csv`, which is written automatically by
`scripts/run_all_ablations.py`.

## How to read / populate this file

Sources, per cell:

| Value | Source file | Field |
| :--- | :--- | :--- |
| Val Macro-F1 (best epoch) | `checkpoints/<EXP>/train_report.json` | `train_result.best_val_macro_f1` (epoch in `best_epoch`) |
| Epochs completed / stop reason | same | `train_result.epochs_completed`, `train_result.stop_reason` |
| Wall time (min) | same | `train_result.trained_seconds / 60` (also in `summary_results.csv`) |
| All test metrics | `results/per_experiment/<EXP>/metrics_test_id.json`, `metrics_test_ood.json` | `accuracy`, `balanced_acc`, `macro_f1`, `macro_auroc`, `ece`, `per_class` |
| Robustness deltas | `results/per_experiment/<EXP>/robustness.json` | `delta_*`, `rr_*` |
| Per-class table | `results/per_experiment/<EXP>/metrics_test_*.json` → `per_class` | precision / recall / f1 / auroc / support |
| Confusion matrices | `results/per_experiment/<EXP>/confusion_test_{id,ood}.csv` and `.png` | — |
| Raw predictions (for re-derivation) | `results/per_experiment/<EXP>/predictions_test_{id,ood}.npz` | `y_true`, `probs` |

`results/metrics/RESULTS_TABLE.md` (produced by `kaggle/notebook_01_run_all_ablations.ipynb` Cell 5)
renders the main table straight from the CSV, so prefer pasting that over typing
numbers by hand.

### Metric definitions (`docs/PLAN.md` §4)

* **Macro-F1** — primary metric. Harmonic mean of precision and recall per class,
  then averaged over the nine classes. Immune to the class imbalance in
  `CRC-VAL-HE-7K` (`ADI` 1,338 vs `DEB` 339).
* **Bal-Acc** — balanced accuracy (macro-averaged recall).
* **Acc** — overall top-1 accuracy; comparable to published numbers, but biased
  toward majority classes.
* **AUROC** — macro one-vs-rest AUROC over softmax probabilities.
* **ECE** — expected calibration error, 15 bins, confidence-based.
* **Δ (delta)** — `ID − OOD`. **Lower is better**; `0` means perfect stain
  invariance.
* **RR (retention rate)** — `OOD / ID × 100`. **Higher is better**; `100` means no
  diagnostic power is lost on external laboratory stains.

### Firewall statement

`CRC-VAL-HE-7K` (`test_ood`) is never used for training, learning-rate scheduling,
early stopping, checkpoint selection or hyper-parameter choice. Checkpoint
selection uses `val_id` Macro-F1 only, and the OOD audit runs strictly after
training has finished. Splits are produced by `scripts/prepare_splits.py`, which
asserts disjoint paths and that `train`/`val_id` contain the source domain only.

---

## Table 1 — Main Ablation Matrix (`docs/PLAN.md` §3)

| Exp ID | Stage | Backbone | Weights | Normalization | Augmentation | Status | Epochs (completed/planned) | Wall time (min) | Val Macro-F1 | Test-ID Macro-F1 | Test-OOD Macro-F1 | **ΔF1 (ID−OOD)** | **RR-F1 (%)** |
| :---: | :---: | :--- | :--- | :--- | :--- | :---: | :---: | ---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Stage 0 | ResNet-50 | ImageNet-1k | None | None | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-02** | Stage 1 | ResNet-50 | ImageNet-1k | Reinhard | None | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-03** | Stage 1 | ResNet-50 | ImageNet-1k | Macenko | None | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-04** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Geo | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-05** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Stain | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-06** | Stage 2 | ResNet-50 | ImageNet-1k | None | Aug-Combined | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-07** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Geo | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-08** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Stain | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-09** | Stage 3 | ResNet-50 | ImageNet-1k | Macenko | Aug-Combined | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-10** | Stage 4 | ConvNeXt-T | ImageNet-1k | None | None | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-11** | Stage 4 | ConvNeXt-T | ImageNet-1k | Macenko | Aug-Combined | pending | pending | pending | pending | pending | pending | pending | pending |

`Aug-Geo` = random H/V flips + random 90° rotations.
`Aug-Stain` = H&E stain concentration scaling/shift (Beer-Lambert deconvolution)
plus brightness/contrast and HSV perturbation.
`Aug-Combined` = `Aug-Geo` ∘ `Aug-Stain`.

---

## Table 2 — Full Metric Set per Cell

Number format: 4 decimals. `ID` = `test_id` (source domain, held out at 15%);
`OOD` = `test_ood` (full `CRC-VAL-HE-7K`).

| Exp ID | Status | Acc ID | Acc OOD | Bal-Acc ID | Bal-Acc OOD | Δ Bal-Acc | RR Bal-Acc | AUROC ID | AUROC OOD | Δ AUROC | RR AUROC | ECE ID | ECE OOD |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-02** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-03** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-04** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-05** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-06** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-07** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-08** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-09** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-10** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| **EXP-11** | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

`ECE` is reported for calibration context only; it is not used to rank cells.
A second definition (`ece_label`, probability assigned to the true class) is
available in the JSON if the confidence-based ECE saturates.

---

## Table 3 — Per-Class OOD Performance

Schema and class order match `docs/dataset_card.md` §2. Populated from
`results/per_experiment/<EXP>/metrics_test_ood.json` → `per_class`. `AUROC` is
one-vs-rest for that class.

### EXP-01 — Primary Baseline (pending)

| Class | Index | Precision | Recall | F1 | AUROC | Support (OOD) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| ADI (adipose) | 0 | pending | pending | pending | pending | 1,338 |
| BACK (background) | 1 | pending | pending | pending | pending | 1,056 |
| DEB (debris) | 2 | pending | pending | pending | pending | 339 |
| LYM (lymphocytes) | 3 | pending | pending | pending | pending | 638 |
| MUC (mucus) | 4 | pending | pending | pending | pending | 617 |
| MUS (smooth muscle) | 5 | pending | pending | pending | pending | 592 |
| NORM (normal mucosa) | 6 | pending | pending | pending | pending | 876 |
| STR (stroma) | 7 | pending | pending | pending | pending | 421 |
| TUM (tumour epithelium) | 8 | pending | pending | pending | pending | 1,303 |
| **Macro average** | — | pending | pending | pending | pending | 7,180 |

*(Blocks for EXP-02 through EXP-11 follow this same schema once runs complete.)*

---

## Table 4 — Confusion Matrix Artefacts

Row-normalised 9×9 matrices (rows = true class, columns = predicted). CSV and PNG
are written per cell per split; list them here once produced.

| Exp ID | Confusion CSV (ID) | Confusion CSV (OOD) | Heatmap PNG (OOD) | Invariant classes (no confusions) | Dominant confusion noted |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | pending | pending | pending | pending | pending |
| **EXP-02** | pending | pending | pending | pending | pending |
| **EXP-03** | pending | pending | pending | pending | pending |
| **EXP-04** | pending | pending | pending | pending | pending |
| **EXP-05** | pending | pending | pending | pending | pending |
| **EXP-06** | pending | pending | pending | pending | pending |
| **EXP-07** | pending | pending | pending | pending | pending |
| **EXP-08** | pending | pending | pending | pending | pending |
| **EXP-09** | pending | pending | pending | pending | pending |
| **EXP-10** | pending | pending | pending | pending | pending |
| **EXP-11** | pending | pending | pending | pending | pending |

Path convention:
`results/per_experiment/<EXP>/confusion_test_id.csv`,
`.../confusion_test_ood.csv`, `.../confusion_test_ood.png`.

---

## Table 5 — Robustness & Configuration Diagnostics

Filled from the run logs and normalisation manifests (`results/logs/ablation_suite.log`
and `results/metrics/summary_results.csv`).

| Exp ID | Normalizer | Reference provenance | Precomputed Macenko cache used? | Config hash | Stop reason |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **EXP-01** | none | n/a | n/a | pending | pending |
| **EXP-02** | reinhard | pending | n/a | pending | pending |
| **EXP-03** | macenko | pending | pending | pending | pending |
| **EXP-04** | none | n/a | n/a | pending | pending |
| **EXP-05** | none | n/a | n/a | pending | pending |
| **EXP-06** | none | n/a | n/a | pending | pending |
| **EXP-07** | macenko | pending | pending | pending | pending |
| **EXP-08** | macenko | pending | pending | pending | pending |
| **EXP-09** | macenko | pending | pending | pending | pending |
| **EXP-10** | none | n/a | n/a | pending | pending |
| **EXP-11** | macenko | pending | pending | pending | pending |

---

## Fill-in checklist (one line per cell; tick when Table 1 is complete)

| Exp ID | ID metrics | OOD metrics | Robustness Δ/RR | Per-class | Confusion matrix | Review note written |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EXP-01 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-02 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-03 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-04 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-05 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-06 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-07 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-08 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-09 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-10 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| EXP-11 | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## Analysis notes

*No analysis, discussion or conclusions are recorded here yet — every cell is
still pending. Interpretation belongs in the final report once Table 1 is
populated from real runs.*
