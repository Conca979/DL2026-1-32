# Project Documentation

Documentation for **Robust Histopathology Image Classification under Staining
Variations**

The repository entry point is [`../README.md`](../README.md). This folder holds everything else.

---

## Which document do I need?

| I want to… | Read |
| :--- | :--- |
| Understand **what is being tested and why** | [`PLAN.md`](PLAN.md) |
| Know the **exact dataset, classes and split rules** | [`dataset_card.md`](dataset_card.md) and [`DATA.md`](../DATA.md) |
| **Run it on Kaggle**, step by step | [`kaggle_guide.md`](kaggle_guide.md) |
| **Report results** in the agreed table shape | [`RESULTS.md`](RESULTS.md) |
---

## Reading order

### 1. Orient (10 minutes)

* [`PLAN.md`](PLAN.md) §1–§3 — problem statement, dataset decision, and the
  13-cell ablation matrix. §3's table is the spine of the whole project: every
  config, log and results table follows its `EXP-01 … EXP-13` ordering.

### 2. Understand the data (10 minutes)

* [`dataset_card.md`](dataset_card.md) — nine tissue classes, class indices 0–8
  (positional, never reorder), the 70/15/15 stratified source split at seed 42,
  the full target cohort as `test_ood`, and the expected folder layout.

The two facts that constrain everything else:

1. The target cohort is **imbalanced** (`ADI` 1,338 vs `DEB` 339), which is why
   **Macro-F1**, not accuracy, is the primary metric.
2. `CRC-VAL-HE-7K` is a **firewall domain**: never trained on, never used for
   scheduling, early stopping or checkpoint selection, never a source for the
   stain reference tile.

### 3. Run it (the long part)

* [`kaggle_guide.md`](kaggle_guide.md) — the zipping rule, the two upload datasets, the click-by-click execution procedure.

The notebook itself is `../kaggle/notebook_01_run_all_ablations.ipynb`

### 4. Report

* [`RESULTS.md`](../results/RESULTS_TABLE.md) — the results tables, one column per metric.

**Start out pending on purpose.** No number in it is invented; every value is mapped after a run has finished.

---

## Document status

| Document | Nature | Status |
| :--- | :--- | :--- |
| [`PLAN.md`](PLAN.md) | Decision record — frozen unless the design genuinely changes | complete |
| [`dataset_card.md`](dataset_card.md) | Decision record — frozen | planning agent | complete |
| [`kaggle_guide.md`](kaggle_guide.md) | Operating procedure| complete |
| [`RESULTS.md`](RESULTS.md) | Fill-in template | Done |

---