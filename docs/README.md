# Project Documentation

Documentation for **Robust Histopathology Image Classification under Staining
Variations**

The repository entry point is [`../README.md`](../README.md). This folder holds everything else.

---

## Which document do I need?

| I want to… | Read |
| :--- | :--- |
| **New contributor onboarding & architecture** | [`contributor_guide.md`](contributor_guide.md) |
| Understand **what is being tested and why** | [`PLAN.md`](PLAN.md) |
| Know the **exact dataset, classes and split rules** | [`dataset_card.md`](dataset_card.md) |
| **Run it on Kaggle**, step by step | [`kaggle_guide.md`](kaggle_guide.md) |
| Record **what a session actually did** | [`RUN_LOG.md`](RUN_LOG.md) |
| **Report results** in the agreed table shape | [`RESULTS.md`](RESULTS.md) |
| Look up a **config key, CLI flag or output file** | [`APPENDICES.md`](APPENDICES.md) |
| Follow the **Git commit message standard** | [`conventional_commits.md`](conventional_commits.md) |
| See the **original three-agent handover brief** | [`workflow.md`](workflow.md) |

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

* [`kaggle_guide.md`](kaggle_guide.md) — the zipping rule, the three upload
  datasets, the between-session checkpoint workflow via **New Version**, and the
  click-by-click execution procedure. §0–§5 is the critical path; §7–§9 are
  offline weights, the storage budget and troubleshooting.

The notebook itself is `../kaggle/notebook_01_run_all_ablations.ipynb`

### 4. Record, then report

* [`RUN_LOG.md`](RUN_LOG.md) — one attempt block per cell per session.
* [`RESULTS.md`](RESULTS.md) — the results tables, one column per metric.

**Both start out pending on purpose.** No number in them is invented; every value
is copied from a file under `results/` after a run has finished. The mapping from
table column to source file is in [`APPENDICES.md`](APPENDICES.md) §1.

### 5. Reference

* [`APPENDICES.md`](APPENDICES.md) — artifact schemas, full config-key
  reference with defaults and allowed values, CLI reference, troubleshooting and
  the glossary.

---

## Document status

| Document | Nature | Owner | Status |
| :--- | :--- | :--- | :--- |
| [`PLAN.md`](PLAN.md) | Decision record — frozen unless the design genuinely changes | planning agent | complete |
| [`dataset_card.md`](dataset_card.md) | Decision record — frozen | planning agent | complete |
| [`kaggle_guide.md`](kaggle_guide.md) | Operating procedure | implementation agent | complete |
| [`RUN_LOG.md`](RUN_LOG.md) | Fill-in template | whoever runs the session | **pending runs** |
| [`RESULTS.md`](RESULTS.md) | Fill-in template | whoever runs the session | **pending runs** |
| [`APPENDICES.md`](APPENDICES.md) | Reference derived from the code | implementation agent | complete |
| [`workflow.md`](workflow.md) | Historical — the original three-agent brief | — | superseded |

`workflow.md` is kept only for provenance. Where it disagrees with `PLAN.md` or
`kaggle_guide.md`, it is wrong — it was written before the dataset and the Kaggle
constraints were settled.

---

## Ground rules these documents share

1. **The split seed is 42** and is fixed in `configs/base_config.yaml`
   (`runtime.seed` and `scripts/prepare_splits.py --seed`).
2. **Class order is positional.** `ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM`
   ↔ indices `0…8`. Reordering it silently invalidates every stored metric.
3. **One cell changes one axis.** All 13 configs must share learning rate,
   epochs, weight decay, label smoothing, resolution, effective batch size,
   schedule and checkpoint policy. `tests/local_selftest.py` enforces this.
4. **Nothing is reported before it is run.** Templates stay `pending`; results
   come from artifacts, not from memory.
5. **Verify at the boundary you can still fix.** The codebase archive is checked
   where it is built (`make_zips.py` asserts 18 required entries), and the Kaggle
   mount is checked before any long work (notebook cell 0). Both checks exist
   because a run was lost to each.

---
