# Robust Histopathology Image Classification under Staining Variations

Study of **what actually makes a histopathology classifier robust to
staining variation**: colour normalisation, augmentation policy, and pretrained
backbone choice, evaluated as an in-domain to out-of-domain transfer problem.

* **Source (in-domain)**: `NCT-CRC-HE-100K-NONORM` — 100,000 raw H&E patches,
  Heidelberg/Mannheim, 9 colorectal tissue classes, 224×224.
* **Target (out-of-domain)**: `CRC-VAL-HE-7K` — 7,180 patches from 50 independent
  RWTH Aachen patients. Never used for training, scheduling, or checkpoint
  selection.

## Documentation

Start at [`docs/README.md`](docs/README.md) for the full index. The short version:

| Document | Read it to… |
| :--- | :--- |
| [`docs/PLAN.md`](docs/PLAN.md) | understand what is tested and why (§3 holds the 13-cell matrix) |
| [`docs/dataset_card.md`](docs/dataset_card.md) | know the exact data, class order and split rules |
| [`docs/kaggle_guide.md`](docs/kaggle_guide.md) | run it on Kaggle, including the between-session checkpoint workflow |
| [`docs/RUN_LOG.md`](docs/RUN_LOG.md) | record what each session did |
| [`docs/RESULTS.md`](docs/RESULTS.md) | report the metrics in the agreed table shape |
| [`docs/APPENDICES.md`](docs/APPENDICES.md) | look up an output file, config key or CLI flag |
| [`docs/workflow.md`](docs/workflow.md) | the original three-agent brief (provenance only) |