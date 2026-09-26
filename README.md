# Robust Histopathology Image Classification under Staining Variations

Single-GPU study of **what actually makes a histopathology classifier robust to
staining variation**: colour normalisation, augmentation policy, and pretrained
backbone choice, evaluated as an in-domain → out-of-domain transfer problem.

* **Source (in-domain)**: `NCT-CRC-HE-100K-NONORM` — 100,000 raw H&E patches,
  Heidelberg/Mannheim, 9 colorectal tissue classes, 224×224.
* **Target (out-of-domain)**: `CRC-VAL-HE-7K` — 7,180 patches from 50 independent
  RWTH Aachen patients. Never used for training, scheduling, or checkpoint
  selection.

Design and data records: [`docs/PLAN.md`](docs/PLAN.md) (experiment design,
evaluation protocol, tech stack) and [`docs/dataset_card.md`](docs/dataset_card.md)
(dataset, splits, folder layout). Everything else is indexed at
[`docs/README.md`](docs/README.md).

This is a **standalone project*