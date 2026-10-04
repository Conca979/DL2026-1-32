# Code Walkthrough: `run_experiments.py` and `notebook.ipynb`

A line-by-line, concrete explanation of the two executable artifacts in this repository.

| File | Lines | Role |
| :--- | :---: | :--- |
| [`run_experiments.py`](../run_experiments.py) | 600 | The entire study: stain normalization, augmentation, data splits, models, training, evaluation, reporting. Self-contained — no repo imports. |
| [`notebook.ipynb`](../notebook.ipynb) | 5 cells | Kaggle wrapper: installs dependencies, copies the script into `/kaggle/working`, runs it as a subprocess, renders the results table. |

**How to read the annotations.** Each section shows the real source with line numbers, followed by per-line notes. `L<n>` refers to a line in `run_experiments.py` unless the section says otherwise. Where a line has a non-obvious consequence — a numerical guard, an ordering dependency, a failure mode — the note explains it.

> **Ground truth note.** Where the docs disagree with the code, the code wins. The known discrepancies are listed in [§ C.5](#c5-known-doc-vs-code-discrepancies). Line numbers refer to the **587-line revision** of `run_experiments.py` current at the time of writing — re-check the anchors if the file changes.

---

## Table of Contents

**Part A — `run_experiments.py`**
1. [Header, constants, and the 13-cell matrix](#a1-header-constants-and-the-13-cell-matrix-l1l41)
2. [Reinhard normalization](#a2-reinhard-normalization-l44l100)
3. [Macenko normalization](#a3-macenko-normalization-l103l164)
4. [Augmentation: HED stain jitter + `PatchTransform`](#a4-augmentation-hed-stain-jitter--patchtransform-l165l226)
5. [Dataset, splits, and the reference tile](#a5-dataset-splits-and-the-reference-tile-l226l351)
6. [Model construction](#a6-model-construction-l352l380)
7. [Evaluation and the training loop](#a7-evaluation-and-the-training-loop-l381l512)
8. [`main()` — the orchestrator](#a8-main--the-orchestrator-l513l587)

**Part B — `notebook.ipynb`**
9. [Cell-by-cell](#part-b--notebookipynb)

**Part C — Cross-cutting**
10. [End-to-end trace of EXP-09](#c1-end-to-end-trace-one-tile-through-exp-09)
11. [Reproducibility and determinism](#c2-reproducibility-and-determinism)
12. [Gotchas and failure modes](#c3-gotchas-and-failure-modes)
13. [Runtime and resource model](#c4-runtime-and-resource-model)
14. [Known doc-vs-code discrepancies](#c5-known-doc-vs-code-discrepancies)

---

# Part A — `run_experiments.py`

## A.0 The big picture

The script executes a 13-cell ablation: every cell is one `{backbone, normalization, augmentation}` combination. All 13 cells see **exactly the same** train/val/test splits and the **same** stain reference tile, so any difference in the results is attributable to the defense being tested, not to data luck.

```
                       ┌──────────────────────────────────────────────┐
data/raw/NCT-CRC-HE-100K-NONORM (source)     data/raw/CRC-VAL-HE-7K (target)
                       │                                              │
              prepare_dataset_splits()  ──► results/splits/*.csv      │
                       │                    results/splits/reference_stain.png
                       ▼                                              │
        ┌──────────────────────────────┐                              │
        │train 70% │ val 15% │ test 15%│   (never touched by target)  │
        └──────────────────────────────┘                              │
                       │                                              │
        for exp in EXPERIMENTS (13 cells):                            │
             ┌─────────┴──────────┐                                   │
             ▼                    ▼                                   ▼
        norm_fn (fit on reference tile, applied per tile)  ─────► test_ood
             │
        PatchTransform(train) = norm → HED jitter (p=.8) → tensor → flips/rot → ImageNet norm
             │
        build_model(backbone)  ──►  AdamW + cosine LR + AMP + label smoothing
             │
        per-epoch validate → keep best macro-F1 checkpoint → evaluate test_id & test_ood
             │
        summary_results.csv + RESULTS_TABLE.md
```

**Design invariant**: `CRC-VAL-HE-7K` is *only ever* read at `test_ood` evaluation time. It is never used for training, normalization-reference selection, hyperparameter tuning, or early stopping — the "domain firewall" from `docs/PLAN.md` §4.1. The code enforces this structurally: `prepare_dataset_splits` builds `test_ood_df` from `target_dir` and nothing else ever reads `target_dir`.

---

## A.1 Header, constants, and the 13-cell matrix (L1–L41)

```python
 1  from __future__ import annotations
 2
 3  import argparse
 4  import copy
 5  import os
 6  import random
 7  import time
 8  from pathlib import Path
 9  from typing import Any, Dict, List, Tuple
10
11  import numpy as np
12  import pandas as pd
13  from PIL import Image
14
15  CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
16  CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
17  NUM_CLASSES = len(CLASSES)
18
19  IMAGENET_MEAN = [0.485, 0.456, 0.406]
20  IMAGENET_STD = [0.229, 0.224, 0.225]
```

- **L1** — `from __future__ import annotations` makes all annotations lazy strings. It lets `Path` / `pd.DataFrame` be used in signatures without runtime cost on every call. Cosmetic here (the project targets Python ≥3.10, per `pyproject.toml`), but harmless.
- **L3–L9** — `argparse` for the CLI; `copy` for `deepcopy` of the best checkpoint (L481); `os` solely to detect Windows for the DataLoader worker count (L447); `random` for augmentation randomness *and* to seed it alongside NumPy (L291); `time` for per-experiment wall-clock; `Path` for all filesystem access; `typing` imports for signatures.
- **L11–L13** — The only three third-party imports at module level: NumPy, pandas, Pillow. **`torch`, `timm`, `transformers`, and `sklearn` are deliberately imported *inside* functions** (L204–209, L317, L355–366, L385–397, L426–439). Two consequences: (a) `python run_experiments.py --dry-run` works on a machine with no deep-learning stack installed, and (b) on Kaggle, the heavy imports happen after the "Running on device" banner instead of delaying startup.
- **L15** — `CLASSES` is the canonical label ordering. The index in this list **is** the training label: `ADI=0, BACK=1, DEB=2, LYM=3, MUC=4, MUS=5, NORM=6, STR=7, TUM=8`. This ordering is used in three places that must stay consistent: `prepare_dataset_splits` assigns `label_idx = c_idx` (L298, L326), the model heads are built with `num_classes=9`, and `np.argmax` at L398 produces the same index space. Reordering this list silently relabels the entire dataset — it is alphabetical, which is why it looks arbitrary but is not.
- **L16** — `CLASS_TO_IDX` is currently **unused** by the script. It is kept as the documented inverse of `CLASSES` for downstream analysis (e.g. mapping a confusion-matrix row back to a tissue name).
- **L17** — `NUM_CLASSES = 9` is derived, not hardcoded, so adding a class to `CLASSES` propagates to the model heads and to the per-class subsample math.
- **L19–L20** — ImageNet channel statistics, applied as the final transform step (L226). Two reasons they are the right constants even for the pathology models: the timm backbones were pretrained *with* this normalization, and Phikon (Owkin's TCGA-pretrained ViT) also uses standard ImageNet normalization. Using per-dataset statistics instead would silently shift the input distribution away from what the frozen/pretrained weights expect.

### The experiment matrix (L22–L41)

```python
22  EXPERIMENTS: List[Dict[str, str]] = [
23    # Stage 0: Anchor Baseline
24    {"id": "EXP-01", "stage": "Stage 0", "backbone": "resnet50",      "norm": "none",     "aug": "none",        "notes": "Raw baseline"},
25    # Stage 1: Color Normalization
26    {"id": "EXP-02", ... "norm": "reinhard", "aug": "none",        "notes": "Statistical LAB transfer"},
27    {"id": "EXP-03", ... "norm": "macenko",  "aug": "none",        "notes": "Optical density deconvolution"},
    ...
41  ]
```

This list is the study's single source of truth. The loop in `main()` iterates it verbatim (L555), so editing this list is how you add or remove a cell — nothing else needs to change.

- **`id`** — the join key across `summary_results.csv`, `docs/RESULTS.md`, and the README matrix.
- **`stage`** — carried through to the output CSV for grouped reporting. It is **not** used for control flow; all 13 cells run identically regardless of stage.
- **`backbone`** — dispatch key read by `build_model` (L359, L376). Exactly three legal values: `resnet50`, `convnext_tiny`, `phikon`. Any other string falls through to the ResNet default at L380 — a typo here silently trains a ResNet-50.
- **`norm`** — dispatch key read by `train_experiment` (L435–L440). Exactly three legal values: `none`, `reinhard`, `macenko`. Anything else leaves `norm_fn = None`, i.e. an undisclosed no-op.
- **`aug`** — dispatch key read by `PatchTransform` (L211, L217). Five legal values in practice: `none`, `aug_geo`, `aug_stain`, `aug_combined` — plus any unrecognized string, which behaves identically to `none`.
- **`notes`** — human-readable only, printed in `--dry-run`.

**Cell-to-code map.** This table is the fastest way to answer "which experiment exercises which code path" — useful when a difference between two cells must be explained:

| Cell | `norm` path | `aug` path | Model |
| :--- | :--- | :--- | :--- |
| EXP-01 / 04 / 05 / 06 | none (norm_fn = `None`) | none / geo / stain / combined | ResNet-50 |
| EXP-02 | `reinhard_fit` + `reinhard_apply` (L436–448) | none | ResNet-50 |
| EXP-03 / 07 / 08 / 09 / 11 / 13 | `macenko_fit` + `macenko_apply` (L439–451) | none / geo / stain / combined | ResNet-50 / ConvNeXt / Phikon |
| EXP-10 / 12 | none | none | ConvNeXt-Tiny / Phikon |

Note the matrix is *not* a full cross-product: Stage 4 only tests each new backbone at its two extremes (raw vs. full defense). Normalization × augmentation interactions are explored only on ResNet-50 (Stage 3).

---

## A.2 Reinhard normalization (L44–L100)

Reinhard color transfer, as originally described for photographs and adapted here for H&E tiles: convert RGB → the decorrelated log-α-β space, then force the source tile's per-channel mean and standard deviation to match a reference tile's.

### The color-space matrices (L44–L64)

```python
44  # Stain Normalization (Reinhard & Macenko)
45  # Ruderman et al. (1998) matrices for Reinhard l-alpha-beta color transfer
46  _LMS_MAT = np.array([
47    [0.3811, 0.5783, 0.0402],
48    [0.1967, 0.7244, 0.0782],
49    [0.0241, 0.1288, 0.8444],
50  ], dtype=np.float64)
```

- **L44–L45** — Section header comments. Earlier revisions wrapped these in `# ---` ruler lines; the rulers were removed, leaving the section title (`L44`) and the source attribution (`L45`) as plain comments.
- **L46–L50** — RGB → LMS cone-response matrix from Ruderman et al. (1998). This is the canonical matrix used by every Reinhard implementation (including the widely-copied `staintools` package). Note it is stored as a `float64` array; `dtype=np.float64` matters because these are applied to `uint8`-derived data and precision loss here would compound through the `log10` at L70.
- **L52–L56** — LMS → log-α-β rotation. The `1/√3`, `1/√6`, `1/√2` structure is an orthonormal rotation, which is why the inverse at L58–L62 can be written out by inspection rather than recomputed.
- **L58–L62** — The inverse rotation, written explicitly rather than as `np.linalg.inv(_LAB_MAT)`. It is the transpose (orthonormal ⇒ inverse = transpose), but spelling it out makes the round trip auditable at a glance.
- **L64** — `_INV_LMS_MAT` **is** computed with `np.linalg.inv`, unlike L58–L62. RGB→LMS is not orthonormal, so there is no transpose shortcut. Computed once at import time, so the inversion cost is paid once per process, not once per tile.

### Forward and inverse transforms (L67–L78)

```python
67  def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
68    norm_rgb = np.clip(rgb.astype(np.float64) / 255.0, 1e-4, 1.0)
69    lms = norm_rgb @ _LMS_MAT.T
70    log_lms = np.log10(np.clip(lms, 1e-4, None))
71    return log_lms @ _LAB_MAT.T
```

- **L67** — Private helper (leading underscore): not part of the public surface, only called from `reinhard_fit`/`reinhard_apply`. Takes an `(H, W, 3)` `uint8` array.
- **L68** — Two things happen: scale to `[0, 1]` by `/255.0`, and clamp into `[1e-4, 1.0]`. `astype(np.float64)` **before** the division is required — dividing a `uint8` array in place would truncate to 0 for every pixel below 255.
  - The lower clamp at `1e-4` is the standard Reinhard trick: it bounds how dark a pixel can be so that `log10` at L70 stays finite. Without it, a single pure-black pixel (RGB 0,0,0 → LMS 0) yields `log10(0) = -inf`, which propagates `inf`/`nan` into the mean and std and corrupts the *entire tile*, not just that pixel.
- **L69** — `@ _LMS_MAT.T` applies the matrix to every pixel at once. The transposes throughout this file follow a row-vector convention: to compute `M·v` for each row-vector `v`, you write `v @ M.T`. The `.T` is easy to drop by accident and would silently produce wrong colors rather than an error.
- **L70** — `log10`, with an upper-only clamp (`None` as the max). Only the lower bound needs guarding; large LMS values are harmless. Log is applied to LMS, not to RGB — this is what makes the space decorrelated.
- **L71** — Final rotation into log-α-β. The return value is `(H, W, 3)` `float64`, with channel 0 = `l` (luminance), 1 = `α` (yellow↔blue), 2 = `β` (red↔green).

```python
74  def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
75    log_lms = lab @ _INV_LAB_MAT.T
76    lms = 10.0 ** log_lms
77    rgb = lms @ _INV_LMS_MAT.T
78    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
```

- **L74–L77** — Exact inverse of L67–L71 in reverse order: undo rotation → undo log (via lms), undo LMS.
- **L78** — Back to `[0, 255]`, clamp, cast to `uint8`. The clamp is required because the statistics transfer can push values out of gamut (e.g. matching a bright reference tile onto a dark tile can produce pixel values above 255). The `astype(np.uint8)` truncates toward zero rather than rounding — a sub-1-LSB bias, irrelevant here but worth knowing if you ever compare this against an implementation that uses `np.round`.

### Fitting and applying (L81–L100)

```python
81  def reinhard_fit(reference_rgb: np.ndarray) -> Dict[str, np.ndarray]:
82    """Extract mean and standard deviation per channel in Reinhard LAB space."""
83    lab = _rgb_to_lab(reference_rgb)
84    return {
85      "mean": np.mean(lab, axis=(0, 1)),
86      "std": np.std(lab, axis=(0, 1)) + 1e-6,
87    }
```

- **L81** — Takes the reference tile once, at experiment setup.
- **L83** — Converts the whole reference tile to log-α-β.
- **L85** — Per-channel mean over **all pixels** (`axis=(0,1)` collapses height and width, keeping the 3 channels). This is the crux of the method *and* its main failure mode here: the statistics include background. A tile that is 60% white slide background has its mean pulled strongly toward white, and normalizing a different tile to that mean will over-stain its tissue (and vice-versa for a background-dominant `BACK` tile normalized to a tumor reference). Reinhard was designed for photographs where every pixel is "content".
- **L86** — `+ 1e-6` guards against a zero standard deviation. A perfectly uniform reference tile (e.g. all-white) would otherwise divide by zero at L98. The epsilon is far below any real intensity variation, so it never perturbs a legitimate fit.

```python
 90  def reinhard_apply(image_rgb: np.ndarray, ref_stats: Dict[str, np.ndarray]) -> np.ndarray:
 91    """Apply Reinhard statistical color transfer (pure NumPy)."""
 92    lab = _rgb_to_lab(image_rgb)
 93    mean = np.mean(lab, axis=(0, 1))
 94    std = np.std(lab, axis=(0, 1)) + 1e-6
 95
 96    norm_lab = np.zeros_like(lab)
 97    for c in range(3):
98      norm_lab[:, :, c] = ((lab[:, :, c] - mean[c]) / std[c]) * ref_stats["std"][c] + ref_stats["mean"][c]
99
100    return _lab_to_rgb(norm_lab)
```

- **L92–L94** — Compute the *source tile's* own statistics (not the reference's). This is what makes Reinhard adaptive: each tile is standardized against itself, then restyled to the reference. So a tile is never compared to the reference directly — only its *statistics* are.
- **L96** — Pre-allocate the output. `np.zeros_like(lab)` preserves shape and `float64` dtype.
- **L97–L98** — The whole algorithm in one line: **z-score** the tile (`(x − μ)/σ`) then **re-scale** to the reference (`× σ_ref + μ_ref`). Written as an explicit 3-iteration loop rather than vectorized, which is slower but keeps the per-channel indexing obvious. At 224×224×3 there is no measurable performance difference.
- **L100** — Back to `uint8` RGB.

**When this is a no-op**: if the reference tile and the input tile have identical mean and std, L98 reduces to the identity. In practice, tiles from the *same* hospital with similar stain will barely change, and only the source↔target shift gets corrected — which is precisely the intended effect, and why ID accuracy is expected to be roughly flat while OOD accuracy moves.

---

## A.3 Macenko normalization (L103–L164)

Macenko's method works in optical density (OD) space, where the Beer–Lambert law makes stain contributions **additive**: total OD is a linear combination of per-stain OD vectors weighted by stain concentrations. Separating hematoxylin from eosin becomes a matrix factorization problem.

### The physics (expressed in numbers)

OD per pixel is `od = -log10((I + 1) / 256)` (L105, L133). A pure-white pixel (255) gives `-log10(256/256) = 0` — no absorption. A dark pixel gives a large positive OD. Because OD is additive, `od ≈ [h_vector, e_vector] · [h_conc, e_conc]^T`, and since the two stain vectors span a 2-D plane inside 3-D OD space, the two vectors can be recovered from the extreme angles of the pixel cloud projected onto that plane.

### `macenko_fit` — learn the reference (L103–L128)

```python
103  def macenko_fit(reference_rgb: np.ndarray, od_threshold: float = 0.15) -> Dict[str, np.ndarray]:
104    """Estimate H&E stain vectors and 99th percentile concentrations from reference tile."""
105    od = -np.log10((reference_rgb.astype(np.float64) + 1.0) / 256.0)
106    flat_od = od.reshape(-1, 3)
107    mask = np.linalg.norm(flat_od, axis=1) > od_threshold
108    flat_od = flat_od[mask]
109
110    _, _, vh = np.linalg.svd(flat_od, full_matrices=False)
111    proj = flat_od @ vh[:2].T
112    phi = np.arctan2(proj[:, 1], proj[:, 0])
113
114    min_phi = np.percentile(phi, 1.0)
115    max_phi = np.percentile(phi, 99.0)
116
117    v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
118    v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
119
120    # Ensure Hematoxylin is column 0 (stronger in red absorption)
121    if v1[0] < v2[0]:
122      v1, v2 = v2, v1
123
124    stain_matrix = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
125    concentrations = flat_od @ np.linalg.pinv(stain_matrix).T
126    q99 = np.percentile(concentrations, 99.0, axis=0)
127
128    return {"stain_matrix": stain_matrix, "q99": q99}
```

- **L103** — `od_threshold=0.15` is the one tunable parameter in the whole method. It defines "this pixel contains tissue". The default matches the reference Macenko implementation.
- **L105** — The `+1.0` and `/256.0` are the standard Macenko formulation (not a typo for `/255`). They make the transform exactly invertible at L161: `I + 1 = 256 · 10^(-od)`. Any mismatch between this constant and L161's reconstruction would produce a systematic brightness offset.
- **L106** — Flatten `(H, W, 3) → (H·W, 3)` so every subsequent operation is a plain matrix operation on a pixel list.
- **L107–L108** — The tissue mask: keep only pixels whose OD vector has magnitude > 0.15. Background pixels cluster at the origin (OD ≈ 0) and would otherwise dominate the fitted plane — this mask is what makes the deconvolution work at all. Note the mask is applied by `flat_od = flat_od[mask]` (a copy), so the original `od` is unaffected.
- **L110** — SVD of the tissue-pixel OD matrix. `vh[:2]` is the 2-D basis of the plane that contains (almost) all of the data — the stain plane. `full_matrices=False` gives the economy SVD, which is both faster and exactly what's needed here.
- **L111–L112** — Project the 3-D OD cloud onto that plane (L111) and measure the polar angle of each projected pixel (L112). Each stain appears as a *direction* in this plane; the two stains are at the two extremes of the angle distribution. `arctan2(y, x)` (rather than `arctan`) correctly handles all four quadrants.
- **L114–L115** — The extremes are taken at the **1st and 99th percentiles**, not min/max. This is the robustness step: a single stray dark pixel (pen mark, dust, artifact) would otherwise define a stain vector and corrupt every tile normalized against this reference. Note that 1%/99% are *not* symmetric around the extremes — they intentionally shave the tails.
- **L117–L118** — Map the two angles back into 3-D: build the unit vector in 2-D at that angle, then lift it via the plane basis. These are the eosin and hematoxylin OD directions (in an arbitrary order at this point).
- **L120–L122** — Fix the ordering so **hematoxylin is always column 0**. Hematoxylin absorbs red most strongly, so its OD vector has the larger red-channel component (`v1[0]` is the red component of `v1`); the conditional swaps the two so the convention holds regardless of which extreme the angle sort happened to produce. This matters because `macenko_apply` computes per-tile vectors and compares/q99-scales them **positionally** (L159) — if the ordering were inconsistent between fit and apply, the hematoxylin scale factor would be applied to eosin.
- **L124** — Normalize each column to unit length and stack into a `3×2` matrix. Unit-norm vectors mean that all the "how much stain" information lives in the concentrations (L125), not in the vector magnitudes.
- **L125** — Recover concentrations by solving `od = M · c` for `c`. `np.linalg.pinv` is used instead of an exact inverse because `M` is `3×2` (non-square) — the pseudo-inverse gives the least-squares solution. The `.T` again implements the row-vector convention.
- **L126** — The 99th percentile of each stain's concentration. This is Macenko's "normalization target": the amount of stain present in the *most stained 1%* of the reference tile. Robust to outliers in the same way as L114–L115.
- **L128** — `macenko_fit` returns only **10 numbers**: a 3×2 stain matrix (6) and two concentration quantiles. Compare with Reinhard's 6 numbers (3 means + 3 stds). Both are tiny — which is why the reference tile only needs to be saved as an image for reproducibility, not as a serialized object.

### `macenko_apply` — normalize one tile (L131–L164)

```python
131  def macenko_apply(image_rgb, ref_params, od_threshold=0.15) -> np.ndarray:
132    """Normalize an H&E tile to canonical reference stain vectors and concentrations."""
133    od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
134    h, w, _ = od.shape
135    flat_od = od.reshape(-1, 3)
136    mask = np.linalg.norm(flat_od, axis=1) > od_threshold
137
138    if mask.sum() < 0.20 * h * w:
139      return image_rgb
```

- **L133–L136** — Identical to L105–L107: convert to OD, flatten, mask. Deliberately duplicated rather than factored into a helper — the two functions are meant to be readable side by side as the "fit" and "apply" halves of the same procedure.
- **L138–L139** — **The most important guard in the file.** If fewer than 20% of pixels contain tissue, return the tile **completely unmodified**. Sparse tiles (mostly background) cannot support a stable 2-component deconvolution; forcing one produces a garbage stain matrix, which would then be rescaled to the reference q99 and could turn a nearly-white tile into a saturated mess. Returning the original is the conservative, correct choice: an un-normalized tile is a valid H&E image, a mis-normalized one is not. This also means such tiles are *not* domain-shifted in the same way as the rest — a small, documented inconsistency worth remembering if you inspect per-tile results.

```python
141    valid_od = flat_od[mask]
142    try:
143      _, _, vh = np.linalg.svd(valid_od, full_matrices=False)
144      proj = valid_od @ vh[:2].T
145      phi = np.arctan2(proj[:, 1], proj[:, 0])
146      min_phi = np.percentile(phi, 1.0)
147      max_phi = np.percentile(phi, 99.0)
148
149      v1 = vh[:2].T @ np.array([np.cos(min_phi), np.sin(min_phi)])
150      v2 = vh[:2].T @ np.array([np.cos(max_phi), np.sin(max_phi)])
151      if v1[0] < v2[0]:
152        v1, v2 = v2, v1
153      tile_stain = np.column_stack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
154
155      tile_conc = flat_od @ np.linalg.pinv(tile_stain).T
156      q99 = np.percentile(tile_conc[mask], 99.0, axis=0) + 1e-6
157
158      # Match reference concentration scaling
159      norm_conc = tile_conc * (ref_params["q99"] / q99)
160      norm_od = norm_conc @ ref_params["stain_matrix"].T
161      norm_rgb = 256.0 * (10.0 ** -norm_od) - 1.0
162      return np.clip(norm_rgb.reshape(h, w, 3), 0, 255).astype(np.uint8)
163    except Exception:
164      return image_rgb
```

- **L141** — Work only on tissue pixels for the fit. Note `flat_od[mask]` appears again (as `tile_conc[mask]` at L156) — the mask is reused, so the pixel correspondence is preserved.
- **L143–L153** — Byte-for-byte the same procedure as `macenko_fit` L110–L124, but fitted to **this tile** instead of the reference. This is the key structural difference from Reinhard: Macenko re-derives the stain vectors per tile, then only *maps the concentrations* to the reference. Reinhard never re-derives anything — it transfers statistics wholesale. This makes Macenko more faithful to the true stain chemistry (it works in physically meaningful stain concentrations) and Reinhard cheaper and simpler (no SVD, no pseudo-inverse).
- **L155** — Concentrations for **all** pixels (background included), using the tile's own stain matrix. Background pixels have `od ≈ 0` → `conc ≈ 0`, so including them is safe and avoids a second masking round-trip.
- **L156** — The tile's own 99th-percentile concentration, on tissue pixels only (consistent with how the reference q99 was computed at L126). `+ 1e-6` prevents division by zero at L159 for a tile with essentially no stain.
- **L159** — The actual normalization: scale every pixel's concentration so that *this tile's* 99th percentile matches *the reference's* 99th percentile, per stain channel. This single line is what makes tiles from different hospitals comparable — it is a per-stain intensity calibration.
  - Note this is a **multiplicative** correction only (a scale factor), whereas Reinhard is affine (scale *and* shift). Macenko therefore cannot correct a uniformly-tinted background, only stain strength.
- **L160** — Recombine the corrected concentrations using the **reference's** stain vectors, not the tile's. So the output tile is expressed in the canonical stain basis, not its own.
- **L161** — Inverse of L105/L133: `I = 256·10^(−OD) − 1`. The `+1`/`256` pairing with L105/L133 is exact, so an identity input yields the identity output.
- **L162** — Clip to `[0, 255]` — the concentration rescaling can push values out of range when a tile is much more/less stained than the reference — then reshape back to image layout and cast to `uint8`.
- **L163–L164** — A blanket `except Exception: return image_rgb`. Same philosophy as L138–L139: **normalization must never abort a 13-experiment run.** A degenerate tile (e.g. `np.linalg.svd` failing to converge) falls back to the untouched image and training continues. The cost is that failures are invisible — there is no counter or warning, so a systematically failing class of tiles would go unnoticed except as unexplained accuracy. If you need that visibility, this is the place to add a counter or a `logging.warning`.

---

## A.4 Augmentation: HED stain jitter + `PatchTransform` (L165–L226)

### The H&E stain matrix (L168–L173)

```python
168  HE_STAIN_MATRIX = np.array([
169    [0.650, 0.072],
170    [0.704, 0.990],
171    [0.286, 0.105],
172  ], dtype=np.float64)
173  HE_STAIN_MATRIX /= np.linalg.norm(HE_STAIN_MATRIX, axis=0, keepdims=True)
```

- **L168–L172** — The Ruifrok & Johnston H&E OD matrix (3 RGB channels × 2 stains), the same constants used by every stain-deconvolution implementation. Columns: hematoxylin, eosin.
- **L173** — Normalize each **column** independently to unit length. `axis=0` + `keepdims=True` is what makes the norm per-column rather than a single matrix norm; the `keepdims=True` is required for correct broadcasting in the division — dropping it would produce a shape error here (good) or, with a different reshape, silent nonsense (bad). After this line, each column is a unit direction and all magnitude information moves into the concentrations at L182.

### `hed_stain_jitter` (L176–L192)

```python
176  def hed_stain_jitter(image_rgb: np.ndarray, sigma: float = 0.2, bias: float = 0.05) -> np.ndarray:
177    """Apply biologically-grounded H&E stain concentration scaling and shifting."""
178    od = -np.log10((image_rgb.astype(np.float64) + 1.0) / 256.0)
179    h, w, _ = od.shape
180    flat_od = od.reshape(-1, 3)
181
182    c = flat_od @ np.linalg.pinv(HE_STAIN_MATRIX).T
183    # Stochastic independent shift and scale per stain channel
184    alpha = np.random.uniform(1.0 - sigma, 1.0 + sigma, size=2)
185    beta = np.random.uniform(-bias, bias, size=2)
186
187    c[:, 0] = np.clip(c[:, 0] * alpha[0] + beta[0], 0, None)
188    c[:, 1] = np.clip(c[:, 1] * alpha[1] + beta[1], 0, None)
189
190    od_jittered = c @ HE_STAIN_MATRIX.T
191    rgb_jittered = 256.0 * (10.0 ** -od_jittered) - 1.0
192    return np.clip(rgb_jittered.reshape(h, w, 3), 0, 255).astype(np.uint8)
```

- **L178–L180** — OD conversion and flattening, identical in form to the Macenko helpers. Same `+1`/`256` convention, so the round trip at L191 is exact.
- **L182** — Deconvolve into two stain concentrations using the **fixed** Ruifrok matrix (not an estimated one — note the contrast with `macenko_apply` L153, which estimates per tile). Using fixed vectors here is deliberate: the augmentation must simulate *stain variation*, so it should perturb concentrations along known biological axes, not along whatever axes the tile happens to suggest.
- **L184** — `alpha` scales each stain channel by an independent uniform draw in `[1−σ, 1+σ]` = `[0.8, 1.2]` at the default `σ = 0.2`. This simulates "this lab's hematoxylin is 20% stronger." `size=2` means the two stains vary independently — you get tiles that are more hematoxylin-heavy but eosin-light, which is exactly the kind of correlated-looking shift seen across scanners.
- **L185** — `beta` shifts each channel by up to ±0.05 in concentration units. Additive shift + multiplicative scale together span a richer variation family than either alone.
- **L187–L188** — Apply the transform and clamp concentrations at 0 (a negative stain concentration is unphysical; without the clip, `10^(-od)` at L191 could exceed 1 and produce >255 RGB values that then get clipped in RGB space instead, which would produce a subtly different and less physical result). Note `c[:, 0]` and `c[:, 1]` are written in place — `c` came from a matmul, so it owns its memory and this is safe.
- **L190–L191** — Recombine with the unperturbed stain matrix and invert the OD transform. Same constants as L161, so the only thing that changed between input and output is the concentration scaling/shifting — no color-space drift is introduced by the augmentation itself.
- **L192** — Clip and cast back to `uint8`. The function returns a valid image that is *always* the same dtype/shape as its input, which is what lets `PatchTransform` splice it in unconditionally at L213.

> **Behavioral note by design:** `hed_stain_jitter` operates on **every** pixel, with no tissue mask (contrast with Macenko's `od_threshold` guard). Pure-white background has `od = 0` → `conc = 0`; after jittering, `c = 0·α + β` — i.e. the *bias* term `β` paints a faint uniform stain onto background regions, up to 0.05 OD. This is a mild but real simulation of a tinted slide background, and it is one reason stain jitter can look different from Macenko normalization on background-heavy classes (`BACK`, `ADI`, `MUC`).

### `PatchTransform` (L195–L226)

```python
195  class PatchTransform:
196    """Unified transform executing optional normalization, augmentation, and tensor conversion."""
197
198    def __init__(self, policy: str = "none", norm_fn: Any = None, is_train: bool = True):
199      self.policy = policy
200      self.norm_fn = norm_fn
201      self.is_train = is_train
202
204      import torchvision.transforms.functional as TF
205
206      arr = np.array(img_pil.convert("RGB"))
207      if self.norm_fn is not None:
208        arr = self.norm_fn(arr)
209
210      if self.is_train and self.policy in ("aug_stain", "aug_combined"):
211        if random.random() > 0.2:
212          arr = hed_stain_jitter(arr)
213
214      tensor = TF.to_tensor(arr)  # scales [0, 255] -> [0.0, 1.0]
215
216      if self.is_train and self.policy in ("aug_geo", "aug_combined"):
217        if random.random() > 0.5:
218          tensor = TF.hflip(tensor)
219        if random.random() > 0.5:
220          tensor = TF.vflip(tensor)
221        rot = random.choice([0, 90, 180, 270])
222        if rot > 0:
223          tensor = TF.rotate(tensor, rot)
224
225      return TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
```

- **L198–L201** — Three pieces of state. `policy` is one of the `aug` strings from `EXPERIMENTS`; `norm_fn` is a closure (or `None`); `is_train` gates *both* augmentation blocks. There is no separate "eval transform" class — evaluation is simply `PatchTransform(policy="none", is_train=False)` (L443–L445).
- **L204** — `torchvision.transforms.functional` imported inside `__call__`. This runs on every tile access (Python caches the module, so it is a dict lookup, not a re-import), but it keeps `import run_experiments` free of torch — the same lazy-import pattern as `build_model`.
- **L206** — `img_pil.convert("RGB")` forces a decode and normalizes grayscale/CMYK/RGBA inputs to 3 channels. This is also what actually **loads the file** from disk — `Image.open` at L239 is lazy and returns before any pixel data is read. So the I/O cost is paid here, inside `__getitem__`, which is why the DataLoader worker processes matter (L447).
- **L207–L208** — Normalization runs **first**, before any augmentation, on the full `uint8` image. Order is not arbitrary: both normalizers estimate their statistics from the *unstained-variation-free* image, and running them before jitter keeps normalization deterministic given a tile. (Under `aug_combined`, stain jitter after normalization means the model sees tiles that are normalized *and then* perturbed — the "defended" configuration.)
- **L210–L212** — Stain jitter applied with probability **0.8** (`random.random() > 0.2`). Randomness is per-tile-per-access, so each epoch sees a different jitter of the same tile — effectively 8× more distinct stain variations per training image than a one-time offline augmentation would give.
- **L214** — `TF.to_tensor` converts `(H, W, 3) uint8 → (3, H, W) float32` in `[0, 1]` (CHW layout, the layout every torchvision model expects). This must happen *after* the NumPy-space operations, which all assume HWC.
- **L216–L223** — Geometric augmentation, applied on the tensor (cheaper than on arrays, and torchvision's functional ops are exactly designed for this). The three operations are independent:
  - h-flip with p = 0.5,
  - v-flip with p = 0.5,
  - one of `{0°, 90°, 180°, 270°}` with uniform probability 0.25 each, applied only if non-zero.
  - **P(no geometric change at all) = 0.5 × 0.5 × 0.25 = 0.0625**, so 93.75% of sampled tiles get at least one transform. Worth knowing when reasoning about how strong `aug_geo` actually is.
  - `TF.rotate` uses `expand=False` by default, so the output keeps the input size. For **square** tiles rotated by exact multiples of 90°, the rotated grid maps onto itself exactly — no interpolation blur and no filled corners. (This reasoning would **not** hold if the patches were non-square, or if arbitrary angles were sampled: you would then get black/zero-filled corners that, after L226's normalization, become a strong non-neutral artifact.)
- **L225** — ImageNet normalization, **always last**, on all paths (train and eval, all policies). Because it happens after the augmentation block, the geometry ops operate on `[0,1]` data where flips/rotations are trivial; and because it's unconditional, no code path can accidentally feed un-normalized tensors to a pretrained backbone.

**Ordering summary (the single most useful thing to remember about this class):**

```
uint8 HWC ──norm_fn──► uint8 HWC ──HED jitter p=.8──► uint8 HWC ──to_tensor──► float CHW [0,1]
          ──flips/rot──► float CHW [0,1] ──ImageNet normalize──► float CHW normalized
```

Each stage requires the previous stage's format. Moving any stage would either crash (wrong dtype) or silently produce wrong inputs (normalizing before augmenting, or augmenting in normalized space).

---

## A.5 Dataset, splits, and the reference tile (L226–L351)

### `HistologyDataset` (L229–L240)

```python
229  class HistologyDataset:
230    def __init__(self, df: pd.DataFrame, transform: PatchTransform):
231      self.df = df.reset_index(drop=True)
232      self.transform = transform
233
234    def __len__(self) -> int:
235      return len(self.df)
236
237    def __getitem__(self, idx: int):
238      row = self.df.iloc[idx]
239      img = Image.open(row["image_path"])
240      return self.transform(img), int(row["label_idx"])
```

- **L231** — `reset_index(drop=True)` guarantees `iloc[idx]` and positional indexing agree. This matters because the DataFrames are the output of `train_test_split` and `groupby().apply()`, both of which leave non-contiguous indices behind. Without this, `self.df.iloc[idx]` would still work (it's positional), but `df.loc`-style access elsewhere would not — the reset removes the footgun entirely.
- **L234–L235** — Required by `DataLoader` to size the sampler. With `shuffle=True` (L448), this is what defines the epoch length.
- **L237–L240** — `Image.open` is **lazy**: it reads the header and returns immediately; pixels are decoded on first access (which is `img.convert("RGB")` at L207). `int(row["label_idx"])` casts from the CSV's `numpy.int64`/`int64` back to a Python `int` — required because PyTorch's default collate converts targets to tensors and is picky about numpy scalars in worker processes.
- **Note**: no image caching, no on-the-fly resizing. The NCT-CRC tiles are already 224×224, so no resize step is needed anywhere in the pipeline — this is an assumption baked in silently. Feeding 512×512 tiles would change the effective receptive-field scale of every model.

### `find_class_images` (L243–L282)

```python
243  SUPPORTED_IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
244
245
246  def find_class_images(base_dir: Path, class_name: str) -> List[Path]:
247    """Locate images for a histological class, handling case variations and nesting."""
248    if not base_dir.exists() or not base_dir.is_dir():
249      return []
250
251    c_folder = None
252    # 1. Direct child match (exact or case-insensitive)
253    if (base_dir / class_name).is_dir():
254      c_folder = base_dir / class_name
255    else:
256      for child in base_dir.iterdir():
257        if child.is_dir() and child.name.upper() == class_name.upper():
258          c_folder = child
259          break
260
261    # 2. Search recursively across subdirectories if not found directly
262    if c_folder is None:
263      for child in base_dir.rglob("*"):
264        if child.is_dir() and child.name.upper() == class_name.upper():
265          c_folder = child
266          break
267
268    if c_folder is None or not c_folder.is_dir():
269      return []
270
271    # Collect all image files with supported extensions (case-insensitive)
272    images = [ ... ]
...
282    return sorted(images)
```

- **L243** — Module-level set of accepted extensions. A set (not a list) because it's only ever used for `in` membership tests at L274/L280.
- **L248–L249** — Defensive early return for a nonexistent base directory. Callers rely on getting `[]` rather than an exception so that `prepare_dataset_splits` can produce its own *informative* error message (L303–L307) instead of a raw `FileNotFoundError`.
- **L253–L254** — Fast path: the expected layout (`<root>/<CLASS>/`) with the exact name.
- **L255–L259** — Case-insensitive fallback. Necessary because Kaggle's dataset extraction and some archive tools alter case (`adi/` vs `ADI/`), and because `CLASSES` is uppercase. The `break` stops at the first match — if two folders differ only by case, the winner depends on filesystem iteration order.
- **L261–L266** — Recursive fallback for archives that nest an extra directory level (`NCT-CRC-HE-100K-NONORM/NCT-CRC-HE-100K-NONORM/ADI/`). `rglob("*")` walks the whole tree, which can be slow on a 100k-file dataset — but it only runs when the direct match failed, and the first hit breaks out.
- **L268–L269** — Second defensive return.
- **L272–L275** — Non-recursive listing of the class folder, filtered by extension. `p.suffix.lower()` gives case-insensitive matching (`.TIF` works).
- **L277–L281** — Recursive fallback if the direct listing found nothing, for the case where patches live one level deeper inside the class folder. `sorted()` at L282 makes the returned order **deterministic** — important because `prepare_dataset_splits` takes `train_df.iloc[0]` as the reference tile (L344) and because the split itself must be reproducible across machines with different filesystem ordering.

### `prepare_dataset_splits` (L285–L351)

This is the function that guarantees every experiment compares like with like. It runs **once per invocation** and its output is reused by all 13 cells (L547–L552).

```python
285  def prepare_dataset_splits(source_dir, target_dir, out_dir, subset_size=25000, seed=42):
288    """Generate 70/15/15 stratified source splits and pick canonical reference tile."""
289    out_dir.mkdir(parents=True, exist_ok=True)
290    np.random.seed(seed)
291    random.seed(seed)
```

- **L285–L287** — `source_dir` = NCT-CRC-HE-100K-NONORM, `target_dir` = CRC-VAL-HE-7K, `out_dir` = `<out-dir>/splits`, `subset_size=25000`, `seed=42`.
- **L289** — Creates `results/splits/` (with parents) before anything else, so all four `to_csv` writes at L340 are guaranteed to have a target.
- **L290–L291** — Seeds **both** global RNGs. NumPy's is what `groupby().sample(random_state=seed)` and sklearn's `train_test_split` use; Python's `random` is what the augmentations use (L212, L218, L220, L222) — and since it is never reseeded per experiment, the augmentation stream differs across the 13 experiments, which is fine (independent draws) but is *not* reproducible across runs (see [§ C.2](#c2-reproducibility-and-determinism)).

```python
293    # Collect source domain patches
294    source_records = []
295    for c_idx, c_name in enumerate(CLASSES):
296      c_images = find_class_images(source_dir, c_name)
297      for p in c_images:
298        source_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "source"})
299
300    source_df = pd.DataFrame(source_records)
301    if len(source_df) == 0:
302      items = [p.name for p in list(source_dir.iterdir())[:15]] if source_dir.is_dir() else "directory does not exist"
303      raise RuntimeError(
304        f"No source images found under {source_dir}.\n"
305        f"Contents found at {source_dir}: {items}.\n"
306        f"Expected 9 class folders {CLASSES} containing {sorted(SUPPORTED_IMAGE_EXTS)} images."
307      )
```

- **L295–L298** — Build one record per image, iterating in `CLASSES` order so `label_idx` is the canonical class index. `str(p)` (not `p`) because the path is written to CSV at L340 — storing `Path` objects would work in-memory but serialize inconsistently.
- **L301–L307** — The empty-input guard, and the most valuable error message in the file. It reports (a) which path failed, (b) up to 15 entries actually found there, and (c) what was expected. On Kaggle this is the error you hit when the dataset is attached under a different slug or nested one level deeper than expected — the "contents found" list is usually enough to fix the path without further debugging. Note the guard at L302 handles the case where the directory itself doesn't exist, printing `"directory does not exist"` instead of crashing inside `iterdir()`.

```python
309    # Stratified subset if requested
310    if 0 < subset_size < len(source_df):
311      per_class = subset_size // NUM_CLASSES
312      source_df = source_df.groupby("class_name", group_keys=False).apply(
313        lambda g: g.sample(min(len(g), per_class), random_state=seed)
314      ).reset_index(drop=True)
```

- **L310** — Subsetting only when it actually shrinks the data. `subset_size=0` (the documented "full 100k" flag) fails the first condition and silently means "use everything" — a slightly implicit convention; the CLI help text at L521 is the only place this is stated.
- **L311** — `25000 // 9 = 2777` per class, so the subset is **24,993** tiles, not 25,000. Floor division means the last 7 tiles of the budget are discarded rather than distributed. This number propagates to every split size printed at L349.
- **L312–L314** — Per-class uniform subsampling via `groupby(...).apply(...)` with `group_keys=False` (which prevents pandas from inserting the group key as an extra index level) and an explicit `reset_index(drop=True)` to restore a clean RangeIndex. `min(len(g), per_class)` protects the small classes — `DEB` and `LYM` are the rarest in NCT-CRC, so if `per_class` ever exceeded a class's population the `sample` would still be valid. Note: on pandas ≥ 2.2 this `apply` emits a `DeprecationWarning` about operating on grouping columns; it is harmless (behavior is what's intended) but will appear in Kaggle logs.

```python
316    # Stratified 70 / 15 / 15 split
317    from sklearn.model_selection import train_test_split
318    train_df, rest_df = train_test_split(source_df, test_size=0.30, stratify=source_df["class_name"], random_state=seed)
319    val_df, test_id_df = train_test_split(rest_df, test_size=0.50, stratify=rest_df["class_name"], random_state=seed)
```

- **L317** — sklearn imported here (lazily), not at module scope.
- **L318–L319** — Two-stage split achieving 70/15/15: take 30% off the top, then halve the remainder. The alternative (three-way `train_test_split` with two `test_size` values) gives the same ratios but is harder to read. `stratify=` preserves per-class proportions at both stages, which is essential given the class imbalance the plan calls out (`ADI` ≈ 1,338 vs `DEB` ≈ 339 in the target set). `random_state=seed` makes the split deterministic.
- **With the default subset**, the arithmetic works out to: 24,993 total → `rest` = `ceil(24993 × 0.30)` = 7,498 → train = 17,495; then val = 3,749 and test_id = 3,749. Training therefore uses **~17.5k tiles**, not the full 100k, unless `--subset 0` is passed. Every reported number in `docs/RESULTS.md` must be interpreted against this.
- **`stratify` failure mode**: if any class has fewer than 2 members, sklearn raises `ValueError: The least populated class in y has only 1 member`. The default subset makes this essentially impossible; a tiny `--subset` (e.g. `--subset 9`) would trigger it.

```python
321    # Collect target domain (100% out-of-domain)
322    target_records = []
323    for c_idx, c_name in enumerate(CLASSES):
324      c_images = find_class_images(target_dir, c_name)
325      for p in c_images:
326        target_records.append({"image_path": str(p), "class_name": c_name, "label_idx": c_idx, "domain": "target"})
327
328    test_ood_df = pd.DataFrame(target_records)
329    if len(test_ood_df) == 0:
330      ...raise RuntimeError(...)
```

- **L321–L326** — Same collection logic as the source, but for the target domain. The only difference is the `"domain"` value.
- **L328** — The entire target set becomes `test_ood`. There is **no** split here — no target validation, no target tuning set. This is the structural enforcement of the domain firewall: because no other variable in the program ever references `target_dir`, there is no code path by which target data could leak into training.
- **L329–L335** — Duplicate of the L301–L307 guard with target-specific wording. (A candidate for factoring into a shared helper if this file is ever refactored; the duplication is tolerated to keep each error message tightly worded.)

```python
337    # Save CSVs
338    splits = {"train": train_df, "val_id": val_df, "test_id": test_id_df, "test_ood": test_ood_df}
339    for name, df in splits.items():
340      df.to_csv(out_dir / f"{name}.csv", index=False)
341
342    # Pick reference tile strictly from training split (dense colorectal tumor tile)
343    tum_train = train_df[train_df["class_name"] == "TUM"]
344    ref_path = Path(tum_train.iloc[0]["image_path"])
345    ref_img = Image.open(ref_path).convert("RGB")
346    ref_save_path = out_dir / "reference_stain.png"
347    ref_img.save(ref_save_path)
348
349    print(f"[splits] Train={len(train_df)} | Val={len(val_df)} | Test-ID={len(test_id_df)} | Test-OOD={len(test_ood_df)}")
350    print(f"[reference] Canonical stain reference: {ref_path.name}")
351    return splits, ref_save_path
```

- **L338–L340** — The four DataFrames are written as `train.csv`, `val_id.csv`, `test_id.csv`, `test_ood.csv` under `results/splits/`. These are **the provenance record of every run** — since the split is seeded and deterministic, these CSVs let you verify after the fact exactly which tiles were in which split. `index=False` keeps the files diff-friendly.
- **L343–L344** — Reference tile selection: the first `TUM` row of the **training** split. Three deliberate constraints are encoded in these two lines:
  1. **TUM only** — a densely stained tumor tile gives the stain estimators strong, well-separated hematoxylin/eosin signal. A background tile would produce a degenerate reference.
  2. **Training split only** — never the test sets, so the reference carries no evaluation information.
  3. **Deterministic** — `iloc[0]` of a deterministically-ordered frame, so all 13 experiments and all re-runs use the *identical* reference tile. This is what makes cross-experiment comparison valid.
- **L345–L347** — The tile is copied to `results/splits/reference_stain.png`, so the reference is preserved as an artifact. This matters for reproducibility: the 10 numbers from `macenko_fit` are not serialized, so inspection/re-derivation requires the image.
- **L349–L350** — The two-line provenance summary you should screenshot for the report.
- **L351** — Returns the four DataFrames (used by every experiment) *and* the reference path (used to build `norm_fn`).

---

## A.6 Model construction (L352–L380)

```python
355  def build_model(backbone_name: str, num_classes: int = NUM_CLASSES):
356    import timm
357    import torch.nn as nn
358
359    if backbone_name == "phikon":
360      from transformers import AutoModel
361      class PhikonClassifier(nn.Module):
362        def __init__(self):
363          super().__init__()
364          self.encoder = AutoModel.from_pretrained("owkin/phikon")
365          # Linear probing: freeze ViT backbone
366          for p in self.encoder.parameters():
367            p.requires_grad = False
368          self.fc = nn.Linear(768, num_classes)
369
370        def forward(self, x):
371          feat = self.encoder(x).last_hidden_state[:, 0]
372          return self.fc(feat)
373
374      return PhikonClassifier()
375
376    if backbone_name == "convnext_tiny":
377      return timm.create_model("convnext_tiny.fb_in1k", pretrained=True, num_classes=num_classes)
378
379    # Default: ResNet-50
380    return timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=num_classes)
```

- **L355** — `num_classes` defaults to the module-level `NUM_CLASSES`, so the heads match the label space by construction.
- **L356–L357** — Lazy `timm`/`torch.nn` imports, consistent with the file's pattern. Note `torch.nn` is imported even on the Phikon path where it's used for `nn.Linear`/`nn.Module`, so both imports are live on all paths.
- **L359–L374 — the Phikon branch.** This is the only backbone that needs custom code, for three reasons:
  - **L361–L362** — A locally-defined `nn.Module` subclass. `super().__init__()` at L363 is mandatory before assigning any submodule in PyTorch.
  - **L364** — `AutoModel.from_pretrained("owkin/phikon")` downloads/loads the ViT-B/16 pathology foundation model. This is a **network call on first run** — the reason Kaggle's Internet toggle must be `On`, and a common cause of apparently-hanging runs.
  - **L365–L367** — Freeze every encoder parameter. `requires_grad = False` means no gradients and — critically — no AdamW state (momentum/variance buffers) allocated for 86M parameters, which is what keeps this variant's memory footprint close to the other backbones. Note the assignment is `p.requires_grad = False` on the parameter object, which works because `requires_grad` is a mutable attribute of `Parameter`.
  - **L368** — A single `Linear(768, 9)`. 768 = ViT-B hidden size; hardcoded rather than read from `config.hidden_size`, which is safe for Phikon specifically but would break if the model ID were swapped.
  - **L371** — `last_hidden_state[:, 0]` is the **CLS token** — index 0 of the sequence dimension, which for ViT is the classification token, *not* a spatial patch. The full hidden-state tensor `(B, 197, 768)` is materialized and immediately sliced, so only the CLS vector reaches the head. `.last_hidden_state` (rather than `.pooler_output`) is the standard Phikon usage.
  - **Effective method: linear probing.** Only 9×768 + 9 = 6,921 parameters are trainable out of ~86M. This makes EXP-12/13 cheap per step but limits how much the defenses can affect the representation — a model whose features are frozen cannot learn stain invariance, it can only be *given* it by normalization.
  - `import torch.nn as nn` at L357 is what makes `nn.Module` available inside this nested class body.
- **L376–L377** — ConvNeXt-Tiny with `fb_in1k` pretrained weights (the original Facebook/ImageNet-1k recipe). Modern ConvNet, ~28M parameters, 224×224 native. The `num_classes=num_classes` argument replaces timm's default 1000-way head, and timm **re-initializes** the new head — so the backbone is pretrained but the classifier is not.
- **L379–L380** — The default branch, which is also the fallthrough for **any unrecognized `backbone` string** in `EXPERIMENTS`. `resnet50.a1_in1k` is timm's ResNet-50 trained with the "A1" recipe ("ResNet strikes back"), a stronger baseline than the classic torchvision weights — appropriate, since ResNet-50 is the study's *anchor*. Same head-replacement behavior as L377.
- **Design note**: because of this fallthrough, a typo in `EXPERIMENTS` (e.g. `"resnet-50"`) silently produces a ResNet-50 run labeled as something else. There is no validation of the `EXPERIMENTS` table anywhere — worth adding a startup assertion if the matrix grows.

---

## A.7 Evaluation and the training loop (L381–L512)

### `evaluate_model` (L384–L413)

```python
384  def evaluate_model(model, loader, device) -> Dict[str, float]:
385    import torch
386    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
387
388    model.eval()
389    all_preds, all_probs, all_targets = [], [], []
390
391    with torch.no_grad():
392      for images, targets in loader:
393        images = images.to(device, non_blocking=True)
394        with torch.amp.autocast("cuda"):
395          logits = model(images)
396        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
397        all_probs.append(probs)
398        all_preds.append(np.argmax(probs, axis=1))
399        all_targets.append(targets.numpy())
```

- **L384** — One function used for all three roles: per-epoch validation (L478), test-ID (L489), and test-OOD (L490). Identical measurement code everywhere is what makes val→test comparisons meaningful.
- **L388** — `model.eval()` switches BatchNorm to running statistics and disables dropout. **Essential and easy to forget** — in train mode a ResNet-50 evaluated on the test set would give systematically different (usually worse) numbers, and the size of the discrepancy would depend on batch composition.
- **L389** — Accumulators. All three are collected as *lists of per-batch arrays* and concatenated once at the end (L401–L403) — far cheaper than repeated `np.concatenate` in the loop, and it avoids quadratic copying.
- **L391** — `torch.no_grad()` disables autograd graph construction. Without it, evaluating 7,180 OOD tiles would build (and retain) a full backward graph for every batch, roughly tripling memory. Note this is *separate* from `model.eval()` — they do different things and both are needed.
- **L392** — Iterates the loader; `images` arrives as `(B, 3, 224, 224)` float, `targets` as `(B,)` int64 (the collate function stacks the per-sample tensors).
- **L393** — `non_blocking=True` allows the host→device copy to overlap with compute when `pin_memory=True` was used on the loader (L448–L449). It's a no-op for the loaders built without `pin_memory` (L450–L451).
- **L394–L395** — Forward pass under CUDA autocast: runs in fp16 where safe. Note there is **no `GradScaler` here** — scaling is only needed to protect backward passes, and there is no backward pass. Wrapping eval in autocast keeps the numerics identical to what the training forward pass produced.
- **L396** — `.float()` before softmax is the important detail: `logits` may be fp16, and `softmax` over fp16 has enough precision loss to change borderline `argmax` decisions and to distort AUROC (which consumes the probabilities directly). Casting to fp32 first costs one temporary and removes the concern. `torch.softmax` (rather than manual exp/sum) uses the numerically stable max-subtracted formulation.
- **L398** — `argmax` over the probability columns gives the predicted label in the same index space as `CLASSES` (because `label_idx` was assigned from `enumerate(CLASSES)`).
- **L399** — `targets.numpy()` works directly on the CPU tensor the loader produced.

```python
401    y_true = np.concatenate(all_targets)
402    y_pred = np.concatenate(all_preds)
403    y_prob = np.concatenate(all_probs)
404
405    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
406    bal_acc = balanced_accuracy_score(y_true, y_pred)
407    acc = accuracy_score(y_true, y_pred)
408    try:
409      auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
410    except Exception:
411      auroc = float("nan")
412
413    return {"macro_f1": ..., "balanced_acc": ..., "accuracy": ..., "auroc": ...}
```

- **L405** — **Macro F1 is the study's primary metric and the model-selection criterion** (used at L479). Macro averaging weights all 9 classes equally regardless of frequency, which is what makes the imbalanced target set (`ADI` 1,338 vs `DEB` 339) fair. `zero_division=0` prevents a warning and a `nan` when some class has zero predictions.
- **L406** — Balanced accuracy = macro-averaged recall; closely related to macro F1 but insensitive to precision, so it separates "missing classes" from "over-predicting classes" when read alongside F1.
- **L407** — Raw accuracy, reported for completeness but **biased toward majority classes** — exactly the bias the plan warns about. It is the weakest of the four metrics on this dataset.
- **L409** — Macro one-vs-rest AUROC. Wrapped in `try/except` because `roc_auc_score` raises if `y_true` contains a class with no samples — a real possibility if someone runs with a tiny `--subset`, or in ad-hoc evaluation on a filtered frame. On failure the metric becomes `nan` rather than killing a 3.5-hour run.
- **L413** — Returns a plain `Dict[str, float]` with all four metrics.
- **Note**: only `macro_f1` and `accuracy` are propagated into the results table (L503–L508). **`balanced_acc` and `auroc` are computed for every evaluation — val, test-ID, and test-OOD, 10 evaluations per experiment — and then discarded.** If you want them in `summary_results.csv`, they need to be added to the result dict at L496–L510; there is no other place they are stored.

### `train_experiment` (L416–L512)

The heart of the study: one full train/evaluate cycle for one cell of the matrix.

```python
416  def train_experiment(exp, splits, ref_img_path, epochs=8, batch_size=64, lr=1e-3, device_str="cuda"):
426    import torch
427    import torch.nn as nn
428    from torch.utils.data import DataLoader
429
430    device = torch.device(device_str)
431    ref_rgb = np.array(Image.open(ref_img_path).convert("RGB"))
432
433    # Setup normalizer
434    norm_fn = None
435    if exp["norm"] == "reinhard":
436      ref_stats = reinhard_fit(ref_rgb)
437      norm_fn = lambda img: reinhard_apply(img, ref_stats)
438    elif exp["norm"] == "macenko":
439      ref_params = macenko_fit(ref_rgb)
440      norm_fn = lambda img: macenko_apply(img, ref_params)
```

- **L416–L424** — Signature. `exp` is one dict from `EXPERIMENTS`; `splits` is the dict returned by `prepare_dataset_splits`; defaults match the CLI defaults so the function is callable from a notebook.
- **L430** — `device_str` is converted once. `torch.device("cpu")` and `torch.device("cuda")` then travel down to every `.to()` call.
- **L431** — The reference tile is loaded **fresh for every experiment**. Cost is negligible (one small PNG) and it keeps the function self-contained — no hidden dependency on a previously-built normalizer.
- **L433–L440** — The normalization dispatch, and a subtle but important design point: `reinhard_fit` / `macenko_fit` are called **once per experiment**, not per tile and not per epoch. The fitted statistics are captured in the closure (`ref_stats` / `ref_params`) and reused. `macenko_apply` does re-estimate *per tile*, but the reference is fixed.
  - **L434** — `norm_fn = None` is the `"none"` case *and* the fallthrough for any unrecognized `norm` value.
  - **L437 / L440** — Lambdas rather than `functools.partial`, purely for readability at the call site. Both capture the fitted dict by closure.

```python
442    train_ds = HistologyDataset(splits["train"], PatchTransform(policy=exp["aug"], norm_fn=norm_fn, is_train=True))
443    val_ds = HistologyDataset(splits["val_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
444    test_id_ds = HistologyDataset(splits["test_id"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
445    test_ood_ds = HistologyDataset(splits["test_ood"], PatchTransform(policy="none", norm_fn=norm_fn, is_train=False))
```

- **L442** — The **only** dataset that gets the experiment's augmentation policy and `is_train=True`.
- **L443–L445** — All three evaluation sets use `policy="none"` and `is_train=False`. This is the correct experimental design: augmentation is a *training-time* intervention, and evaluating on augmented data would confound the comparison (a model trained with rotations would face rotated test tiles that others never see). It also means **every** model is evaluated on identically-transformed tiles — only `norm_fn` differs between experiments.
- Note `is_train=False` is technically redundant for these three (since `policy="none"` already disables both augmentation blocks), but it is passed explicitly as a belt-and-braces statement of intent.

```python
447    num_workers = 4 if os.name != "nt" else 0
448    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
449    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
450    test_id_loader = DataLoader(test_id_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
451    test_ood_loader = DataLoader(test_ood_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

- **L447** — The Windows check. Windows uses `spawn` for multiprocessing, which re-imports the module in every worker; with the default `num_workers>0` that would be slow or (in a notebook/interactive context without an `if __name__ == "__main__"` guard on the *caller*) can recurse catastrophically. `0` means everything runs in the main process. On Kaggle/Linux this evaluates to **4**, so tiles are decoded in parallel across 4 processes.
- **L448** — `shuffle=True` on train only. The shuffle order comes from torch's global RNG, which is **never seeded** — one of the reasons two runs of the same experiment differ slightly (see [§ C.2](#c2-reproducibility-and-determinism)). `pin_memory=True` speeds up the host→device copy for the training loop, where the GPU is actually the bottleneck.
- **L449–L451** — `shuffle=False` everywhere else, so evaluation is deterministic *given weights*, and `pin_memory` omitted on the two test loaders (a minor inconsistency: they'd benefit from it too; the omission is harmless but arbitrary).
- **Important interaction**: because `num_workers=4` on Linux, the *augmentation randomness* (Python `random` at L212/L218) is drawn in worker processes with torch-seeded-per-worker state derived from an unseeded base — so the augmentation stream is not reproducible run to run even with identical splits.

```python
453    model = build_model(exp["backbone"]).to(device)
454    optimizer = torch.optim.AdamW(model.parameters(), lr=lr if exp["backbone"] != "phikon" else 3e-4, weight_decay=0.05)
455    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)
456    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
457    scaler = torch.amp.GradScaler("cuda")
```

- **L453** — Model built and moved to the device. For Phikon this triggers the HuggingFace download on first use; subsequent experiments reuse the local cache, which is why EXP-12 and EXP-13 start faster than EXP-12's first epoch.
- **L454** — AdamW with `lr=1e-3` for the CNNs, `3e-4` for Phikon (a lower rate for a frozen-backbone linear probe), and `weight_decay=0.05` across the board. Note `model.parameters()` includes the frozen Phikon encoder — those parameters still appear in the optimizer's param groups but receive no gradient (and AdamW applies no decay where there is no gradient), so the practical effect is that the optimizer contains ~86M inert entries.
- **L455** — Cosine annealing stepped **per batch** (`T_max = epochs × steps_per_epoch`), decaying to `eta_min=1e-5`. With the default numbers: 17,495 tiles ÷ 64 ≈ **274 steps/epoch** → `T_max ≈ 2,192` scheduler steps for an 8-epoch run. Stepping per batch (rather than per epoch) is what gives the smooth curve usually seen in modern recipes. The `scheduler.step()` at L475 must come *after* the optimizer step, or PyTorch ≥1.13 emits a "step before optimizer.step" warning — the ordering here is correct.
- **L456** — `label_smoothing=0.1` softens the one-hot targets, which counteracts overconfidence on this noisy, high-label-noise task and typically improves macro F1 on imbalanced data. It also means the reported training loss is not comparable to a plain cross-entropy value.
- **L457** — `torch.amp.GradScaler("cuda")` for mixed-precision training. On a CPU-only machine this self-disables with a warning rather than crashing, which is why the script can be smoke-tested locally on CPU.

```python
459    best_val_f1 = -1.0
460    best_state = None
461    t0 = time.time()
...
464    for epoch in range(1, epochs + 1):
465      model.train()
466      total_loss = 0.0
467      for images, targets in train_loader:
468        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
469        optimizer.zero_grad(set_to_none=True)
470        with torch.amp.autocast("cuda"):
471          loss = criterion(model(images), targets)
472        scaler.scale(loss).backward()
473        scaler.step(optimizer)
474        scaler.update()
475        scheduler.step()
476        total_loss += loss.item()
```

- **L459** — Initialized to `-1.0` (below any achievable F1, which is ≥ 0) so the first epoch always registers as an improvement.
- **L460** — Holds the best weights in CPU memory. `None` until the first validation.
- **L465** — `model.train()` per epoch — must be re-set every epoch because `evaluate_model` at L478 leaves the model in `eval()` mode.
- **L466 / L476** — `total_loss` accumulates the *batch* loss; the epoch average is computed at L483.
- **L468** — Both tensors moved; `non_blocking=True` pairs with the `pin_memory=True` at L448.
- **L469** — `zero_grad(set_to_none=True)` clears gradients. `set_to_none=True` releases the gradient buffers rather than writing zeros, which is slightly faster and is the modern default.
- **L470–L471** — Autocast forward and loss. The loss is computed **inside** the autocast region (correct: the softmax/cross-entropy should run in the same precision regime as the forward). `criterion` is applied to the raw logits from `model(images)` — CrossEntropyLoss expects logits, not probabilities.
- **L472–L474** — The AMP trio: scale the loss, step the optimizer through the scaler (which skips the step if any gradient is non-finite), then update the scale factor for the next iteration. All three must appear in this order and all three must be present — omitting `scaler.update()` silently freezes the scale at its initial 65536 and produces `inf` gradients on the first bad batch.
- **L475** — Scheduler stepped per batch, after the optimizer. See L455.
- **L476** — `loss.item()` transfers a Python float, detaching it from the graph. Accumulating the tensor itself would retain 274 graphs per epoch.

```python
478      val_res = evaluate_model(model, val_loader, device)
479      if val_res["macro_f1"] > best_val_f1:
480        best_val_f1 = val_res["macro_f1"]
481        best_state = copy.deepcopy(model.state_dict())
482
483      print(f"  ep {epoch}/{epochs} | loss={total_loss / len(train_loader):.4f} | val_f1={val_res['macro_f1']:.4f} (best={best_val_f1:.4f})")
```

- **L478** — Full validation pass after every epoch on the 3,749-tile val split. This is the **only** signal used for model selection — a deliberate choice that keeps test sets untouched.
- **L479–L481** — Strict `>` comparison, so ties keep the earlier checkpoint. `deepcopy` of the `state_dict` snapshots the weights; a shallow copy would alias the live tensors and record the *final* epoch instead of the best. `deepcopy` (rather than manual `{k: v.clone()}`) is the idiomatic form.
  - Memory note: this holds one extra full model in CPU RAM — trivial for ResNet-50/ConvNeXt, larger for Phikon (~340MB in fp32), still fine on Kaggle's ~13GB.
- **L483** — One progress line per epoch. `len(train_loader)` is 274 with the default subset, so `.item()`-based loss averaging is exact (no `drop_last`).

```python
485    # Load best checkpoint for test evaluations
486    if best_state is not None:
487      model.load_state_dict(best_state)
488
489    id_res = evaluate_model(model, test_id_loader, device)
490    ood_res = evaluate_model(model, test_ood_loader, device)
491    trained_min = round((time.time() - t0) / 60.0, 2)
492
493    delta_f1 = round(id_res["macro_f1"] - ood_res["macro_f1"], 4)
494    rr_f1 = round((ood_res["macro_f1"] / max(id_res["macro_f1"], 1e-6)) * 100.0, 2)
```

- **L486–L487** — Restore the best-on-validation weights **before** touching either test set. This is the standard protocol and the reason `best_state` exists at all: reporting the last epoch's weights would make results depend on how many epochs were run, which would make `--epochs` changes incomparable.
- **L489–L490** — The two headline evaluations: test-ID (held-out source tiles) and test-OOD (the firewalled external set). Note both are evaluated from the *same* checkpoint, so the Δ/RR below describe one model, not two.
- **L491** — Wall-clock minutes, rounded to 2 decimals, measured from L461 (i.e. from just before epoch 1, so it *includes* training and validation but *excludes* split preparation and model download — the number in the CSV is per-experiment train time, not total wall time).
- **L493** — `Delta_F1 = F1_ID − F1_OOD` ("Stain Drop"). Lower is better; 0 = perfect stain invariance. This is the study's primary robustness quantity (naming per `docs/PLAN.md` §4.3 and `docs/RESULTS.md`).
- **L494** — `RR_F1 = F1_OOD / F1_ID × 100` ("Retention Rate"). Higher is better; 100% = full retention of in-domain diagnostic power across institutions. The `max(..., 1e-6)` guards against division by zero in the pathological case of an ID F1 of exactly 0 — the only realistic way to reach it is a crashed/degenerate run.

```python
496    result = {
497      "exp_id": exp["id"],
498      "stage": exp["stage"],
499      "backbone": exp["backbone"],
500      "norm": exp["norm"],
501      "aug": exp["aug"],
502      "best_val_f1": round(best_val_f1, 4),
503      "test_id_f1": round(id_res["macro_f1"], 4),
504      "test_ood_f1": round(ood_res["macro_f1"], 4),
505      "delta_f1": delta_f1,
506      "rr_f1": rr_f1,
507      "test_id_acc": round(id_res["accuracy"], 4),
508      "test_ood_acc": round(ood_res["accuracy"], 4),
509      "minutes": trained_min,
510    }
511    print(f"[{exp['id']} Result] ID F1={result['test_id_f1']} | OOD F1={result['test_ood_f1']} | Delta F1={delta_f1} | RR={rr_f1}% ({trained_min}m)")
512    return result
```

- **L497–L501** — The five identity columns, copied verbatim from the experiment dict so the CSV is self-describing (you never need to cross-reference `EXPERIMENTS` to decode a row).
- **L502** — Best validation macro F1. Useful as a sanity check: if `best_val_f1` and `test_id_f1` diverge substantially, something is off with the split or the run.
- **L503–L504** — The two headline F1s (macro, from `evaluate_model` L405).
- **L505–L506** — Δ and RR as computed at L493–L494.
- **L507–L508** — Accuracies, included because they are the most legible numbers for a general audience even though they are the least fair metric here.
- **L509** — Training minutes from L491. Named `minutes` — see the discrepancy note in [§ C.5](#c5-known-doc-vs-code-discrepancies).
- **L511** — The one-line result banner, printed to the piped stdout that the Kaggle notebook streams.

---

## A.8 `main()` — the orchestrator (L513–L587)

```python
516  def main():
517    parser = argparse.ArgumentParser(description="Run 13-cell Histopathology Staining Robustness Ablations")
518    parser.add_argument("--data-root", default="data/raw/NCT-CRC-HE-100K-NONORM", help="Path to in-domain dataset")
519    parser.add_argument("--target-root", default="data/raw/CRC-VAL-HE-7K", help="Path to out-of-domain dataset")
520    parser.add_argument("--out-dir", default="results", help="Directory for output metrics and tables")
521    parser.add_argument("--subset", type=int, default=25000, help="Subset size (default 25000; 0 for full 100k)")
522    parser.add_argument("--epochs", type=int, default=8, help="Epochs per experiment")
523    parser.add_argument("--batch-size", type=int, default=64)
524    parser.add_argument("--experiments", nargs="*", default=None, help="Filter experiment IDs to run (e.g. EXP-01 EXP-02)")
525    parser.add_argument("--dry-run", action="store_true", help="Print experiment plan and exit")
526    args = parser.parse_args()
```

| Flag | Default | Effect | Notes |
| :--- | :--- | :--- | :--- |
| `--data-root` | `data/raw/NCT-CRC-HE-100K-NONORM` | Source of the 70/15/15 splits | Local default; Kaggle overrides with `/kaggle/input/...` |
| `--target-root` | `data/raw/CRC-VAL-HE-7K` | The OOD test set | Read exactly once, at split time |
| `--out-dir` | `results` | Parent of `splits/`, `summary_results.csv`, `RESULTS_TABLE.md` | Created at L529 |
| `--subset` | `25000` | Per-class subsample cap | **`0` = full 100k** (see L310) |
| `--epochs` | `8` | Epochs per experiment | Multiplied into `T_max` at L455 |
| `--batch-size` | `64` | All four loaders | 64 × 224² × 3 fp32 ≈ 38MB per batch |
| `--experiments` | `None` | Filter by ID | See the gotcha below |
| `--dry-run` | `False` | Print the plan and exit before importing torch | The < 1s local verification path |

- **L517** — Program description; appears in `--help`.
- **L521** — The `0 = full dataset` convention is documented *only* here and at L310's condition; there's no validation that a negative value is rejected (a negative `--subset` fails L310's first condition and therefore silently means "full dataset" too).
- **L524** — `nargs="*"` collects zero or more IDs after the flag.
  - **Gotcha**: because `if args.experiments:` at L532 tests *truthiness*, passing `--experiments` with **no** values yields `[]`, which is falsy → **all 13 experiments run**. There is no way to select zero experiments, and the failure is silent (it looks like you asked for nothing and got everything).
  - **Gotcha 2**: `nargs="*"` is greedy — `--experiments EXP-01 EXP-02` consumes both, but a following positional-looking token would also be swallowed. With only flags in this CLI that's fine in practice.

```python
528    out_path = Path(args.out_dir)
529    out_path.mkdir(parents=True, exist_ok=True)
530
531    exp_list = EXPERIMENTS
532    if args.experiments:
533      wanted = set(args.experiments)
534      exp_list = [e for e in exp_list if e["id"] in wanted]
535
536    if args.dry_run:
537      print(f"Plan: {len(exp_list)} experiments queued (subset={args.subset}, epochs={args.epochs}, batch={args.batch_size})")
538      for e in exp_list:
539        print(f"  {e['id']} | {e['stage']} | backbone={e['backbone']:<13} | norm={e['norm']:<8} | aug={e['aug']:<12} | {e['notes']}")
540      return
```

- **L528–L529** — Output directory created up front, so a failure later still leaves a directory to inspect.
- **L531–L534** — Filtering preserves the **original matrix order** (list comprehension over `EXPERIMENTS`, not over the requested set). This means `--experiments EXP-13 EXP-01` still runs EXP-01 first — the results CSV is always in canonic order regardless of the command line. Unknown IDs are silently ignored (a typo like `EXP-1` yields an empty `exp_list` and, since `--dry-run` is the only thing that would show it, a *silently zero-experiment run* that still rewrites `summary_results.csv` with an empty table — worth knowing before you trust that file).
- **L536–L540** — The dry run. Because this returns **before** L542 (`import torch`), the plan is printable on a machine with no PyTorch installed. The `<13`/`<8`/`<12` format specs left-align and pad every column, which is what makes the output readable as a table in a terminal. `--dry-run` is the fastest way to verify that an edit to `EXPERIMENTS` is well-formed.

```python
542    import torch
543    device = "cuda" if torch.cuda.is_available() else "cpu"
544    print(f"Running on device: {device} | Total experiments: {len(exp_list)}")
545
546    splits_dir = out_path / "splits"
547    splits, ref_tile = prepare_dataset_splits(
548      source_dir=Path(args.data_root),
549      target_dir=Path(args.target_root),
550      out_dir=splits_dir,
551      subset_size=args.subset,
552    )
```

- **L542–L544** — Device selection with a *silent* CPU fallback. On Kaggle, forgetting to enable the GPU accelerator does **not** error — it prints `Running on device: cpu` and proceeds to train 13 experiments on CPU, which would take days rather than hours. Always check this line in the first minute of a run.
- **L547–L552** — Splits built **once**, before the experiment loop. Two consequences: (1) all 13 cells share byte-identical splits, which is the core fairness property of the study; (2) the ~24,993-image directory walk happens once rather than 13 times.
  - Note `subset_size=args.subset` is passed through positionally-by-keyword; `seed` is left at its default of 42 hardcoded in the function signature. There is **no `--seed` CLI flag** — reproducibility of the split is not user-configurable.

```python
554    results = []
555    for exp in exp_list:
556      res = train_experiment(
557        exp=exp,
558        splits=splits,
559        ref_img_path=ref_tile,
560        epochs=args.epochs,
561        batch_size=args.batch_size,
562        device_str=device,
563      )
564      results.append(res)
```

- **L554–L564** — The main loop: **strictly sequential**, one experiment at a time. No parallelism, no checkpointing of intermediate results — if the run dies at EXP-11, the first 10 results are lost (they were only in `results`, which is written at L571 *after* the loop). This is the single biggest operational fragility in the file: a 3.5-hour Kaggle run has no crash recovery. If you need it, write `res` to a per-experiment CSV inside the loop.
- **L559** — The reference tile path returned from split preparation is passed to every experiment, guaranteeing they all normalize to the same target.
- Note `lr` is not exposed on the CLI and is left at the function default `1e-3` (with the Phikon 3e-4 override at L454).

```python
567    res_df = pd.DataFrame(results)
568    csv_file = out_path / "summary_results.csv"
569    md_file = out_path / "RESULTS_TABLE.md"
570
571    res_df.to_csv(csv_file, index=False)
572    try:
573      md_content = res_df.to_markdown(index=False)
574    except Exception:
575      md_content = res_df.to_string(index=False)
576
577    md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")
578
579    print("\n" + "=" * 80)
580    print("FINAL ABLATION RESULTS TABLE")
581    print("=" * 80)
582    print(md_content)
583    print(f"\n[Artifacts] Wrote {csv_file} and {md_file}")
```

- **L567** — Column order in the CSV follows the insertion order of the `result` dict (Python 3.7+), i.e. identity columns first, then metrics — so the CSV is readable left to right.
- **L571** — `summary_results.csv` is the machine-readable artifact; `index=False` avoids a spurious unnamed index column.
- **L572–L575** — `to_markdown` requires the **`tabulate`** package, which is not a hard dependency. The fallback to `to_string` means the pipeline still completes on a machine without it (the notebook's Cell 1 installs `tabulate` precisely so this path produces a real Markdown table).
- **L577** — Writes `RESULTS_TABLE.md` with a `# Final Ablation Results` heading, ready to paste into `docs/RESULTS.md` or the report. Explicit `encoding="utf-8"` — relevant on Windows, where the default locale encoding is not UTF-8 and the `Δ`-adjacent characters in a report could otherwise mangle.
- **L579–L583** — The end-of-run banner and artifact paths. Since the script has no logging framework, this print block is the only run summary; on Kaggle it arrives at the very end of the piped stream.

---

# Part B — `notebook.ipynb`

Five cells, no logic of its own: everything scientific lives in `run_experiments.py`. The notebook's job is to satisfy Kaggle's environment contract (packages, paths, GPU) and then shell out.

### Cell 0 — Markdown header (cell id `be40732a`)

Describes the study and, more practically, the **exact Kaggle configuration the code assumes**:

- **Accelerator: GPU P100** (or T4 ×2). Note `docs/kaggle_guide.md` §3 says `GPU T4 x2` while this cell says P100 preferred — the two documents disagree; either works, but the expected ~3.5h runtime was measured on one of them.
- **Internet: On.** Required for three downloads that happen at runtime: the ImageNet weights inside `timm.create_model(..., pretrained=True)`, the `owkin/phikon` weights via `AutoModel.from_pretrained`, and the `pip install` in Cell 1. With Internet off, the run fails at model construction — *after* the splits have been prepared.
- **Three attached datasets.** The names here must match the *slugs* used in Cell 2's hardcoded paths: `histo-robust-code`, `nct-crc-he-100k-nonorm`, `crc-val-he-7k`. The distinction "7,180 external patches — strictly out-of-domain" restates the firewall at the notebook level.

### Cell 1 — Environment & GPU verification (`bbfcf60d`)

```python
!pip install timm transformers tabulate --quiet

import torch
print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
else:
    print("WARNING: No GPU detected. Please enable Accelerator -> GPU P100 in Kaggle Settings.")
```

- **Line 1** — Kaggle's base image ships PyTorch but not these three. `timm` provides ResNet-50/ConvNeXt-Tiny, `transformers` provides Phikon, `tabulate` enables `DataFrame.to_markdown` (L573). `--quiet` suppresses pip's progress bars so the cell output stays short.
- **Line 2** — Torch version check; a mismatch with what the models expect is the usual cause of odd runtime errors.
- **Line 3** — `torch.cuda.is_available()` is the same check the script performs at L543. Printing it here, *before* the 3.5-hour run, is the whole point of the cell.
- **Lines 4–6** — On success, prints the device name and total VRAM in GB (`total_memory` is in bytes; the two divisions convert to GiB).
- **Lines 7–8** — The failure branch is a **warning, not an assertion**. The cell succeeds either way, so the run proceeds on CPU if you don't read the output — the notebook does not protect you from the mistake, it only narrates it. (Making this a hard `assert torch.cuda.is_available()` would be a one-line improvement; it is deliberately soft to allow CPU smoke-tests.)

### Cell 2 — Paths and script staging (`840ba1c3`)

```python
SOURCE_DIR = Path("/kaggle/input/nct-crc-he-100k-nonorm/NCT-CRC-HE-100K-NONORM")
TARGET_DIR = Path("/kaggle/input/crc-val-he-7k/CRC-VAL-HE-7K")
WORKING    = Path("/kaggle/working")

script_dst = WORKING / "run_experiments.py"
shutil.copy2(
    next(Path("/kaggle/input/histo-robust-code").rglob("run_experiments.py")),
    script_dst
)
```

- **`SOURCE_DIR` / `TARGET_DIR`** — Kaggle always mounts datasets at `/kaggle/input/<dataset-slug>/`, and the *uploaded zip's own top-level folder* becomes the first subdirectory. So the path is `slug/folder` — two levels, not one. If the zip was built with a different internal folder name, this path is wrong and the run fails at split preparation with the `No source images found under ... Contents found: [...]` error from L303.
- **`WORKING`** — `/kaggle/working/` is the only writable location, and it is exactly what gets bundled into the notebook's output. Everything is written under it so it survives the run.
- **`shutil.copy2`** — Copies the script *and its metadata* (timestamps, permissions) into the writable directory. Necessary because `/kaggle/input` is **read-only**: the script could be *executed* from the input path, but any path resolution or `.pyc` writing is safer from a writable directory.
- **`next(...rglob(...))`** — Locates `run_experiments.py` anywhere inside the attached code dataset, so the zip's internal structure doesn't need to be predicted.
  - **Gotcha**: `next()` on an exhausted iterator raises **`StopIteration`** — a bare, unhelpful exception — if the dataset isn't attached or the file is named differently. This happens *in Cell 2*, one cell **before** the friendly `FileNotFoundError` check in Cell 3, so the good error message never gets a chance to fire. The failure looks like an obscure iterator error rather than "attach the dataset."
- **The three `print` statements** — Echo each path with an `exists()` boolean. This is the notebook's only path verification, and it is worth reading before walking away from a long run: `False` on either `exists` predicts exactly which error you'll get.
- **Side effect worth noting**: `script_dst` is defined as a **module-level global** here and consumed by Cell 3 — so Cell 3 cannot be run standalone.

### Cell 3 — Execute all experiments (`dee42499`)

```python
if not script_dst.exists():
    raise FileNotFoundError(f"Cannot find {script_dst}. Ensure histo-robust-code dataset is attached.")

results_dir = WORKING / "results"
results_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, str(script_dst),
    "--data-root", str(SOURCE_DIR),
    "--target-root", str(TARGET_DIR),
    "--out-dir", str(results_dir),
    "--subset", "25000",
    "--epochs", "8",
    "--batch-size", "64",
]

print("Starting ablation pipeline:", " ".join(cmd))
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.wait()

if proc.returncode != 0:
    raise RuntimeError(f"Ablation suite exited with error code {proc.returncode}")
```

- **`if not script_dst.exists()`** — The friendly guard, but see the Cell 2 gotcha: by the time this runs, `script_dst` must already exist, so this check protects only against the file disappearing (or against running Cell 3 after a kernel restart where the name happens to survive).
- **`results_dir`** — Created before launch so the script's own `mkdir` at L529 is a no-op and the directory is visible even if the subprocess dies instantly.
- **`sys.executable`** — Runs the script with **the same interpreter as the kernel**, which is what guarantees it sees the packages Cell 1 just installed. Using the string `"python"` here would risk a different interpreter without `timm`/`transformers`.
- **`"--subset", "25000"` …** — All values are passed as **strings**; `argparse` converts them via the `type=int` declared at L521–L523. These seven flags exactly reproduce the script's own defaults — the notebook is explicit rather than relying on them, so the run configuration is visible and greppable in the notebook.
- **`subprocess.Popen` with `stdout=PIPE, stderr=STDOUT`** — Merges stderr into stdout so a traceback appears inline with the progress log rather than in a separate stream that Kaggle might not surface.
- **`text=True, bufsize=1`** — Text mode (universal newlines, no manual decode) and line buffering on *the parent's* reading side.
  - **Gotcha**: line buffering on the read side does **not** make the child's output line-by-line. The child is a plain `python script.py` whose stdout is a pipe, so Python block-buffers it (~8KB). Progress lines therefore arrive in **bursts**, not smoothly — the notebook looks frozen for minutes and then dumps a chunk. Adding `"-u"` after `sys.executable` (or `PYTHONUNBUFFERED=1`) would give true live logs. This affects only log latency, not correctness.
- **`for line in proc.stdout: print(..., flush=True)`** — Streams through to the notebook's own output with an explicit flush so each received chunk renders immediately.
- **`proc.wait()`** — Blocks until exit. (Iterating `proc.stdout` to EOF already implies the process has closed stdout, so this is a formality that also captures the return code reliably.)
- **`if proc.returncode != 0: raise`** — Converts a failed run into a *failed cell*. This matters: it stops the notebook from proceeding to Cell 4 and displaying a stale or empty table as if the run had succeeded.

### Cell 4 — View final results (`cell-4`)

```python
table_path = WORKING / "results" / "RESULTS_TABLE.md"
csv_path   = WORKING / "results" / "summary_results.csv"

if table_path.exists():
    display(Markdown(table_path.read_text(encoding="utf-8")))
elif csv_path.exists():
    display(pd.read_csv(csv_path))
else:
    print("No results files found in results/ directory.")
```

- **Two paths, in priority order** — Prefers the Markdown table (already formatted, already headed) and falls back to rendering the CSV as a DataFrame. The CSV is the *more* robust artifact (it doesn't depend on `tabulate` having been installed, per L572–L575), so the fallback is genuinely useful rather than decorative.
- **`display(Markdown(...))`** — Renders the table natively in the Kaggle output rather than as a code block, which is what makes the cell's output copy-pasteable into the report.
- **`read_text(encoding="utf-8")`** — Matches the explicit `encoding="utf-8"` at L577. Without it, the read would use the platform default and could mojibake.
- **The closing print** — Points at the Kaggle **Output** tab, where everything under `/kaggle/working/` (including `results/splits/*.csv` and `reference_stain.png`) can be downloaded after the run.
- **Note**: neither branch verifies that the table is *complete* (e.g. 13 rows). Combined with Cell 3's return-code check this is adequate, but a partial run that exits 0 — such as the `--experiments` typo case described at L531–L534 — would render a short table with no warning.

---

# Part C — Cross-cutting notes

## C.1 End-to-end trace: one tile through EXP-09

Concrete walkthrough for a single `TUM` patch in EXP-09 (`resnet50` + `macenko` + `aug_combined`).

**Setup (once, before the loop):**
1. `prepare_dataset_splits` walks both dataset roots, finds 9 class folders → 100,000 source records (24,993 after subsetting) and 7,180 target records.
2. Stratified split → 17,495 train / 3,749 val / 3,749 test-ID rows; all four CSVs written to `results/splits/`.
3. Reference tile = first `TUM` row of the training split, saved as `reference_stain.png`.
4. `macenko_fit(reference)` → `stain_matrix` (3×2 unit vectors, H in column 0) and `q99` (2 numbers).
5. `norm_fn = lambda img: macenko_apply(img, ref_params)`.

**One training sample:**
6. `__getitem__` → `Image.open(path)` (lazy) → `PatchTransform.__call__`.
7. `convert("RGB")` → `np.array` → `(224, 224, 3) uint8`. Decode happens here.
8. `norm_fn(arr)`: OD → mask tissue pixels (OD > 0.15) → if ≥20% tissue: SVD → stain plane → 1st/99th-percentile angles → this tile's `(H, E)` vectors → concentrations → rescale so this tile's q99 matches the reference q99 → recombine in *reference* stain basis → back to `uint8`. If <20% tissue, the array is returned untouched.
9. `random.random() > 0.2` → with p=0.8, `hed_stain_jitter`: deconvolve against the *fixed* Ruifrok matrix, draw `α ~ U(0.8, 1.2)` and `β ~ U(-0.05, 0.05)` per stain, clip concentrations at 0, recombine.
10. `TF.to_tensor` → `(3, 224, 224) float32` in `[0, 1]`.
11. h-flip (p=.5), v-flip (p=.5), rotation ∈ {0, 90, 180, 270} (uniform).
12. `TF.normalize(IMAGENET_MEAN, IMAGENET_STD)` → final tensor handed to the collate function.

**Training:** batch of 64 such tensors → ResNet-50 forward under autocast → cross-entropy with label smoothing 0.1 → scaled backward → AdamW step → cosine scheduler step. Repeated 274×/epoch for 8 epochs, validating on the 3,749-tile val set after each epoch and snapshotting the best macro-F1 weights.

**Reporting:** best weights restored → `evaluate_model` on test-ID (3,749 tiles) and test-OOD (7,180 tiles) → `Delta_F1 = F1_ID − F1_OOD`, `RR_F1 = F1_OOD/F1_ID × 100` → one row appended to `results` → after all 13 cells, `summary_results.csv` + `RESULTS_TABLE.md`.

## C.2 Reproducibility and determinism

| Aspect | Seeded? | Consequence |
| :--- | :--- | :--- |
| Train/val/test split | **Yes** — `seed=42` (L290, L318–326) | Identical splits on every run and every machine, as long as the input file set is identical |
| Per-class subsampling | **Yes** — `random_state=seed` (L313) | Same 2,777 tiles per class every run |
| Reference tile | **Yes** — deterministic `iloc[0]` (L344) | All 13 cells and all re-runs normalize to the same tile |
| Augmentation draws | **Partially** — Python `random` seeded once at L291, then consumed by DataLoader workers on Linux | Different augmentation stream per experiment (fine) and per run (not reproducible) |
| DataLoader shuffle order | **No** — torch global RNG never seeded | Different batch order every run |
| Model weight init / dropout / AMP | **No** — no `torch.manual_seed` anywhere | Different initialization every run |
| cuDNN kernels | **No** — no `torch.backends.cudnn.deterministic = True` | Non-deterministic reductions on GPU |

**Practical expectation**: re-running the same command will reproduce the *splits* exactly but will give test metrics that differ by a small amount (typically a few tenths of a point of macro F1, occasionally more). Differences of that size between two experiments in the matrix should therefore **not** be over-interpreted as an effect of the defense — this is the single most important caveat when reading `docs/RESULTS.md`. Reducing it would require `torch.manual_seed`, a seeded `generator=` for each DataLoader, `worker_init_fn`, and cuDNN determinism flags.

## C.3 Gotchas and failure modes

| # | Where | Symptom | Cause / fix |
| :---: | :--- | :--- | :--- |
| 1 | `main()` L543 | `Running on device: cpu`, then a run that takes days | Silent CPU fallback when the Kaggle accelerator isn't enabled. Check the banner in the first minute. |
| 2 | `--experiments` L524/L532 | All 13 experiments run when you expected none | `nargs="*"` with no values → `[]` → falsy → no filtering. |
| 3 | `--experiments` typo L534 | Run completes, `summary_results.csv` is rewritten **empty** | Unknown IDs are silently filtered out; there is no "no experiments matched" error. |
| 4 | Split prep L303 | `No source images found under …` | Wrong dataset path/slug or an extra nesting level. The error lists the first 15 entries found — use them to correct the path. |
| 5 | Notebook Cell 2 | Bare `StopIteration` | `next()` on `rglob` when the code dataset isn't attached — fires before Cell 3's friendly message. |
| 6 | Notebook Cell 3 | Logs arrive in minutes-long bursts | Child Python block-buffers stdout when piped. Add `-u` to `sys.executable` for live logs. |
| 7 | Notebook Cell 1 | Run proceeds on CPU despite the warning | The check prints a warning but does not assert. |
| 8 | `macenko_apply` L138 | Some tiles are visibly *not* normalized | Intentional: tiles with <20% tissue are returned untouched. |
| 9 | `macenko_apply` L163 | Normalization silently skipped | Blanket `except` returns the original image with no warning or counter. |
| 10 | `EXPERIMENTS` typos | A cell trains the wrong architecture | `build_model` falls through to ResNet-50; `train_experiment` falls through to `norm_fn=None`. No validation of the matrix. |
| 11 | `evaluate_model` L409 | `nan` AUROC | Expected on tiny subsets where some class has no samples; wrapped in `try/except`. |
| 12 | Main loop L555 | A crash at EXP-11 loses EXP-01…10 | Results are only written after the loop. Add per-experiment CSV writes inside the loop if you need crash recovery. |
| 13 | `--subset` | 13 experiments silently run on 25k instead of 100k | Default is 25000; `0` is the documented "full" value. |
| 14 | pandas ≥ 2.2 | `DeprecationWarning` from `groupby().apply()` at L312 | Harmless; the `group_keys=False` form behaves as intended. |

## C.4 Runtime and resource model

Per experiment at defaults (`--subset 25000 --epochs 8 --batch-size 64`):

| Quantity | Value | Derivation |
| :--- | :--- | :--- |
| Training tiles | 17,495 | 24,993 × 0.70 |
| Steps per epoch | ~274 | 17,495 ÷ 64, rounded up |
| Scheduler steps per run | ~2,192 | 274 × 8 (this is `T_max` at L455) |
| Validation passes | 8 | one per epoch, 3,749 tiles each |
| Test passes | 2 | test-ID 3,749 + test-OOD 7,180 |
| Total forward passes | ~46k train + ~37k eval per experiment | |
| Measured wall-clock | ~16 min/experiment → **~3.5 h for 13** | As documented in `README.md` / `docs/kaggle_guide.md` |

Memory: the largest single object is one batch (64 × 3 × 224 × 224 × 4B ≈ 38 MB in fp32) plus activations; the `best_state` deepcopy (L481) holds a second copy of the model in CPU RAM (~100 MB ResNet-50, ~110 MB ConvNeXt-Tiny, ~340 MB Phikon fp32). Kaggle's P100 (16 GB) / T4×2 (2×15 GB) has ample headroom — the binding constraint is time, not memory.

Note that Phikon's frozen encoder means EXP-12/13 backpropagate through the ViT but update only a 6,921-parameter linear head. Step time is still dominated by the ViT forward/backward; convergence is usually faster per epoch but the ceiling is lower, since the representation itself cannot adapt to stain variation.

## C.5 Known doc-vs-code discrepancies

These are places where the documentation and the code disagree. The code is authoritative here; the list exists so a reader of `docs/` isn't misled.

1. **`summary_results.csv` column names.** `docs/kaggle_guide.md` (line 78) lists the columns as `(… , val_f1, … , runtime)`. The code actually emits **`best_val_f1`** (L502) and **`minutes`** (L509). Anyone scripting against the CSV using the documented names gets a `KeyError`.
2. **`docs/dataset_card.md` is superseded but still present.** `DATA.md` states that it supersedes the former dataset card, yet the older file still exists and is still listed in the README and `docs/PLAN.md` §5 trees. Its split arithmetic is also wrong — it quotes 17,500 / 3,750 / 3,750, while the splits actually produced are **17,495 / 3,749 / 3,749**.
3. **`--dry-run` is no longer documented anywhere.** The flag still exists (L525) and prints the 13-cell plan without touching data or torch, but the README no longer mentions it now that the local-execution section was removed.
4. **`CLASS_TO_IDX`** (L16) is defined but never used — harmless, but it is not the inverse mapping used anywhere in the pipeline.
5. **Accelerator recommendation.** `notebook.ipynb` Cell 0 and `docs/kaggle_guide.md` §3 both specify `GPU T4 x2`; the README lists `T4 x2` or `P100`. All three are usable — the ~3.3 h measured runtime is not attributed to any specific accelerator.

**Resolved in earlier revisions** — recorded so older notes are not mistaken for live issues:

- The `tests/` directory that an earlier `README.md` advertised does not exist, and the current README no longer references it.
- The `src/histo_robust/` package and all `uv` tooling (`pyproject.toml`, `uv.lock`, `.python-version`, `.venv`) have been removed; the project is Kaggle-only and the pipeline is a single standalone script.
- `docs/PLAN.md` §5 now documents the repository structure that actually ships, rather than the `src/` package layout that was planned but never built.

---

*This walkthrough documents `run_experiments.py` (587 lines) and `notebook.ipynb` (5 cells) as they exist at the time of writing. Line numbers refer to that revision; if the files change, re-check the anchors rather than assuming the numbers still point where they did.*
