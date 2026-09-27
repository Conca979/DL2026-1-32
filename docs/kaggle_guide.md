# Kaggle Setup Guide — Robust Histopathology Classification under Staining Variations


This is the exact, click-by-click procedure for running the 13-cell ablation matrix
(`docs/PLAN.md` §3) on Kaggle's free GPU tier.

Read the five rules first — they are the ones that actually cost people runs:

| Rule | Consequence if ignored |
| :--- | :--- |
| **Zip from the project root, with forward slashes.** Never the Windows "Send to → Compressed folder" tool, and never from a parent folder. | Backslash entry names extract as flat files; a parent folder archived by mistake yields a codebase Kaggle cannot import. |
| **Never let `/kaggle/working` exceed 20 GB.** | Kaggle kills the session's outputs; checkpoints and metrics are lost with no warning. |
| **Never plan to run past 12 h.** | Kaggle terminates the session at exactly 12:00:00 with no cleanup hook. |

The code handles the middle three automatically. The zipping rule is mechanical —
`scripts/make_zips.py` enforces it and now verifies the result.

---

## 0. What you need before you start

* A Kaggle account with **phone verification** (free GPU and Internet require it).
* The two dataset archives from Zenodo record [1214456](https://doi.org/10.5281/zenodo.1214456):
  * `NCT-CRC-HE-100K-NONORM.zip` (~11.4 GB compressed, 100,000 patches)
  * `CRC-VAL-HE-7K.zip` (~700 MB compressed, 7,180 patches)
* A standalone checkout of this repository (it is a single self-contained project:
  `pyproject.toml`, `src/`, `scripts/`, `configs/`, `kaggle/`, `docs/`, `tests/`).

---

## 1. The Zipping Rule (do this first, it prevents the #1 failure)

### 1a. Why the Windows zip button breaks Kaggle

Windows' built-in "Compressed (zipped) folder" writes archive entry names with
**backslashes**: `data\raw\ADI\patch.png`. Linux — and therefore Kaggle — treats
a backslash as an ordinary filename character, so the extraction produces a
single file whose *name* contains backslashes rather than nested directories.
You then spend an hour debugging a "missing class folder" error that has nothing
to do with your data.

`scripts/make_zips.py` never has this problem: Python's `zipfile` always writes
`/`-separated names, and the script re-opens every archive it produces and
verifies that no entry contains a backslash, an absolute path or a drive letter.

### 1b. Run it

From the repository root:

```powershell
python scripts/make_zips.py dist
```

Expected output (counts drift as the project grows — 62 entries at the time of writing):

```text
Packaging 62 files into dist\histo-robust-code.zip...
[OK] Wrote dist\histo-robust-code.zip (62 files, 0.37 MB)
Next step: Upload to Kaggle dataset 'histo-robust-code'.
```

`dist/` now holds the three upload artifacts.

### Why the build now verifies itself

`make_zips.py` refuses to write a codebase archive that Kaggle cell 0 would
reject, and it aborts with the reason. Two guards, both added after real
failures:

* **Repo-root check.** The folder being archived must *directly* contain
  `pyproject.toml`, `src/histo_robust`, `scripts/run_all_ablations.py` and
  `src/histo_robust/data/paths.py`. Archiving a parent folder (a workspace
  holding several projects) is rejected with `FATAL: the folder being archived is
  not the project root`.

  ```text
  ModuleNotFoundError: No module named 'histo_robust.data'
  ```

  The exclusion list is now root-only for data/result directories
  (`CODE_EXCLUDE_ROOT_DIRS`) and any-depth only for caches and virtualenvs
  (`CODE_EXCLUDE_ANY_DIRS`).

Housekeeping: the codebase archive also carries `docs/`, which is what lets you
read `docs/kaggle_guide.md` and `docs/APPENDICES.md` from inside a Kaggle session.

### 1c. Manual alternative (Linux / WSL only)

If you would rather do it by hand, use `zip` and never a GUI tool:

```bash
cd /path/to/32_histopathology_staining_robustness
zip -r dist/histo-robust-code.zip \
    src scripts configs kaggle tests docs \
    pyproject.toml README.md \
    -x '*/__pycache__/*' '*.pyc' 'data/*' 'checkpoints/*' 'results/*' 'dist/*'
```

Verify before uploading:

```bash
python -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); bad=[n for n in z.namelist() if '\\\\' in n]; print('BAD' if bad else 'OK', len(z.namelist()), bad[:3])" dist/histo-robust-code.zip
```

---

## 2. Create the three (then four) Kaggle datasets

Go to <https://www.kaggle.com/datasets> → **New Dataset** for each item below.
All four are **private** — this project's results are not public.

| # | Dataset title (use exactly this slug) | Upload file | Notes |
| :-: | :--- | :--- | :--- |
| 1 | `histo-robust-code` | `dist/histo-robust-code.zip` | The codebase. Re-upload a **New Version** whenever you change any `.py`/`.yaml`/`.md`. |
| 2 | `nct-crc-he-100k-nonorm` | `dist/NCT-CRC-HE-100K-NONORM.zip` | In-domain source (~1.4 GB zipped, ~11 GB expanded). |
| 3 | `crc-val-he-7k` | `dist/CRC-VAL-HE-7K.zip` | Out-of-domain target. |
| 4 | `histo-robust-checkpoints` | *created after session 1* | See §5. Do **not** create it now with a placeholder. |

Uploading tips that matter at these sizes:

* Use the Kaggle **web uploader** and leave the browser tab open until it says
  "Completed". The 1.4 GB source archive typically takes 10–30 minutes on a
  domestic uplink.
* Do **not** unzip before uploading. Upload the `.zip`; the uploader chokes on a
  folder containing 100,000 PNGs.
* Set visibility to **Private**.

### 2.1 What Kaggle actually does with your zip (read this before cell 0)

**The web uploader auto-extracts a zip.** You upload `histo-robust-code.zip`, but
what appears under `/kaggle/input` is a *directory* named after the dataset.

**Kaggle mounts inputs in one of two places.** Newer accounts nest them one level
deeper, which is what this project's notebook targets:

```text
/kaggle/input/
└── datasets/
    └── <your-kaggle-username>/
        ├── histo-robust-code/            <- auto-extracted zip, NO .zip present
        │   ├── configs/  docs/  kaggle/  scripts/  src/  tests/
        │   ├── pyproject.toml
        │   └── README.md
        ├── nct-crc-he-100k-nonorm/
        │   └── NCT-CRC-HE-100K-NONORM/   <- slug and inner folder differ
        │       ├── ADI/  BACK/  ...  TUM/
        └── crc-val-he-7k/
            └── CRC-VAL-HE-7K/            <- slug and inner folder differ
                ├── ADI/  BACK/  ...  TUM/
```

Older accounts mount the same datasets directly as `/kaggle/input/<slug>/`. Cell 0
declares the nested form, because that is what the target notebook sees:

```python
KAGGLE_USER = "tahuy138" (for eg)
DATASETS = INPUT / "datasets" / KAGGLE_USER
CODEBASE_DATASET = DATASETS / "histo-robust-code"
SOURCE_DATASET = DATASETS / "nct-crc-he-100k-nonorm" / "NCT-CRC-HE-100K-NONORM"
TARGET_DATASET = DATASETS / "crc-val-he-7k" / "CRC-VAL-HE-7K"
CHECKPOINT_DATASET = DATASETS / "histo-robust-checkpoints"
```

**If your mount is the flat form**, change one line: `DATASETS = INPUT`. That is
the only edit needed, and it is the reason the paths are declared in one place.

### 2.2 Two real failures this section exists to prevent

Both happened on actual runs, in order.

**Failure 1 — searching for a zip that Kaggle had already extracted.**

```text
SystemExit: FATAL: could not find the codebase zip under /kaggle/input.
Attach the 'histo-robust-code' dataset (see docs/kaggle_guide.md), then re-run.
```

Early cell 1 looked only for `histo-robust-code.zip`. The dataset *was* attached;
the uploader had unpacked it, so no zip existed. Cell 0 now accepts an extracted
directory or an archive, and prints an inventory of what *is* mounted when it
fails. That first run also took ~15 minutes to fail, because the discovery scan
was unbounded over 107,180 PNGs; the current walk is depth-limited and prunes
image directories, so it takes milliseconds.

**Failure 2 — the archive omitted the `data` subpackage.**

```text
File "/kaggle/working/histo-robust/scripts/prepare_splits.py", line 43, in <module>
    from histo_robust.data.paths import IMAGE_SUFFIXES, find_class_dirs, iter_image_files
ModuleNotFoundError: No module named 'histo_robust.data'
```

The archiver excluded directories named `data` **at any depth** to keep the raw
dataset out of the codebase zip — which also excluded `src/histo_robust/data/`,
the data-access subpackage. Every script imports it, so nothing could run. Three
things changed:

* exclusions are now root-only for data/result directories
  (`CODE_EXCLUDE_ROOT_DIRS`) and any-depth only for caches and virtualenvs
  (`CODE_EXCLUDE_ANY_DIRS`);
* `make_zips.py` asserts 18 required entries are *inside* the written archive and
  exits non-zero otherwise;
* cell 0 imports all six subpackages and refuses to continue if any is missing.

The lesson generalises: **verify at the boundary you can still fix.** The archive
is now checked where it is built, and the mount is checked before any long work.

### 2.3 Verify the mount before running anything else

Cell 0 is the check. On success it ends with:

```text
[0a] datasets
  [OK ] codebase dataset       /kaggle/input/datasets/<owner>/histo-robust-code
  [OK ] source (in-domain)     100000 images, 9/9 classes
  [OK ] target (OOD)             7180 images, 9/9 classes
  [-- ] checkpoints dataset    (not attached: fine for session 1)

[0b] codebase structure
  [OK ] pyproject.toml
  [OK ] src/histo_robust/data/paths.py
  [OK ] scripts/run_all_ablations.py
  [OK ] kaggle/notebook_helpers.py

[0c] working copy + package import
  copied … -> /kaggle/working/histo-robust
  [OK ] histo_robust 0.1.0 from /kaggle/working/histo-robust/src/histo_robust/__init__.py
  [OK ] all six subpackages import (data, normalization, augmentation, models, engine, utils)
       29 python files in the working copy

[0d] verdict
  datasets      : OK
  codebase      : OK
  package       : OK
  -> safe to continue. Next cell installs the project and asserts the GPU.

```text
source (in-domain) : /kaggle/input/datasets/<owner>/nct-crc-he-100k-nonorm/NCT-CRC-HE-100K-NONORM
target (OOD)       : /kaggle/input/datasets/<owner>/crc-val-he-7k/CRC-VAL-HE-7K
  source   100000 images across 9 classes
  target     7180 images across 9 classes
checkpoint files copied: 0  (fresh start -- expected on session 1)
```

### 2.4 Prerequisite: phone-verified Kaggle account

Free GPU access (P100 / T4 x2) and the Internet toggle require a one-time SMS
**phone verification**: <https://www.kaggle.com/settings> → *Phone Verification*.
Without it the Accelerator setting is locked to `None` and cell 1's GPU assertion
stops the run with `FATAL: no CUDA device`.

---

## 3. Create the notebook and attach everything

1. **Code** → **New Notebook**.
2. **File → Import Notebook** → upload
   `kaggle/notebook_01_run_all_ablations.ipynb` from this repository.
3. Open the right-hand **Settings** panel and set:
   * **Accelerator**: `GPU P100` (preferred) or `GPU T4 x2`.
   * **Internet**: `On` for the first session — pretrained weights are downloaded
     and then cached in `~/.cache` for the rest of the session.
   * **Persistence**: `Files only` (there is nothing to persist; state travels in
     the checkpoint dataset).
4. Open the **Data** panel → **Add Input** → **Your Datasets** and attach:
   * `histo-robust-code`
   * `nct-crc-he-100k-nonorm`
   * `crc-val-he-7k`
5. Run **cell 0** first. It takes about three seconds and verifies every dataset,
   the codebase structure and the package import. Correct the mount before
   running anything long.

### What each cell does

The notebook has one markdown overview and seven code cells numbered **0** to **6**.
Each cell is short: it sets paths and calls one script. The only cell that checks
anything on your behalf is cell 0.

| Cell | Purpose | Typical runtime |
| :--- | :--- | :--- |
| **0 — Paths + verify** | Declares every dataset path **once**, then checks: each dataset exists, each has nine non-empty class folders, the codebase contains `pyproject.toml` + `src/histo_robust/data/paths.py` + `scripts/run_all_ablations.py` + `kaggle/notebook_helpers.py`, and `histo_robust` plus all six subpackages import. Prints what *is* mounted and raises with the fix if anything fails. | ~3 s |
| **1 — Install + GPU** | Installs the project (a no-op on Kaggle), sets `PYTHONPATH` for the subprocesses, **hard-asserts CUDA**. | 1–4 min |
| **2 — Splits + reference** | `scripts/prepare_splits.py`: stratified 70/15/15 source split at seed 42, the full target set as `test_ood`, one canonical H&E reference tile chosen **from the training split only**, plus firewall assertions. | 2–5 min |
| **3 — The run** | `scripts/run_all_ablations.py`: trains every pending cell under the 10.5 h time box, evaluates ID and OOD, updates the manifest. | up to ~11 h |
| **4 — RESULTS TABLE + CLEANUP** | Done | 1–3 min |

## 4. Execution: trigger the background run

1. **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
2. Kaggle immediately detaches the run. You can close the browser; the run
   continues in the background.

