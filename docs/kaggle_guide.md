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
python scripts/make_zips.py --out-dir dist
```

Expected output (counts drift as the project grows — 71 entries at the time of writing):

```text
[1/3] codebase
  wrote histo-robust-code.zip: 71 files, 0.24 MB
  verified: all 18 required entries present
[2/3] datasets
  copied NCT-CRC-HE-100K-NONORM.zip (11183 MB)
  copied CRC-VAL-HE-7K.zip (763 MB)
[3/3] checkpoints
  skipped (pass --checkpoints <dir> to bundle a previous session)

[OK  ] histo-robust-code.zip: 71 entries
[OK  ] NCT-CRC-HE-100K-NONORM.zip: 100010 entries
[OK  ] CRC-VAL-HE-7K.zip: 7190 entries

VERIFYING the codebase archive contains everything Kaggle cell 0 needs
  [OK  ] histo-robust-code.zip: all 18 required entries present (including the whole histo_robust package)
         29 package modules, 8 documentation files
```

`dist/` now holds the three upload artifacts. The two dataset archives are
copied verbatim from `zipped_dataset/` (Zenodo's own archives already use
forward slashes — verified above). Add `--repack-data` only if you have the raw
PNG folders and no archive; it re-creates them from scratch and takes a long
time for the 100K set.

### Why the build now verifies itself

`make_zips.py` refuses to write a codebase archive that Kaggle cell 0 would
reject, and it aborts with the reason. Two guards, both added after real
failures:

* **Repo-root check.** The folder being archived must *directly* contain
  `pyproject.toml`, `src/histo_robust`, `scripts/run_all_ablations.py` and
  `src/histo_robust/data/paths.py`. Archiving a parent folder (a workspace
  holding several projects) is rejected with `FATAL: the folder being archived is
  not the project root`.
* **Archive-content check.** 18 required entries are asserted to be *inside* the
  written zip, and the run exits non-zero if any is missing. This exists because
  an earlier version excluded directories named `data` **at any depth**, which
  silently dropped the `src/histo_robust/data/` subpackage. The result on Kaggle
  was:

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
```

Any `[!! ]` line means a dataset is missing, flattened, or incomplete: the cell
raises with the specific fix rather than letting the run continue. Cell 2 then
reports the image counts per split:

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
   * (from session 2 onwards) `histo-robust-checkpoints`
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
| **2 — Data + checkpoints** | Reports image counts per dataset and copies any previous session's `*.pt` into `/kaggle/working/checkpoints`. Datasets stay read-only on `/kaggle/input`, so ~12 GB of PNGs never touch the 20 GB quota. | < 1 min (0 files on session 1) |
| **3 — Splits + reference** | `scripts/prepare_splits.py`: stratified 70/15/15 source split at seed 42, the full target set as `test_ood`, one canonical H&E reference tile chosen **from the training split only**, plus firewall assertions. | 2–5 min |
| **4 — Pre-flight benchmark** | `scripts/smoke_test.py`: throughput, a real forward/backward step, minutes/epoch, `OK` / `CPU-BOUND` / `IDLE-GPU`. | 3–8 min |
| **5 — The run** | `scripts/run_all_ablations.py`: trains every pending cell under the 10.5 h time box, evaluates ID and OOD, updates the manifest. | up to ~11 h |
| **6 — Artifacts** | Zips the checkpoints into `for_upload/` and prints the download list with per-cell status. | 1–3 min |

Every script call uses `subprocess.run(..., check=True)`. There is deliberately no
retry/fallback scaffolding around them: when a script fails, **its own traceback
appears in the cell output naming the file and line**, which is where the bug is.

> **Normalisation cache.** No dedicated cell. If cell 4 reports `CPU-BOUND`, call
> `scripts/preprocess_normalize.py` from the shell, or add
> `--normalization-cache <dir>` to cell 5's orchestrator command. The fast path is
> optional: the online pipeline is equivalent by construction.

### Expected first-session timeline

| Elapsed | What happens | Log line to expect |
| ---: | :--- | :--- |
| 0:00 | Session starts | `CELL 0 -- VERIFY   session start (UTC): ...` |
| 0:00–0:01 | **Cell 0** verifies paths, datasets, codebase, imports | `VERIFIED -- safe to continue` |
| 0:01–0:05 | **Cell 1** installs, asserts the GPU | `SETUP COMPLETE` |
| 0:05–0:06 | **Cell 2** mounts data, copies checkpoints (0 on session 1) | `source   100000 images across 9 classes` |
| 0:06–0:10 | **Cell 3** splits + reference tile | `DOMAIN FIREWALL: PASSED` |
| 0:10–0:20 | **Cell 4** pre-flight benchmark | `PRE-FLIGHT VERDICT: OK` |
| 0:20 → 11:00 | **Cell 5** runs ablation cells under the 630-minute box | `EXP-01 \| epoch 4/12 \| ...` |
| ~11:00 | Clean self-stop, checkpoints flushed | `TIME BOX REACHED (run_time_budget)` |
| 11:00–11:05 | **Cell 6** zips checkpoints, writes the manifest | `wrote histo-robust-checkpoints.zip` |

On session 1, cell 3 spends ~7 minutes scanning 100,000 source patches (counted
per class, class by class) before writing the split CSVs. That is the one slow step
before training, and it only happens once per session.

### Reading the pre-flight verdict

```text
  loader throughput   : 240.5 img/s (0.27 s/batch)
  train step          : 0.19 s (337 img/s)
  peak GPU memory     : 3.9 GB
  minutes per epoch   : 1.8
  epochs in budget    : 6.9 of 12 planned
  VERDICT             : OK
```

* `OK` — go ahead.
* `CPU-BOUND` (loader slower than the GPU step) — build the normalisation cache
  and point cell 5 at it:

  ```python
  # in cell 5, before the orchestrator call
  import subprocess, sys
  subprocess.run([sys.executable, str(REPO / "scripts" / "preprocess_normalize.py"),
                  "--normalization", "macenko",
                  "--splits-dir", str(SPLITS_DIR),
                  "--out-dir", str(WORKING / "data" / "processed" / "cache"),
                  "--reference", str(TEMPLATE_DIR / "reference_stain.png"),
                  "--workers", "4"], check=True)
  NORMALIZATION_CACHE = WORKING / "data" / "processed" / "cache" / "macenko"
  command += ["--normalization-cache", str(NORMALIZATION_CACHE)]
  ```

  Cached tiles are produced by the same normaliser, reference and crop as the
  online path, so the two are equivalent by construction; the run simply reads
  pre-normalised JPEGs instead of deconvolving every tile on the CPU.
* `IDLE-GPU` — raise `data.num_workers` (4 is the Kaggle vCPU count; 2 is
  sometimes faster under memory pressure), or build the cache as above.
* `epochs in budget` below ~3 — reduce `SUBSET` in cell 3, or set
  `freeze_backbone: true` in the config for the Phikon cells.

---

## 4. Execution: trigger the background run

1. **Save Version** (top right) → choose **Save & Run All (Commit)** → **Save**.
2. Kaggle immediately detaches the run. You can close the browser; the run
   continues in the background.
3. The session is hard-killed at **12 h**. The code stops itself at **630 min
   (10.5 h)** by default and also reserves **20 min** before the session
   deadline — whichever comes first — and before exiting it:
   * breaks at a batch boundary (never mid-write),
   * writes `last.pt` (model + optimizer + scheduler + AMP scaler + RNG state) and
     `best.pt` (best `val_id` macro-F1),
   * evaluates the frozen checkpoint on `test_id` and `test_ood`,
   * prunes checkpoints and updates the manifest.

   That is why the run ends "successfully" rather than as a 12-hour timeout: a
   clean exit is what makes the next session resumable.

### Monitoring

`results/logs/run_all_ablations.log` is the full transcript. Watch for:

```text
EXP-03 | epoch 4/12 | train_loss=0.4213 | val_acc=0.9102 val_macro_f1=0.8931 ... | 38.4 min elapsed
EXP-03 | TIME BOX REACHED (run_time_budget) after epoch 7. Saving last.pt and exiting cleanly ...
```

`[TimeBudget:EXP-03] elapsed=75.0 min | remaining=0.0 min | session_left=214.6 min` is
printed at the top of every epoch and is the number to watch if you are unsure
how much of the session is left.


## 5. Offline model weights (Internet = OFF)

Only the Phikon cells (`EXP-12`, `EXP-13`) and the timm ImageNet weights need a
download. With Internet ON they are fetched once and cached for the session. If
you must run with Internet OFF, mount the weights as a dataset:

1. On a machine with Internet, download both:
   * timm: `python -c "import timm; timm.create_model('resnet50.a1_in1k', pretrained=True); timm.create_model('convnext_tiny.fb_in1k', pretrained=True)"`
     → cached under `~/.cache/huggingface/hub/`
   * Phikon: `huggingface-cli download owkin/phikon --local-dir phikon`
     (Phikon is **ungated**, so no token is required.)
2. Zip the cache with forward slashes and upload it as a dataset, e.g.
   `histo-robust-weights`, so it mounts at
   `/kaggle/input/histo-robust-weights/`.
3. In cell 5, or via the config, point the loader at it:

   ```python
   command += ["--weights-dir", "/kaggle/input/histo-robust-weights"]
   ```
   or in a config file: `paths: { weights_dir: /kaggle/input/histo-robust-weights }`

The loader recognises `phikon/`, `models--owkin--phikon/snapshots/<rev>/` and a
plain directory containing `config.json` + `model.safetensors`.

---