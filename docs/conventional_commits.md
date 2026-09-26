# Conventional Commits Specification & Guide

This document defines the **Conventional Commits standard** enforced in the **Robust Histopathology Image Classification under Staining Variations** repository.

---

## 1. Specification Overview

Commit messages in this repository must follow the **Conventional Commits v1.0.0** format:

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### The Three Hard Rules
1. **Header format**: Must be `<type>(<scope>): <description>`.
2. **Header length**: ....
3. **Imperative mood**: The description must use lowercase, imperative mood ("add", "fix", "update", not "added", "fixes", "updating"), and must **not** end with a period (`.`).

---

## 2. Commit Types

| Type | Purpose | When to Use |
| :--- | :--- | :--- |
| `feat` | New feature or capability | Adding a stain normalizer, augmentation transform, model backbone, or new CLI flag. |
| `fix` | Bug fix | Correcting a runtime crash, CUDA OOM, checkpoint pruning logic, or API rate limit. |
| `perf` | Performance improvement | Pre-normalizing Macenko tiles, enabling AMP fp16 autocast, DataLoader worker optimizations. |
| `refactor` | Code restructuring | Reorganizing code structure without changing user-facing behavior or fixing a bug. |
| `docs` | Documentation only | Updating `PLAN.md`, `kaggle_guide.md`, `RESULTS.md`, docstrings, or diagrams. |
| `test` | Test suites | Adding or updating unit tests, synthetic tests, or the `tests/run_all.py` harness. |
| `build` | Build system & dependencies | Modifying `pyproject.toml`, `uv.lock`, packaging scripts, or build configs. |
| `ci` | Continuous integration | Updating GitHub Actions, automated test workflows, or verification hooks. |
| `chore` | Maintenance tasks | Updating `.gitignore`, repo directory restructuring, or cleaning temporary files. |
| `exp` | Experimental configuration | Adjusting ablation hyper-parameters, updating `experiments_registry.json`, or cell configs. |

---

## 3. Project Scopes

Scopes contextualize where the change occurred. Contributors should choose from the following predefined scopes:

| Scope | Area of Codebase | Primary Path(s) |
| :--- | :--- | :--- |
| `data` | Datasets, splits, dataloaders, domain firewall | `src/histo_robust/data/` |
| `norm` | Stain normalizers (Reinhard, Macenko, Vahadane) | `src/histo_robust/normalization/` |
| `aug` | Augmentations (HedJitter, RandAugment, ColorJitter) | `src/histo_robust/augmentation/` |
| `models` | Model backbones, encoders, heads (ResNet, ConvNeXt, Phikon) | `src/histo_robust/models/` |
| `engine` | Trainer, optimizer, scheduler, loss, eval loop | `src/histo_robust/engine/` |
| `checkpoint`| CheckpointManager, save/resume, quota pruning | `src/histo_robust/utils/checkpoint.py` |
| `kaggle` | Notebook generator, helper scripts, zip packager | `kaggle/`, `scripts/make_zips.py`, `_kaggle_/` |
| `config` | Experiment configs and registry definitions | `configs/`, `configs/experiments/` |
| `cli` | Command-line interface and dispatchers | `src/histo_robust/cli.py`, `scripts/` |
| `metrics` | Evaluation metrics, confusion matrix, robustness calculations | `src/histo_robust/engine/metrics.py` |

---

## 4. Examples: Good vs. Bad Commits

### Feature (`feat`)
* **Good**:
  ```text
  feat(norm): add vahadane stain decomposition normalizer
  
  Implements SPAMS-based sparse non-negative matrix factorization for
  Vahadane stain separation as an alternative to SVD in Macenko.
  ```
* **Bad**: `added vahadane` *(missing type/scope, non-imperative, too vague)*

### Bug Fix (`fix`)
* **Good**:
  ```text
  fix(checkpoint): prevent ZeroDivisionError when save_every_steps is zero
  
  Disabling step checkpoints with save_every_steps: 0 caused a modulo by zero
  exception in should_save_step. Check that save_every_steps > 0 first.
  ```
* **Bad**: `fixed division bug.` *(missing type, ends with period, not capitalized properly)*

### Performance (`perf`)
* **Good**:
  ```text
  perf(data): pre-normalize Macenko tiles to eliminate CPU deconvolution bottleneck
  
  Running Macenko SVD deconvolution on CPU during training batch iteration
  caused a 10x throughput slowdown. Caching pre-normalized tiles to disk
  reduces epoch duration from 30m to 2m.
  ```
* **Bad**: `perf: speed up macenko` *(missing scope, lack of explanation in body)*

### Kaggle Workflow (`kaggle`)
* **Good**:
  ```text
  fix(kaggle): handle KGAT token Bearer auth and add 429 retry backoff in download script
  
  Kaggle personal access tokens starting with KGAT_ require Bearer authorization
  rather than Basic auth. Also added exponential backoff on HTTP 429.
  ```
* **Bad**: `fix download script` *(missing scope, non-standard format)*

---

### 2. Validating Commits with Git Hooks
To catch non-compliant commit messages before they are recorded, you can create a `.git/hooks/commit-msg` hook:

```bash
#!/usr/bin/env bash
# .git/hooks/commit-msg
commit_regex='^(feat|fix|perf|refactor|docs|test|build|ci|chore|exp)(\([a-z0-9_-]+\))?!?: .+$'

if ! grep -qE "$commit_regex" "$1"; then
    echo "ERROR: Commit message does not follow Conventional Commits standard!"
    echo "Format: <type>(<scope>): <subject>"
    echo "Example: feat(norm): add macenko stain decomposition"
    exit 1
fi
```
Make the hook executable:
```bash
chmod +x .git/hooks/commit-msg
```

---

## 5. Summary Checklist Before Committing

- [ ] Does the commit header start with an allowed type (`feat`, `fix`, `perf`, `docs`, etc.)?
- [ ] Is there an appropriate scope enclosed in parentheses (e.g. `(norm)`, `(kaggle)`)?
- [ ] Is the description in the imperative mood, all lowercase, without a trailing period?
- [ ] Is the header line within 72 characters?
- [ ] If breaking changes are introduced, is `!` included or `BREAKING CHANGE:` documented in the footer?
