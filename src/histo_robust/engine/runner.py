from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from histo_robust.config import EXPERIMENTS

if TYPE_CHECKING:
  import pandas as pd


def build_parser() -> argparse.ArgumentParser:
  """Build command-line interface arguments parser."""
  parser = argparse.ArgumentParser(description="Run 13-cell Histopathology Staining Robustness Ablations")
  parser.add_argument("--data-root", default="data/raw/NCT-CRC-HE-100K-NONORM", help="Path to in-domain dataset")
  parser.add_argument("--target-root", default="data/raw/CRC-VAL-HE-7K", help="Path to out-of-domain dataset")
  parser.add_argument("--out-dir", default="results", help="Directory for output metrics and tables")
  parser.add_argument("--subset", type=int, default=25000, help="Subset size (default 25000; 0 for full 100k)")
  parser.add_argument("--epochs", type=int, default=8, help="Epochs per experiment")
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--experiments", nargs="*", default=None, help="Filter experiment IDs to run (e.g. EXP-01 EXP-02)")
  parser.add_argument("--dry-run", action="store_true", help="Print experiment plan and exit")
  return parser


def main(args: list[str] | None = None) -> None:
  """CLI orchestration runner for histology robustness ablation experiments."""
  parser = build_parser()
  parsed_args = parser.parse_args(args)

  out_path = Path(parsed_args.out_dir)
  out_path.mkdir(parents=True, exist_ok=True)

  exp_list = EXPERIMENTS
  if parsed_args.experiments:
    wanted = set(parsed_args.experiments)
    exp_list = [e for e in exp_list if e["id"] in wanted]

  if parsed_args.dry_run:
    print(f"Plan: {len(exp_list)} experiments queued (subset={parsed_args.subset}, epochs={parsed_args.epochs}, batch={parsed_args.batch_size})")
    for e in exp_list:
      print(f"  {e['id']} | {e['stage']} | backbone={e['backbone']:<13} | norm={e['norm']:<8} | aug={e['aug']:<12} | {e['notes']}")
    return

  import pandas as pd
  import torch

  try:
    from histo_robust.data.splits import prepare_dataset_splits
  except (ImportError, ModuleNotFoundError):
    from run_experiments import prepare_dataset_splits

  from histo_robust.engine.trainer import train_experiment

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Running on device: {device} | Total experiments: {len(exp_list)}")

  splits_dir = out_path / "splits"
  splits, ref_tile = prepare_dataset_splits(
    source_dir=Path(parsed_args.data_root),
    target_dir=Path(parsed_args.target_root),
    out_dir=splits_dir,
    subset_size=parsed_args.subset,
  )

  results = []
  for exp in exp_list:
    res = train_experiment(
      exp=exp,
      splits=splits,
      ref_img_path=ref_tile,
      epochs=parsed_args.epochs,
      batch_size=parsed_args.batch_size,
      device_str=device,
    )
    results.append(res)

  # Output final summary table
  res_df = pd.DataFrame(results)
  csv_file = out_path / "summary_results.csv"
  md_file = out_path / "RESULTS_TABLE.md"

  res_df.to_csv(csv_file, index=False)
  try:
    md_content = res_df.to_markdown(index=False)
  except Exception:
    md_content = res_df.to_string(index=False)

  md_file.write_text(f"# Final Ablation Results\n\n{md_content}\n", encoding="utf-8")

  print("\n" + "=" * 80)
  print("FINAL ABLATION RESULTS TABLE")
  print("=" * 80)
  print(md_content)
  print(f"\n[Artifacts] Wrote {csv_file} and {md_file}")


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
  main()

