"""Training engine and CLI orchestration runner module."""

from histo_robust.engine.runner import build_parser, main


def __getattr__(name: str):
  if name == "train_experiment":
    from histo_robust.engine.trainer import train_experiment
    return train_experiment
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["build_parser", "main", "train_experiment"]
