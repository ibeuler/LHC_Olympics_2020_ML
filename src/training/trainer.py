from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"


def train(*, config: TrainConfig, output_dir: Path) -> None:
    """Placeholder training entry.

    Intended to: build dataloaders, create model/optimizer, run epochs, save weights/logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Implement training once dataset + model choice are finalized.")


def validate() -> None:
    raise NotImplementedError
