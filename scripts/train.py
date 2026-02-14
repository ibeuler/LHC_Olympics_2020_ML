from __future__ import annotations

import argparse
from pathlib import Path

from src.training.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model for LHC Olympics 2020.")
    p.add_argument("--output", type=Path, default=Path("outputs"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
    train(config=cfg, output_dir=args.output / "models")


if __name__ == "__main__":
    main()
