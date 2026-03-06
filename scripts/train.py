"""CLI entry-point: train a model for LHC Olympics 2020."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import LHCDataset, SyntheticLHCDataset, build_dataloaders
from src.training.trainer import TrainConfig, train
from src.utils.config import load_config, get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a model for LHC Olympics 2020.")
    p.add_argument("--config", type=Path, default=None,
                    help="Path to YAML config file (e.g. configs/config.yaml)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--data", type=Path, default=None,
                    help="Path to HDF5 data file. If not given, synthetic data is used.")
    p.add_argument("--model-type", type=str, default=None,
                    choices=["autoencoder", "classifier"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.config is not None:
        cfg = load_config(args.config)
    else:
        cfg = {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    output_root = cfg.get("outputs", {}).get("root", "outputs")

    model_type = args.model_type or model_cfg.get("type", "autoencoder")
    batch_size = args.batch_size or train_cfg.get("batch_size", 512)
    lr = args.lr or train_cfg.get("lr", 1e-3)
    epochs = args.epochs or train_cfg.get("epochs", 10)
    device = args.device or train_cfg.get("device", "cpu")
    output_dir = args.output or Path(output_root) / "models"

    if args.data is not None:
        print(f"Loading data from {args.data} ...")
        dataset = LHCDataset(args.data)
        input_dim = dataset.input_dim
    else:
        print("No data file specified — using synthetic dataset for demonstration.")
        input_dim = model_cfg.get("input_dim", 128)
        dataset = SyntheticLHCDataset(n_samples=10_000, input_dim=input_dim)

    train_loader, val_loader = build_dataloaders(
        dataset, batch_size=batch_size, seed=cfg.get("seed", 42)
    )

    cfg.setdefault("model", {})
    cfg["model"]["type"] = model_type
    cfg["model"]["input_dim"] = input_dim
    model = get_model(cfg)

    print(f"Model: {model.__class__.__name__}  |  input_dim={input_dim}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")

    tc = TrainConfig(
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
        model_type=model_type,
    )

    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=tc,
        output_dir=Path(output_dir),
    )

    from src.analysis.plotting import plot_loss_curves

    figures_dir = Path(output_root) / "figures"
    train_losses = [e["train_loss"] for e in loss_log]
    val_losses = [e["val_loss"] for e in loss_log]
    plot_loss_curves(train_losses, val_losses, figures_dir / "loss_curves.png")
    print(f"Loss curves saved to {figures_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
