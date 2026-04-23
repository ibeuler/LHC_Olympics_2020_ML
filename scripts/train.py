"""CLI entry-point: train a model for LHC Olympics 2020."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/train.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import (
    LHCDataset, SyntheticLHCDataset, SyntheticParticleDataset,
    BackgroundOnlyDataset, build_dataloaders,
)
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
                    choices=["autoencoder", "vae", "classifier", "part_ae"])
    p.add_argument("--n-particles", type=int, default=None,
                    help="Max particles per event for part_ae (overrides config)")
    p.add_argument("--background-only", action="store_true",
                    help="Train on background events only (recommended for AE/VAE/part_ae)")
    p.add_argument("--wandb-project", type=str, default=None,
                    help="W&B project name to enable experiment tracking")
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

    n_particles        = args.n_particles or model_cfg.get("n_particles", 200)
    n_particle_features = model_cfg.get("n_particle_features", 3)

    if args.data is not None:
        print(f"Loading data from {args.data} ...")
        if model_type == "part_ae":
            dataset = LHCDataset(
                args.data,
                particle_mode=True,
                n_particles=n_particles,
                n_particle_features=n_particle_features,
            )
        else:
            dataset = LHCDataset(args.data)
        input_dim = dataset.input_dim
    else:
        print("No data file specified — using synthetic dataset for demonstration.")
        if model_type == "part_ae":
            dataset = SyntheticParticleDataset(
                n_samples=2_000,
                n_particles=n_particles,
                n_features=n_particle_features,
            )
            input_dim = n_particle_features
        else:
            input_dim = model_cfg.get("input_dim", 128)
            dataset = SyntheticLHCDataset(n_samples=10_000, input_dim=input_dim)

    if args.background_only and model_type in ("autoencoder", "vae", "part_ae"):
        print("Background-only mode: filtering to label==0 events for training.")
        dataset = BackgroundOnlyDataset(dataset)
        print(f"  {len(dataset)} background events retained.")

    train_loader, val_loader, _ = build_dataloaders(
        dataset, batch_size=batch_size, seed=cfg.get("seed", 42)
    )

    cfg.setdefault("model", {})
    cfg["model"]["type"] = model_type
    cfg["model"]["input_dim"] = input_dim
    if model_type == "part_ae":
        cfg["model"]["n_particles"] = n_particles
        cfg["model"]["n_particle_features"] = n_particle_features
    model = get_model(cfg)

    print(f"Model: {model.__class__.__name__}  |  input_dim={input_dim}")
    print(f"Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")

    tc = TrainConfig(
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        device=device,
        model_type=model_type,
        wandb_project=args.wandb_project,
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
