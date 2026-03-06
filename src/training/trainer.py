"""Training loop and validation for LHC Olympics 2020 models.

Supports both autoencoder (MSE loss) and classifier (CrossEntropy loss)
training through a unified interface.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """All hyper-parameters needed by the training loop."""
    batch_size: int = 512
    lr: float = 1e-3
    epochs: int = 10
    seed: int = 42
    device: str = "cpu"
    model_type: str = "autoencoder"  # "autoencoder" | "classifier"



def _save_loss_log(
    log: List[Dict[str, float]], path: Path
) -> None:
    """Write per-epoch train/val loss to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(log)



def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    model_type: str = "autoencoder",
) -> float:
    """Run one validation pass and return the mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_type == "autoencoder":
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
            else:  # classifier
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)



def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    output_dir: Path,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """Full training loop with checkpointing and logging.

    Parameters
    ----------
    model : nn.Module
        Model to train (SimpleAutoencoder or MLPClassifier).
    train_loader, val_loader : DataLoader
        Training and validation data loaders.
    config : TrainConfig
        Training hyper-parameters.
    output_dir : Path
        Where to save ``best_model.pt`` and ``loss_log.csv``.

    Returns
    -------
    model : nn.Module
        The trained model (best checkpoint loaded).
    loss_log : list[dict]
        Per-epoch train/val losses.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    model = model.to(device)

    if config.model_type == "autoencoder":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    loss_log: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if config.model_type == "autoencoder":
                x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        avg_val_loss = validate(
            model, val_loader, criterion, device, model_type=config.model_type
        )

        loss_log.append(
            {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        print(
            f"Epoch {epoch:>3d}/{config.epochs}  |  "
            f"train_loss: {avg_train_loss:.6f}  |  "
            f"val_loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    torch.save(model.state_dict(), output_dir / "final_model.pt")

    _save_loss_log(loss_log, output_dir / "loss_log.csv")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoint saved to {output_dir / 'best_model.pt'}")

    return model, loss_log
