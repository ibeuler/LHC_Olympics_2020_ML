"""Configuration loading and model factory helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.models.autoencoder import SimpleAutoencoder
from src.models.classifier import MLPClassifier
from src.models.part_autoencoder import ParTAutoencoder
from src.models.part_classifier import ParTClassifier


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dict.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file (e.g. ``configs/config.yaml``).

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    return cfg


def get_model(config: Dict[str, Any]):
    """Instantiate the correct model based on *config['model']['type']*.

    Parameters
    ----------
    config : dict
        Full project configuration dict (as returned by :func:`load_config`).

    Returns
    -------
    torch.nn.Module
        Either a :class:`SimpleAutoencoder` or :class:`MLPClassifier`.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "autoencoder").lower()
    input_dim = model_cfg.get("input_dim", 128)

    if model_type == "autoencoder":
        latent_dim = model_cfg.get("latent_dim", 16)
        return SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    elif model_type == "classifier":
        hidden_dim = model_cfg.get("hidden_dim", 256)
        num_classes = model_cfg.get("num_classes", 2)
        return MLPClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    elif model_type == "part_autoencoder":
        return ParTAutoencoder(
            input_dim=input_dim,
            n_particles=model_cfg.get("n_particles", input_dim // 3),
            embed_dims=model_cfg.get("embed_dims", [128, 512, 128]),
            pair_embed_dims=model_cfg.get("pair_embed_dims", [64, 64, 64]),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 8),
            num_cls_layers=model_cfg.get("num_cls_layers", 2),
            decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        )
    elif model_type == "part_classifier":
        return ParTClassifier(
            input_dim=input_dim,
            n_particles=model_cfg.get("n_particles", input_dim // 3),
            num_classes=model_cfg.get("num_classes", 2),
            embed_dims=model_cfg.get("embed_dims", [128, 512, 128]),
            pair_embed_dims=model_cfg.get("pair_embed_dims", [64, 64, 64]),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 8),
            num_cls_layers=model_cfg.get("num_cls_layers", 2),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Choose 'autoencoder', 'classifier', 'part_autoencoder', or 'part_classifier'."
        )
