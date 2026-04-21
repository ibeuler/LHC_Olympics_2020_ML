"""Configuration loading and model factory helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from src.models.autoencoder import SimpleAutoencoder, VariationalAutoencoder
from src.models.classifier import MLPClassifier


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
    elif model_type == "vae":
        latent_dim = model_cfg.get("latent_dim", 16)
        hidden_dim = model_cfg.get("hidden_dim", 256)
        beta = model_cfg.get("beta", 1.0)
        return VariationalAutoencoder(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, beta=beta
        )
    elif model_type == "classifier":
        hidden_dim = model_cfg.get("hidden_dim", 256)
        num_classes = model_cfg.get("num_classes", 2)
        return MLPClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. Choose 'autoencoder', 'vae', or 'classifier'."
        )
