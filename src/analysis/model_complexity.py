"""Compare model sizes for the ParT pairwise-feature ablation."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch.nn as nn

from src.models.autoencoder import SimpleAutoencoder
from src.models.part_autoencoder import ParTAutoencoder


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count all parameters and the subset updated during training."""
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def parameter_count_table(models: Mapping[str, nn.Module]) -> pd.DataFrame:
    """Return a readable parameter-count table for a set of named models."""
    rows = []
    for name, model in models.items():
        row = {"model": name, **count_parameters(model)}
        rows.append(row)
    return pd.DataFrame(rows, columns=["model", "total_parameters", "trainable_parameters"])


def autoencoder_ablation_parameter_table(
    *,
    input_dim: int = 2100,
    latent_dim: int = 16,
    n_particles: int | None = None,
    max_particles: int | None = 128,
    embed_dims: list[int] | None = None,
    pair_embed_dims: list[int] | None = None,
    num_heads: int = 8,
    num_layers: int = 8,
    num_cls_layers: int = 2,
    decoder_hidden_dim: int = 256,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compare SimpleAE with the two ParTAE ablation variants.

    Both ParTAE models share the same transformer and decoder settings. The
    no-pairwise variant is the only one that omits the U-matrix embedding.
    """
    particles = n_particles if n_particles is not None else input_dim // 3
    common = dict(
        input_dim=input_dim,
        n_particles=particles,
        max_particles=max_particles,
        embed_dims=embed_dims,
        pair_embed_dims=pair_embed_dims,
        num_heads=num_heads,
        num_layers=num_layers,
        num_cls_layers=num_cls_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        use_amp=False,
    )
    table = parameter_count_table(
        {
            "SimpleAE": SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim),
            "ParTAE (without pairwise)": ParTAutoencoder(**common, use_pairwise=False),
            "ParTAE (with pairwise)": ParTAutoencoder(**common, use_pairwise=True),
        }
    )
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
    return table
