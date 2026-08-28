"""Particle Transformer autoencoder for unsupervised anomaly detection.

ParT compresses each event into its CLS representation, and a small MLP decodes
that representation back to the original particle features. Per-event MSE is
used as the anomaly score.

The forward interface matches ``SimpleAutoencoder``: ``x -> (x_hat, z)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.particle_transformer import ParticleTransformer
from src.models.preprocessing import LHCOPreprocessor


class ParTAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 2100,
        n_particles: int = 700,
        max_particles: int | None = 128,
        embed_dims: list[int] | None = None,
        pair_embed_dims: list[int] | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        decoder_hidden_dim: int = 256,
        use_pairwise: bool = True,
        use_amp: bool = True,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [128, 512, 128]
        if pair_embed_dims is None:
            pair_embed_dims = [64, 64, 64]

        self.input_dim = input_dim
        self.n_particles = n_particles
        self.max_particles = max_particles if max_particles is not None else n_particles
        self.use_pairwise = bool(use_pairwise)
        latent_dim = embed_dims[-1]

        self.preprocessor = LHCOPreprocessor(
            n_particles=n_particles,
            max_particles=self.max_particles,
            sort_by_pt=True,
        )

        self.encoder = ParticleTransformer(
            input_dim=3,
            num_classes=None,
            pair_input_dim=4,
            use_pairwise=self.use_pairwise,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=None,
            trim=True,
            use_amp=use_amp,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        features, vectors, mask = self.preprocessor(x)
        z = self.encoder(features, v=vectors, mask=mask)
        x_hat = self.decoder(z)
        return x_hat, z
