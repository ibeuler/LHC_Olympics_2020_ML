"""Version A: Particle Transformer Autoencoder for unsupervised anomaly detection.

Uses ParT as encoder (CLS token = latent representation) and a simple MLP
decoder for reconstruction. Anomaly score = reconstruction error (MSE).

Forward contract: x -> (x_hat, z) — same as SimpleAutoencoder.
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
        embed_dims: list[int] | None = None,
        pair_embed_dims: list[int] | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        decoder_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [128, 512, 128]
        if pair_embed_dims is None:
            pair_embed_dims = [64, 64, 64]

        self.input_dim = input_dim
        self.n_particles = n_particles
        latent_dim = embed_dims[-1]

        self.preprocessor = LHCOPreprocessor(n_particles=n_particles)

        self.encoder = ParticleTransformer(
            input_dim=3,
            num_classes=None,
            pair_input_dim=4,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=None,
            trim=True,
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
