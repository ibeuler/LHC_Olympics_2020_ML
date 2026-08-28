"""Particle Transformer classifier for supervised signal/background training.

The model is trained on labelled R&D events, then its signal probability can be
used as an anomaly score on black-box data.

The forward interface matches ``MLPClassifier`` and returns class logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.particle_transformer import ParticleTransformer
from src.models.preprocessing import LHCOPreprocessor


class ParTClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 2100,
        n_particles: int = 700,
        max_particles: int | None = 128,
        num_classes: int = 2,
        embed_dims: list[int] | None = None,
        pair_embed_dims: list[int] | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
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

        self.preprocessor = LHCOPreprocessor(
            n_particles=n_particles,
            max_particles=self.max_particles,
            sort_by_pt=True,
        )

        self.model = ParticleTransformer(
            input_dim=3,
            num_classes=num_classes,
            pair_input_dim=4,
            use_pairwise=self.use_pairwise,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=[],
            trim=True,
            use_amp=use_amp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, vectors, mask = self.preprocessor(x)
        logits = self.model(features, v=vectors, mask=mask)
        return logits
