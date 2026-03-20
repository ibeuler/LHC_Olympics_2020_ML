"""Version C: Particle Transformer Classifier for supervised transfer learning.

Train on R&D dataset with truth labels, then use classifier score as anomaly
score on black box data. Classifier score = softmax probability of signal class.

Forward contract: x -> logits — same as MLPClassifier.
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
        num_classes: int = 2,
        embed_dims: list[int] | None = None,
        pair_embed_dims: list[int] | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
    ) -> None:
        super().__init__()

        if embed_dims is None:
            embed_dims = [128, 512, 128]
        if pair_embed_dims is None:
            pair_embed_dims = [64, 64, 64]

        self.input_dim = input_dim
        self.n_particles = n_particles

        self.preprocessor = LHCOPreprocessor(n_particles=n_particles)

        self.model = ParticleTransformer(
            input_dim=3,
            num_classes=num_classes,
            pair_input_dim=4,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=[],
            trim=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, vectors, mask = self.preprocessor(x)
        logits = self.model(features, v=vectors, mask=mask)
        return logits
