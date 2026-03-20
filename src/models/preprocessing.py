"""Preprocessing: convert flat LHCO2020 vectors to Particle Transformer input format.

LHCO2020 events are stored as flat (2100,) vectors = 700 particles x (pT, eta, phi).
ParT expects three separate tensors: features (N, C, P), 4-vectors (N, 4, P), mask (N, 1, P).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LHCOPreprocessor(nn.Module):
    """Convert flat (batch, n_particles*3) tensors to ParT input format.

    This module has no learnable parameters. It performs deterministic
    coordinate transformations and is used inside ParT wrapper models.
    """

    def __init__(self, n_particles: int = 700, eps: float = 1e-8) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.eps = eps

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        particles = x.view(batch_size, self.n_particles, 3)
        pt = particles[:, :, 0]
        eta = particles[:, :, 1]
        phi = particles[:, :, 2]

        mask = (pt > 0).unsqueeze(1).float()

        log_pt = torch.log(pt.clamp(min=self.eps))
        features = torch.stack([log_pt, eta, phi], dim=1)
        features = features * mask

        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta.clamp(-10, 10))
        energy = pt * torch.cosh(eta.clamp(-10, 10))
        vectors = torch.stack([px, py, pz, energy], dim=1)
        vectors = vectors * mask

        return features, vectors, mask
