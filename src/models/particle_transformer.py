"""Particle Transformer (ParT) for jet tagging and anomaly detection.

Reference: Qu, Li & Qian, "Particle Transformer for Jet Tagging",
           arXiv:2202.03772, ICML 2022.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_features(x: torch.Tensor) -> torch.Tensor:
    """Compute (ln Δ, ln kT, ln z, ln m²) for every particle pair.

    Parameters
    ----------
    x : Tensor (B, N, F), F >= 3
        Particle features; columns 0-2 are (pT, eta, phi).

    Returns
    -------
    Tensor (B, N, N, 4)
    """
    eps = 1e-8
    pt  = x[..., 0]   # (B, N)
    eta = x[..., 1]
    phi = x[..., 2]

    d_eta = eta.unsqueeze(2) - eta.unsqueeze(1)   # (B, N, N)
    d_phi = phi.unsqueeze(2) - phi.unsqueeze(1)
    d_phi = torch.atan2(torch.sin(d_phi), torch.cos(d_phi))

    delta = torch.sqrt(d_eta ** 2 + d_phi ** 2 + eps)

    pt_i = pt.unsqueeze(2)   # (B, N, 1) -> broadcasts to (B, N, N)
    pt_j = pt.unsqueeze(1)   # (B, 1, N)

    k_t = torch.minimum(pt_i, pt_j) * delta
    z   = torch.minimum(pt_i, pt_j) / (pt_i + pt_j + eps)
    m2  = (2.0 * pt_i * pt_j * (torch.cosh(d_eta) - torch.cos(d_phi))).clamp(min=eps)

    return torch.stack([
        torch.log(delta),
        torch.log(k_t + eps),
        torch.log(z   + eps),
        torch.log(m2),
    ], dim=-1)  # (B, N, N, 4)


class ParticleEmbedding(nn.Module):
    """3-layer MLP: C -> 128 -> 512 -> d with GELU and LayerNorm."""

    def __init__(self, in_features: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractionEmbedding(nn.Module):
    """Pointwise MLP: 4 -> 64 -> 64 -> 64 -> d' with GELU and BatchNorm."""

    def __init__(self, interaction_dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, interaction_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, N, N, 4)
        B, N, _, C = u.shape
        out = self.net(u.reshape(B * N * N, C))
        return out.reshape(B, N, N, -1)  # (B, N, N, d')


class PMHAttention(nn.Module):
    """Physics-augmented Multi-Head Attention.

    The interaction matrix U is projected to one value per head and added
    as an additive bias to the pre-softmax attention logits.
    """

    def __init__(self, embed_dim: int, n_heads: int, interaction_dim: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.u_proj = nn.Linear(interaction_dim, n_heads)

    def forward(
        self,
        x: torch.Tensor,
        U: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        # (B, N, N, n_heads) -> (B*n_heads, N, N)
        attn_bias = (
            self.u_proj(U)
            .permute(0, 3, 1, 2)
            .contiguous()
            .view(B * self.n_heads, N, N)
        )
        # Fold padding mask into the float attn_bias to avoid dtype mismatch warnings.
        # Padded key positions get -inf, which zeroes them out after softmax.
        if key_padding_mask is not None:
            pad = (
                torch.zeros(B, N, dtype=attn_bias.dtype, device=attn_bias.device)
                .masked_fill(key_padding_mask, float("-inf"))         # (B, N)
                .unsqueeze(1).unsqueeze(2)                            # (B, 1, 1, N)
                .expand(B, self.n_heads, N, N)
                .reshape(B * self.n_heads, N, N)
            )
            attn_bias = attn_bias + pad
        out, _ = self.mha(x, x, x, attn_mask=attn_bias, need_weights=False)
        return out


class ParticleAttentionBlock(nn.Module):
    """NormFormer-style block: P-MHA then 2-layer FFN, each with pre+post LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        interaction_dim: int = 8,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.pre_attn_norm  = nn.LayerNorm(embed_dim)
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = PMHAttention(embed_dim, n_heads, interaction_dim)
        self.pre_ffn_norm  = nn.LayerNorm(embed_dim)
        self.post_ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        U: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), U, key_padding_mask))
        x = x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))
        return x


class ClassAttentionBlock(nn.Module):
    """Class token attends to full sequence [class, particles] with standard MHA."""

    def __init__(self, embed_dim: int, n_heads: int, ffn_ratio: int = 4) -> None:
        super().__init__()
        self.pre_norm       = nn.LayerNorm(embed_dim)
        self.post_attn_norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.pre_ffn_norm  = nn.LayerNorm(embed_dim)
        self.post_ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * ffn_ratio, embed_dim),
        )

    def forward(
        self,
        cls: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Q = class token only; K = V = [class, particles]
        z = torch.cat([cls, x], dim=1)         # (B, N+1, d)
        z_norm = self.pre_norm(z)
        q_norm = z_norm[:, :1]                 # (B, 1, d)

        kv_mask = None
        if key_padding_mask is not None:
            cls_valid = torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device)
            kv_mask = torch.cat([cls_valid, key_padding_mask], dim=1)  # (B, N+1)

        attn_out, _ = self.mha(q_norm, z_norm, z_norm, key_padding_mask=kv_mask, need_weights=False)
        cls = cls + self.post_attn_norm(attn_out)
        cls = cls + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(cls)))
        return cls


class ParticleTransformer(nn.Module):
    """ParT encoder: particle embedding → L particle attention blocks → M class attention blocks.

    Returns (class_token, particle_embeddings):
        class_token         : (B, embed_dim) — global jet representation
        particle_embeddings : (B, N, embed_dim) — per-particle contextual embeddings
    """

    def __init__(
        self,
        in_features: int = 3,
        embed_dim: int = 128,
        interaction_dim: int = 8,
        n_heads: int = 8,
        n_layers: int = 8,
        n_class_layers: int = 2,
    ) -> None:
        super().__init__()
        self.particle_embed    = ParticleEmbedding(in_features, embed_dim)
        self.interaction_embed = InteractionEmbedding(interaction_dim)
        self.part_blocks = nn.ModuleList([
            ParticleAttentionBlock(embed_dim, n_heads, interaction_dim)
            for _ in range(n_layers)
        ])
        self.class_blocks = nn.ModuleList([
            ClassAttentionBlock(embed_dim, n_heads)
            for _ in range(n_class_layers)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        U = self.interaction_embed(pairwise_features(x))  # (B, N, N, d')
        h = self.particle_embed(x)                        # (B, N, d)

        for block in self.part_blocks:
            h = block(h, U, padding_mask)

        cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, d)
        for block in self.class_blocks:
            cls = block(cls, h, padding_mask)

        return self.norm(cls.squeeze(1)), h  # (B, d), (B, N, d)


class ParticleTransformerAE(nn.Module):
    """ParT autoencoder for unsupervised anomaly detection.

    Encodes each particle to a contextual embedding via the ParT encoder, then
    decodes per-particle features from those embeddings independently.  The
    anomaly score is the mean per-particle reconstruction MSE (padding excluded).

    forward() returns (x_hat, z) matching the SimpleAutoencoder signature:
        x_hat : (B, N, F) — reconstructed particle features
        z     : (B, embed_dim) — global class-token embedding
    """

    def __init__(
        self,
        in_features: int = 3,
        embed_dim: int = 128,
        interaction_dim: int = 8,
        n_heads: int = 8,
        n_layers: int = 8,
        n_class_layers: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = ParticleTransformer(
            in_features=in_features,
            embed_dim=embed_dim,
            interaction_dim=interaction_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            n_class_layers=n_class_layers,
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, in_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z, h = self.encoder(x, padding_mask)  # (B, d), (B, N, d)
        x_hat = self.decoder(h)               # (B, N, F)
        return x_hat, z

    def anomaly_score(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean per-particle MSE, padding excluded. Shape: (B,)."""
        mse = ((x_hat - x) ** 2).mean(dim=-1)   # (B, N)
        if padding_mask is not None:
            valid = (~padding_mask).float()
            return (mse * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return mse.mean(dim=1)


def part_ae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar MSE reconstruction loss for ParT-AE, ignoring padded particles."""
    loss = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)  # (B, N)
    if padding_mask is not None:
        valid = (~padding_mask).float()
        return (loss * valid).sum() / valid.sum().clamp(min=1.0)
    return loss.mean()
