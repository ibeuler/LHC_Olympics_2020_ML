from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class VariationalAutoencoder(nn.Module):
    """VAE with reparameterisation trick.

    forward() returns (x_hat, mu, log_var).
    Use vae_loss() to compute the ELBO. The reconstruction term alone is the
    anomaly score — events with high MSE are anomalous.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.fc_enc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = F.relu(self.fc_enc1(x))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu, log_var):
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at eval time

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def vae_loss(self, x, x_hat, mu, log_var) -> torch.Tensor:
        """ELBO loss: mean reconstruction MSE + beta * KL divergence."""
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + self.beta * kl

    def anomaly_score(self, x, x_hat) -> torch.Tensor:
        """Per-sample reconstruction MSE (use as anomaly score at eval time)."""
        return ((x_hat - x) ** 2).mean(dim=1)


def ensure_torch_available() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch is required for models. Install it per your CUDA/CPU setup."
        ) from _TORCH_IMPORT_ERROR
