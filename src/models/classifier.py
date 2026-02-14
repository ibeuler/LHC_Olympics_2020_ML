from __future__ import annotations

try:
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    nn = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def ensure_torch_available() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch is required for models. Install it per your CUDA/CPU setup."
        ) from _TORCH_IMPORT_ERROR
