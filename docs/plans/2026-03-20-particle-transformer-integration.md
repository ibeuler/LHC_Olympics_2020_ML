# Particle Transformer Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two ParT-based models to the LHCO2020 pipeline — a ParT Autoencoder (Version A, unsupervised anomaly detection) and a ParT Classifier (Version C, supervised transfer learning) — while keeping the existing SimpleAutoencoder and MLPClassifier fully functional.

**Architecture:** Both ParT wrappers accept flat `(batch, 2100)` tensors (same as existing models) and internally preprocess them into ParT's multi-tensor format `(features, 4-vectors, mask)`. This means the dataset, trainer input format, and existing tests remain unchanged. The wrappers conform to the existing forward signature contracts: autoencoder returns `(x_hat, z)`, classifier returns `logits`.

**Tech Stack:** PyTorch (already in repo), no new dependencies. ParT core model extracted from `weaver-core` (single file, one import line patched).

---

## Shared Context

### Current Interface Contracts

**Autoencoder contract** (used by trainer when `model_type` starts with `"autoencoder"`):
- Input: `x` of shape `(batch, input_dim)` — flat float32 tensor
- Output: `(x_hat, z)` — reconstruction and latent
- Batch unpacking: `x = batch[0]` (ignores labels if present)
- Loss: `MSELoss(x_hat, x)`

**Classifier contract** (used by trainer for all other `model_type` values):
- Input: `x` of shape `(batch, input_dim)` — flat float32 tensor
- Output: `logits` of shape `(batch, num_classes)`
- Batch unpacking: `x, y = batch` (requires labels)
- Loss: `CrossEntropyLoss(logits, y)`

### LHCO2020 Data Layout

Each event is a flat vector of 2100 floats = 700 particles x 3 features (pT, eta, phi). Padded particles have pT == 0.

### ParT Input Requirements

- `x`: `(N, C, P)` — particle features (C channels, P particles)
- `v`: `(N, 4, P)` — Lorentz 4-vectors (px, py, pz, E)
- `mask`: `(N, 1, P)` — binary mask (1 = real particle, 0 = padded)

### Key Design Decision

The `LHCOPreprocessor` module lives inside the wrapper models. It converts flat `(batch, 2100)` input into ParT's multi-tensor format. This means:
- No changes to `dataset.py`
- No changes to DataLoader batch format
- Trainer only needs new `elif` branches for model dispatch (loss/forward are already correct)
- Existing models and tests are completely untouched

### Physics: (pT, eta, phi) to 4-vectors

Massless particle approximation:
```
px = pT * cos(phi)
py = pT * sin(phi)
pz = pT * sinh(eta)
E  = pT * cosh(eta)
```

Particle features for ParT embedding (3 channels):
```
feature[0] = log(pT + 1e-8)   # log-scale pT
feature[1] = eta              # pseudorapidity
feature[2] = phi              # azimuthal angle
```

### Test Strategy

All ParT tests use `input_dim` divisible by 3 (e.g. 63 = 21 particles x 3) and a tiny ParT config (2 layers, 2 heads, embed_dim=32) so they run fast on CPU. Existing tests with `input_dim=64` are untouched.

---

## Task 1: Extract ParT Core Model

**Files:**
- Create: `src/models/particle_transformer.py`

**Step 1: Write the ParT core model file**

Extract the `ParticleTransformer` class and its dependencies from `weaver-core` (commit: main branch). Include only what we need:
- Helper functions: `delta_phi`, `delta_r2`, `to_pt2`, `to_m2`, `atan2`, `to_ptrapphim`, `pairwise_lv_fts`, `trunc_normal_`
- Classes: `SequenceTrimmer`, `Embed`, `PairEmbed`, `Block`, `ParticleTransformer`
- Do NOT include: `ParticleTransformerTagger`, `ParticleTransformerTaggerWithExtraPairFeatures`, `build_sparse_tensor`, `boost`, `p3_norm`

**Patch the only external dependency:**
```python
# REPLACE:
from weaver.utils.logger import _logger
# WITH:
import logging
_logger = logging.getLogger(__name__)
```

No other changes to the model code. Keep all original logic, comments, and defaults intact.

**Step 2: Verify import works**

Run: `/opt/homebrew/bin/python3.10 -c "from src.models.particle_transformer import ParticleTransformer; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/models/particle_transformer.py
git commit -m "feat: extract Particle Transformer core from weaver-core"
```

---

## Task 2: Create Preprocessing Module

**Files:**
- Create: `src/models/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Step 1: Write the failing test**

```python
"""Tests for LHCOPreprocessor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.preprocessing import LHCOPreprocessor

def test_preprocessor_shapes():
    """Verify output tensor shapes for a known input."""
    n_particles = 21
    input_dim = n_particles * 3  # 63
    batch_size = 4
    prep = LHCOPreprocessor(n_particles=n_particles)
    x = torch.randn(batch_size, input_dim)
    features, vectors, mask = prep(x)
    assert features.shape == (batch_size, 3, n_particles), f"features: {features.shape}"
    assert vectors.shape == (batch_size, 4, n_particles), f"vectors: {vectors.shape}"
    assert mask.shape == (batch_size, 1, n_particles), f"mask: {mask.shape}"
    print("PASSED: test_preprocessor_shapes")

def test_preprocessor_mask():
    """Verify mask correctly identifies padded particles (pT == 0)."""
    n_particles = 10
    input_dim = n_particles * 3
    prep = LHCOPreprocessor(n_particles=n_particles)
    x = torch.zeros(2, input_dim)
    # Set first 3 particles with nonzero pT in each event
    for i in range(3):
        x[0, i * 3] = 5.0  # pT > 0
        x[1, i * 3] = 3.0
    _, _, mask = prep(x)
    # First 3 particles should be unmasked, rest masked
    assert mask[0, 0, :3].sum() == 3, "First 3 should be real"
    assert mask[0, 0, 3:].sum() == 0, "Rest should be padded"
    print("PASSED: test_preprocessor_mask")

def test_preprocessor_4vectors():
    """Verify 4-vector computation: px=pT*cos(phi), py=pT*sin(phi), etc."""
    import math
    n_particles = 1
    prep = LHCOPreprocessor(n_particles=n_particles)
    # Single particle: pT=10, eta=0, phi=0
    x = torch.tensor([[10.0, 0.0, 0.0]])  # (1, 3)
    _, vectors, _ = prep(x)
    # px=10*cos(0)=10, py=10*sin(0)=0, pz=10*sinh(0)=0, E=10*cosh(0)=10
    assert torch.allclose(vectors[0, 0, 0], torch.tensor(10.0), atol=1e-5), "px"
    assert torch.allclose(vectors[0, 1, 0], torch.tensor(0.0), atol=1e-5), "py"
    assert torch.allclose(vectors[0, 2, 0], torch.tensor(0.0), atol=1e-5), "pz"
    assert torch.allclose(vectors[0, 3, 0], torch.tensor(10.0), atol=1e-5), "E"
    print("PASSED: test_preprocessor_4vectors")

if __name__ == "__main__":
    test_preprocessor_shapes()
    test_preprocessor_mask()
    test_preprocessor_4vectors()
    print("\nAll preprocessing tests PASSED")
```

**Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.10 tests/test_preprocessing.py`
Expected: `ModuleNotFoundError: No module named 'src.models.preprocessing'`

**Step 3: Write the implementation**

```python
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
        """
        Parameters
        ----------
        x : Tensor of shape (N, n_particles * 3)
            Flat LHCO2020 event vector. Layout: [pT_0, eta_0, phi_0, pT_1, ...].

        Returns
        -------
        features : Tensor (N, 3, P) — log(pT+eps), eta, phi
        vectors  : Tensor (N, 4, P) — px, py, pz, E (massless approximation)
        mask     : Tensor (N, 1, P) — 1 for real particles (pT > 0), 0 for padding
        """
        batch_size = x.size(0)
        # (N, P, 3)
        particles = x.view(batch_size, self.n_particles, 3)
        pt = particles[:, :, 0]    # (N, P)
        eta = particles[:, :, 1]   # (N, P)
        phi = particles[:, :, 2]   # (N, P)

        # Mask: real particles have pT > 0
        mask = (pt > 0).unsqueeze(1).float()  # (N, 1, P)

        # Particle features for embedding: (N, 3, P)
        log_pt = torch.log(pt.clamp(min=self.eps))
        features = torch.stack([log_pt, eta, phi], dim=1)  # (N, 3, P)
        # Zero out padded particles
        features = features * mask

        # 4-vectors (massless): (N, 4, P)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta.clamp(-10, 10))  # clamp to avoid overflow
        energy = pt * torch.cosh(eta.clamp(-10, 10))
        vectors = torch.stack([px, py, pz, energy], dim=1)  # (N, 4, P)
        vectors = vectors * mask

        return features, vectors, mask
```

**Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.10 tests/test_preprocessing.py`
Expected: `All preprocessing tests PASSED`

**Step 5: Commit**

```bash
git add src/models/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add LHCOPreprocessor for ParT input conversion"
```

---

## Task 3: Create ParTAutoencoder (Version A)

**Files:**
- Create: `src/models/part_autoencoder.py`
- Test: `tests/test_part_autoencoder.py`

**Step 1: Write the failing test**

```python
"""Tests for ParTAutoencoder (Version A)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.part_autoencoder import ParTAutoencoder

# Tiny config for fast CPU tests
TINY_CFG = dict(
    input_dim=63,        # 21 particles x 3
    n_particles=21,
    embed_dims=[32, 32],
    pair_embed_dims=[16, 16],
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
    decoder_hidden_dim=64,
)

def test_forward_shape():
    """Verify forward returns (x_hat, z) with correct shapes."""
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    x_hat, z = model(x)
    assert x_hat.shape == (4, 63), f"x_hat: {x_hat.shape}"
    embed_dim = TINY_CFG["embed_dims"][-1]
    assert z.shape == (4, embed_dim), f"z: {z.shape}"
    print("PASSED: test_forward_shape")

def test_gradient_flow():
    """Verify gradients flow through the entire model."""
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    x_hat, z = model(x)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters())
    assert n_grad > 0, "No gradients!"
    print(f"PASSED: test_gradient_flow ({n_grad}/{n_total} params have gradients)")

def test_autoencoder_contract():
    """Verify it matches the existing autoencoder trainer contract."""
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    result = model(x)
    assert isinstance(result, tuple) and len(result) == 2, "Must return (x_hat, z)"
    x_hat, z = result
    loss = torch.nn.functional.mse_loss(x_hat, x)
    assert loss.requires_grad, "Loss must be differentiable"
    print("PASSED: test_autoencoder_contract")

if __name__ == "__main__":
    test_forward_shape()
    test_gradient_flow()
    test_autoencoder_contract()
    print("\nAll ParTAutoencoder tests PASSED")
```

**Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.10 tests/test_part_autoencoder.py`
Expected: `ModuleNotFoundError: No module named 'src.models.part_autoencoder'`

**Step 3: Write the implementation**

```python
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
        # ParT encoder params
        embed_dims: list[int] | None = None,
        pair_embed_dims: list[int] | None = None,
        num_heads: int = 8,
        num_layers: int = 8,
        num_cls_layers: int = 2,
        # Decoder params
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

        # ParT encoder: fc=None means forward returns CLS token embedding directly
        self.encoder = ParticleTransformer(
            input_dim=3,  # (log_pt, eta, phi)
            num_classes=None,
            pair_input_dim=4,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=None,  # no classification head -> returns CLS token
            trim=True,
        )

        # Simple MLP decoder: latent -> reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor (N, input_dim) — flat LHCO2020 event vector

        Returns
        -------
        x_hat : Tensor (N, input_dim) — reconstruction
        z     : Tensor (N, latent_dim) — CLS token embedding (latent)
        """
        features, vectors, mask = self.preprocessor(x)
        z = self.encoder(features, v=vectors, mask=mask)  # (N, latent_dim)
        x_hat = self.decoder(z)  # (N, input_dim)
        return x_hat, z
```

**Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.10 tests/test_part_autoencoder.py`
Expected: `All ParTAutoencoder tests PASSED`

**Step 5: Commit**

```bash
git add src/models/part_autoencoder.py tests/test_part_autoencoder.py
git commit -m "feat: add ParTAutoencoder (Version A) for unsupervised anomaly detection"
```

---

## Task 4: Create ParTClassifier (Version C)

**Files:**
- Create: `src/models/part_classifier.py`
- Test: `tests/test_part_classifier.py`

**Step 1: Write the failing test**

```python
"""Tests for ParTClassifier (Version C)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.part_classifier import ParTClassifier

TINY_CFG = dict(
    input_dim=63,
    n_particles=21,
    num_classes=2,
    embed_dims=[32, 32],
    pair_embed_dims=[16, 16],
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
)

def test_forward_shape():
    """Verify forward returns logits with correct shape."""
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    assert logits.shape == (4, 2), f"logits: {logits.shape}"
    print("PASSED: test_forward_shape")

def test_gradient_flow():
    """Verify gradients flow through the entire model."""
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "No gradients!"
    print(f"PASSED: test_gradient_flow ({n_grad} params have gradients)")

def test_classifier_contract():
    """Verify it matches the existing classifier trainer contract."""
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    assert isinstance(logits, torch.Tensor), "Must return a single tensor"
    assert logits.dim() == 2, f"Expected 2D, got {logits.dim()}D"
    assert logits.size(1) == 2, f"Expected 2 classes, got {logits.size(1)}"
    print("PASSED: test_classifier_contract")

if __name__ == "__main__":
    test_forward_shape()
    test_gradient_flow()
    test_classifier_contract()
    print("\nAll ParTClassifier tests PASSED")
```

**Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.10 tests/test_part_classifier.py`
Expected: `ModuleNotFoundError: No module named 'src.models.part_classifier'`

**Step 3: Write the implementation**

```python
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
        # ParT params
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
            input_dim=3,  # (log_pt, eta, phi)
            num_classes=num_classes,
            pair_input_dim=4,
            embed_dims=embed_dims,
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            fc_params=[],  # direct Linear(embed_dim, num_classes)
            trim=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, input_dim) — flat LHCO2020 event vector

        Returns
        -------
        logits : Tensor (N, num_classes)
        """
        features, vectors, mask = self.preprocessor(x)
        logits = self.model(features, v=vectors, mask=mask)  # (N, num_classes)
        return logits
```

**Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.10 tests/test_part_classifier.py`
Expected: `All ParTClassifier tests PASSED`

**Step 5: Commit**

```bash
git add src/models/part_classifier.py tests/test_part_classifier.py
git commit -m "feat: add ParTClassifier (Version C) for supervised transfer learning"
```

---

## Task 5: Update Config Factory

**Files:**
- Modify: `src/utils/config.py:9-10` (imports), `src/utils/config.py:47-63` (get_model)

**Step 1: Write the failing test**

```python
"""Test config factory recognizes new model types."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_model
from src.models.part_autoencoder import ParTAutoencoder
from src.models.part_classifier import ParTClassifier

def test_part_autoencoder_factory():
    cfg = {
        "model": {
            "type": "part_autoencoder",
            "input_dim": 63,
            "n_particles": 21,
            "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16],
            "num_heads": 2,
            "num_layers": 2,
            "num_cls_layers": 1,
            "decoder_hidden_dim": 64,
        }
    }
    model = get_model(cfg)
    assert isinstance(model, ParTAutoencoder), f"Got {type(model)}"
    print("PASSED: test_part_autoencoder_factory")

def test_part_classifier_factory():
    cfg = {
        "model": {
            "type": "part_classifier",
            "input_dim": 63,
            "n_particles": 21,
            "num_classes": 2,
            "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16],
            "num_heads": 2,
            "num_layers": 2,
            "num_cls_layers": 1,
        }
    }
    model = get_model(cfg)
    assert isinstance(model, ParTClassifier), f"Got {type(model)}"
    print("PASSED: test_part_classifier_factory")

if __name__ == "__main__":
    test_part_autoencoder_factory()
    test_part_classifier_factory()
    print("\nAll config factory tests PASSED")
```

Save as `tests/test_config_part.py`.

**Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/python3.10 tests/test_config_part.py`
Expected: `ValueError: Unknown model type: 'part_autoencoder'`

**Step 3: Update config.py**

Add imports after line 10:
```python
from src.models.part_autoencoder import ParTAutoencoder
from src.models.part_classifier import ParTClassifier
```

Replace the `get_model` body (lines 47-63) with:
```python
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "autoencoder").lower()
    input_dim = model_cfg.get("input_dim", 128)

    if model_type == "autoencoder":
        latent_dim = model_cfg.get("latent_dim", 16)
        return SimpleAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    elif model_type == "classifier":
        hidden_dim = model_cfg.get("hidden_dim", 256)
        num_classes = model_cfg.get("num_classes", 2)
        return MLPClassifier(
            input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes
        )
    elif model_type == "part_autoencoder":
        return ParTAutoencoder(
            input_dim=input_dim,
            n_particles=model_cfg.get("n_particles", input_dim // 3),
            embed_dims=model_cfg.get("embed_dims", [128, 512, 128]),
            pair_embed_dims=model_cfg.get("pair_embed_dims", [64, 64, 64]),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 8),
            num_cls_layers=model_cfg.get("num_cls_layers", 2),
            decoder_hidden_dim=model_cfg.get("decoder_hidden_dim", 256),
        )
    elif model_type == "part_classifier":
        return ParTClassifier(
            input_dim=input_dim,
            n_particles=model_cfg.get("n_particles", input_dim // 3),
            num_classes=model_cfg.get("num_classes", 2),
            embed_dims=model_cfg.get("embed_dims", [128, 512, 128]),
            pair_embed_dims=model_cfg.get("pair_embed_dims", [64, 64, 64]),
            num_heads=model_cfg.get("num_heads", 8),
            num_layers=model_cfg.get("num_layers", 8),
            num_cls_layers=model_cfg.get("num_cls_layers", 2),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Choose 'autoencoder', 'classifier', 'part_autoencoder', or 'part_classifier'."
        )
```

**Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/python3.10 tests/test_config_part.py`
Expected: `All config factory tests PASSED`

**Step 5: Commit**

```bash
git add src/utils/config.py tests/test_config_part.py
git commit -m "feat: add part_autoencoder and part_classifier to model factory"
```

---

## Task 6: Update Trainer Dispatch

**Files:**
- Modify: `src/training/trainer.py:26` (docstring), `src/training/trainer.py:57-65` (validate), `src/training/trainer.py:106-109` (criterion), `src/training/trainer.py:124-132` (train loop)

**Step 1: Identify changes needed**

The trainer dispatches on `model_type` string:
- `"autoencoder"` → MSE loss, `batch[0]` unpacking, `model(x)` returns `(x_hat, z)`
- everything else → CrossEntropy, `x, y = batch`, `model(x)` returns logits

For ParT integration:
- `"part_autoencoder"` should follow the autoencoder path (same loss, same unpacking)
- `"part_classifier"` should follow the classifier path (same loss, same unpacking)

**Change:** Replace `if config.model_type == "autoencoder":` with `if config.model_type in ("autoencoder", "part_autoencoder"):` in all three dispatch locations.

That's it. The wrappers conform to existing contracts, so no other changes needed.

**Step 2: Make the edits**

In `trainer.py`, at these three locations, change the condition:

Line 26 — update docstring:
```python
    model_type: str = "autoencoder"  # "autoencoder" | "classifier" | "part_autoencoder" | "part_classifier"
```

Line 57 — validate():
```python
            if model_type in ("autoencoder", "part_autoencoder"):
```

Line 106 — train() criterion selection:
```python
    if config.model_type in ("autoencoder", "part_autoencoder"):
```

Line 124 — train() forward pass:
```python
            if config.model_type in ("autoencoder", "part_autoencoder"):
```

**Step 3: Run existing smoke tests (no regressions)**

Run: `/opt/homebrew/bin/python3.10 tests/test_smoke.py`
Expected: `ALL SMOKE TESTS PASSED`

**Step 4: Commit**

```bash
git add src/training/trainer.py
git commit -m "feat: extend trainer dispatch for ParT model types"
```

---

## Task 7: End-to-End ParT Smoke Tests

**Files:**
- Create: `tests/test_part_smoke.py`

**Step 1: Write the end-to-end test**

```python
"""End-to-end smoke tests for ParT models with synthetic data."""
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.dataset import SyntheticLHCDataset, build_dataloaders
from src.models.part_autoencoder import ParTAutoencoder
from src.models.part_classifier import ParTClassifier
from src.training.trainer import TrainConfig, train
from src.utils.config import get_model

TEST_OUTPUT = Path("outputs/_part_smoke_test")

# input_dim must be divisible by 3 for ParT
INPUT_DIM = 63  # 21 particles x 3

def test_part_autoencoder_training():
    """Train ParTAutoencoder for 2 epochs on synthetic data."""
    print("=" * 60)
    print("TEST: ParT Autoencoder Training (Version A)")
    print("=" * 60)

    dataset = SyntheticLHCDataset(n_samples=200, input_dim=INPUT_DIM)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=32)

    model = ParTAutoencoder(
        input_dim=INPUT_DIM,
        n_particles=21,
        embed_dims=[32, 32],
        pair_embed_dims=[16, 16],
        num_heads=2,
        num_layers=2,
        num_cls_layers=1,
        decoder_hidden_dim=64,
    )

    cfg = TrainConfig(
        batch_size=32, lr=1e-3, epochs=2,
        device="cpu", model_type="part_autoencoder",
    )

    out = TEST_OUTPUT / "part_autoencoder"
    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=out,
    )

    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert (out / "loss_log.csv").exists(), "loss_log.csv not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries"
    print("PASSED: ParT Autoencoder training\n")

def test_part_classifier_training():
    """Train ParTClassifier for 2 epochs on synthetic data."""
    print("=" * 60)
    print("TEST: ParT Classifier Training (Version C)")
    print("=" * 60)

    dataset = SyntheticLHCDataset(n_samples=200, input_dim=INPUT_DIM)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=32)

    model = ParTClassifier(
        input_dim=INPUT_DIM,
        n_particles=21,
        num_classes=2,
        embed_dims=[32, 32],
        pair_embed_dims=[16, 16],
        num_heads=2,
        num_layers=2,
        num_cls_layers=1,
    )

    cfg = TrainConfig(
        batch_size=32, lr=1e-3, epochs=2,
        device="cpu", model_type="part_classifier",
    )

    out = TEST_OUTPUT / "part_classifier"
    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=out,
    )

    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries"
    print("PASSED: ParT Classifier training\n")

def test_config_factory_integration():
    """Verify get_model produces working ParT models from config dicts."""
    print("=" * 60)
    print("TEST: Config Factory Integration")
    print("=" * 60)

    cfg_ae = {
        "model": {
            "type": "part_autoencoder", "input_dim": INPUT_DIM,
            "n_particles": 21, "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16], "num_heads": 2,
            "num_layers": 2, "num_cls_layers": 1, "decoder_hidden_dim": 64,
        }
    }
    model_ae = get_model(cfg_ae)
    assert isinstance(model_ae, ParTAutoencoder)
    x_hat, z = model_ae(torch.randn(2, INPUT_DIM))
    assert x_hat.shape == (2, INPUT_DIM)

    cfg_cl = {
        "model": {
            "type": "part_classifier", "input_dim": INPUT_DIM,
            "n_particles": 21, "num_classes": 2, "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16], "num_heads": 2,
            "num_layers": 2, "num_cls_layers": 1,
        }
    }
    model_cl = get_model(cfg_cl)
    assert isinstance(model_cl, ParTClassifier)
    logits = model_cl(torch.randn(2, INPUT_DIM))
    assert logits.shape == (2, 2)

    print("PASSED: Config factory integration\n")

def main():
    print("\nParticle Transformer Smoke Tests\n")

    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)

    test_config_factory_integration()
    test_part_autoencoder_training()
    test_part_classifier_training()

    print("=" * 60)
    print("ALL PART SMOKE TESTS PASSED")
    print("=" * 60)

    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)
        print("Cleaned up test outputs.\n")

if __name__ == "__main__":
    main()
```

**Step 2: Run the test**

Run: `/opt/homebrew/bin/python3.10 tests/test_part_smoke.py`
Expected: `ALL PART SMOKE TESTS PASSED`

**Step 3: Run ALL tests to verify no regressions**

Run: `/opt/homebrew/bin/python3.10 tests/test_smoke.py`
Expected: `ALL SMOKE TESTS PASSED` (existing models unaffected)

**Step 4: Commit**

```bash
git add tests/test_part_smoke.py
git commit -m "test: add end-to-end smoke tests for ParT models"
```

---

## Task 8: Add Example Config

**Files:**
- Create: `configs/part_autoencoder.yaml`
- Create: `configs/part_classifier.yaml`

**Step 1: Create ParT autoencoder config**

```yaml
project: lhc-olympics-2020
seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed

model:
  type: part_autoencoder
  input_dim: 2100
  n_particles: 700
  embed_dims: [128, 512, 128]
  pair_embed_dims: [64, 64, 64]
  num_heads: 8
  num_layers: 8
  num_cls_layers: 2
  decoder_hidden_dim: 256

train:
  batch_size: 128
  lr: 0.001
  epochs: 20
  device: cpu  # change to cuda if available

outputs:
  root: outputs
  models: outputs/models
  logs: outputs/logs
  figures: outputs/figures
```

**Step 2: Create ParT classifier config**

```yaml
project: lhc-olympics-2020
seed: 42

data:
  raw_dir: data/raw
  processed_dir: data/processed

model:
  type: part_classifier
  input_dim: 2100
  n_particles: 700
  num_classes: 2
  embed_dims: [128, 512, 128]
  pair_embed_dims: [64, 64, 64]
  num_heads: 8
  num_layers: 8
  num_cls_layers: 2

train:
  batch_size: 128
  lr: 0.001
  epochs: 20
  device: cpu  # change to cuda if available

outputs:
  root: outputs
  models: outputs/models
  logs: outputs/logs
  figures: outputs/figures
```

**Step 3: Commit**

```bash
git add configs/part_autoencoder.yaml configs/part_classifier.yaml
git commit -m "feat: add example configs for ParT autoencoder and classifier"
```

---

## Summary of All Files

| Action | File | Purpose |
|--------|------|---------|
| Create | `src/models/particle_transformer.py` | ParT core (extracted from weaver-core) |
| Create | `src/models/preprocessing.py` | (pT,eta,phi) -> ParT input format |
| Create | `src/models/part_autoencoder.py` | Version A wrapper |
| Create | `src/models/part_classifier.py` | Version C wrapper |
| Modify | `src/utils/config.py` | Add 2 new model types to factory |
| Modify | `src/training/trainer.py` | Extend dispatch conditions |
| Create | `tests/test_preprocessing.py` | Preprocessor unit tests |
| Create | `tests/test_part_autoencoder.py` | Version A unit tests |
| Create | `tests/test_part_classifier.py` | Version C unit tests |
| Create | `tests/test_config_part.py` | Factory integration tests |
| Create | `tests/test_part_smoke.py` | End-to-end training tests |
| Create | `configs/part_autoencoder.yaml` | Example config for Version A |
| Create | `configs/part_classifier.yaml` | Example config for Version C |
