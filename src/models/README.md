# Models

This directory contains all model implementations for the LHC Olympics 2020 anomaly detection pipeline.

## Architecture Overview

```
                         flat (batch, 2100)
                               |
           +-------------------+-------------------+
           |                   |                   |
    SimpleAutoencoder    ParTAutoencoder      ParTClassifier
      (baseline)          (Version A)         (Version C)
           |                   |                   |
      Linear layers    [LHCOPreprocessor]   [LHCOPreprocessor]
           |              |    |    |          |    |    |
           |          features vectors mask  features vectors mask
           |              |    |    |          |    |    |
           |          [ParticleTransformer] [ParticleTransformer]
           |            CLS token             logits
           |              |                    |
           |          MLP Decoder         Classification Head
           |              |                    |
        (x_hat, z)    (x_hat, z)            logits
```

## Files

| File | Description |
|------|-------------|
| `autoencoder.py` | `SimpleAutoencoder` — 2-layer MLP autoencoder (baseline) |
| `classifier.py` | `MLPClassifier` — 3-layer MLP classifier (baseline) |
| `particle_transformer.py` | Core Particle Transformer model extracted from [weaver-core](https://github.com/hqucms/weaver-core) |
| `preprocessing.py` | `LHCOPreprocessor` — converts flat LHCO2020 vectors to ParT input format |
| `part_autoencoder.py` | `ParTAutoencoder` (Version A) — ParT encoder + MLP decoder for unsupervised anomaly detection |
| `part_classifier.py` | `ParTClassifier` (Version C) — ParT with classification head for supervised transfer learning |

## Particle Transformer (ParT)

The Particle Transformer ([arXiv:2202.03772](https://arxiv.org/abs/2202.03772)) is a transformer-based architecture enhanced with **pairwise particle interaction features** as attention bias. It is state-of-the-art on jet tagging benchmarks.

### Key Innovation: Pairwise Interaction Features

For every pair of particles (i, j), the model computes 4 physics-motivated quantities from their Lorentz 4-vectors:

- `ln(kT)` — transverse momentum clustering distance
- `ln(z)` — momentum fraction
- `ln(deltaR)` — angular distance in (rapidity, phi) space
- `ln(m^2)` — invariant mass squared of the pair

These are projected through Conv1d layers and **added as bias to the attention logits** before softmax. This injects relational physics directly into the attention mechanism — the model knows which particles are close, which share momentum, and which form resonances.

### Architecture Details

```
Input: (N, 3, P) particle features + (N, 4, P) Lorentz 4-vectors + (N, 1, P) mask
  |
  +-- Embed: BatchNorm1d -> [LayerNorm -> Linear -> GELU] x len(embed_dims)
  |     -> (P, N, embed_dim)
  |
  +-- PairEmbed: 4-vector pairs -> [Conv1d -> BN -> GELU] x len(pair_embed_dims)
  |     -> (N*num_heads, P, P) attention bias
  |
  +-- Transformer Blocks (x num_layers):
  |     Pre-norm, multi-head attention (with pairwise bias), FFN
  |     Per-head scaling, per-residual scaling
  |
  +-- CLS Blocks (x num_cls_layers):
  |     CaIT-style class attention (CLS token attends to sequence)
  |
  +-- LayerNorm -> CLS token output (N, embed_dim)
  |
  +-- FC head (if classifier) or return CLS token (if encoder)
```

Default configuration: 8 transformer layers, 2 CLS layers, 8 heads, embed_dim=128.

## Preprocessing: LHCOPreprocessor

The LHCO2020 dataset stores events as flat vectors of shape `(2100,)` = 700 particles x (pT, eta, phi). ParT expects multi-tensor input. The `LHCOPreprocessor` (in `preprocessing.py`) handles this conversion:

```
Input: (batch, 2100) flat vector

Reshape to (batch, 700, 3):
  - pT:  transverse momentum
  - eta: pseudorapidity
  - phi: azimuthal angle

Output tensors:
  - features: (batch, 3, 700) -- [log(pT), eta, phi]
  - vectors:  (batch, 4, 700) -- [px, py, pz, E] (massless approximation)
  - mask:     (batch, 1, 700) -- 1 where pT > 0, 0 for padding
```

4-vector computation (massless particle approximation):
```
px = pT * cos(phi)
py = pT * sin(phi)
pz = pT * sinh(eta)
E  = pT * cosh(eta)
```

The preprocessor is embedded inside the ParT wrapper models, so the training pipeline sees the same `(batch, input_dim)` interface as the baseline models.

## Version A: ParTAutoencoder

**Purpose:** Unsupervised anomaly detection — no labels needed.

**How it works:**
1. ParT encoder processes particle-level data with physics-aware attention
2. CLS token embedding (128-dim) serves as the latent representation
3. MLP decoder reconstructs the original flat vector from the latent
4. **Anomaly score = reconstruction error (MSE)**

Background events are well-reconstructed (low MSE). Signal events, being unlike anything in training data, produce high reconstruction error.

**Config:** `configs/part_autoencoder.yaml`

```bash
python scripts/train.py --config configs/part_autoencoder.yaml \
    --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5
```

## Version C: ParTClassifier

**Purpose:** Supervised transfer learning — requires labeled data.

**How it works:**
1. Train on R&D dataset with truth labels (background=0, signal=1)
2. ParT learns to distinguish signal from background at particle level
3. Apply to black box data — classifier confidence = anomaly score

**Limitation:** Only detects signal types similar to those in training data. The R&D dataset contains a specific signal topology (Z' -> XY -> qqq). Signals with different decay channels may not transfer.

**Config:** `configs/part_classifier.yaml`

```bash
python scripts/train.py --config configs/part_classifier.yaml \
    --data data/raw/events_LHCO2020_RnD.h5
```

## Configuration

Both ParT models accept these hyperparameters via YAML config:

```yaml
model:
  type: part_autoencoder  # or part_classifier
  input_dim: 2100         # 700 particles x 3 features
  n_particles: 700
  embed_dims: [128, 512, 128]     # embedding projection dimensions
  pair_embed_dims: [64, 64, 64]   # pairwise feature projection
  num_heads: 8                     # attention heads
  num_layers: 8                    # transformer blocks
  num_cls_layers: 2                # CaIT class-attention blocks
  # For part_autoencoder only:
  decoder_hidden_dim: 256
  # For part_classifier only:
  num_classes: 2
```

For quick experiments, reduce `num_layers` and `embed_dims` to speed up training:
```yaml
  embed_dims: [64, 64]
  pair_embed_dims: [32, 32]
  num_heads: 4
  num_layers: 4
  num_cls_layers: 1
```
