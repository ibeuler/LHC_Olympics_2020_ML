# LHC Olympics 2020 — ML Anomaly Detection

A PyTorch-based machine learning pipeline for the [LHC Olympics 2020](https://lhco2020.github.io/homepage/) anomaly detection challenge. Trains either an **autoencoder** (unsupervised), a **Variational Autoencoder** (VAE), a **Particle Transformer** (ParT-AE), or an **MLP classifier** (supervised) on collider event data.

The pipeline scores events for new-physics signatures using reconstruction loss or class probability, followed by a bump-hunt statistical test on the invariant-mass spectrum.

---

## Overview

| Component | Description |
|-----------|-------------|
| `src/data/` | HDF5 dataset loader, particle-level and event-level modes, anti-kT jet clustering (`pyjet`) |
| `src/models/` | `SimpleAutoencoder`, `VariationalAutoencoder` (beta-VAE), `ParticleTransformerAE` (ParT), `MLPClassifier` |
| `src/training/` | Unified training loop with W&B logging (MSE/ELBO for AE/VAE/ParT, CrossEntropy for classifier) |
| `src/analysis/` | Anomaly score plots, ROC/AUC curves, CMS-style plotting (`mplhep`), sliding-window bump-hunt |
| `src/utils/` | YAML config loading, model factory |
| `scripts/` | `train.py` and `evaluate.py` entry points |
| `configs/` | `config.yaml` (main config), `sweep.yaml` (W&B hyperparameter sweep) |

---

## Quickstart

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd LHC_Olympics_2020_ML

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train (synthetic data — no download needed)
python scripts/train.py --config configs/config.yaml --model-type autoencoder

# 5. Evaluate a saved checkpoint
python scripts/evaluate.py \
    --checkpoint outputs/models/best_model.pt \
    --config configs/config.yaml \
    --model-type autoencoder
```

> **Real data:** Download `events_anomalydetection.h5` from the [LHC Olympics Zenodo record](https://zenodo.org/record/6547837) and pass it via `--data path/to/events_anomalydetection.h5`.

---

## Training

The pipeline supports four model architectures:

### 1. Particle Transformer Autoencoder (ParT-AE)
Modern transformer-based architecture using physics-augmented attention. Operates directly on particle-level features (pT, eta, phi).
```bash
python scripts/train.py \
    --model-type part_ae \
    --background-only \
    --epochs 50 \
    --batch-size 256
```

### 2. Variational Autoencoder (VAE)
Trains on background events only; signal is detected via high reconstruction loss (MSE + KL divergence).
```bash
python scripts/train.py \
    --model-type vae \
    --background-only \
    --epochs 30
```

### 3. Plain Autoencoder
Simple MLP-based autoencoder for event-level high-level features.
```bash
python scripts/train.py \
    --model-type autoencoder \
    --background-only
```

### 4. MLP Classifier
Supervised classification (requires both signal and background labels).
```bash
python scripts/train.py \
    --model-type classifier \
    --lr 3e-4
```

Checkpoints are saved to `outputs/models/`. Loss curves are saved to `outputs/figures/loss_curves.png`.

---

## Evaluation

Evaluation produces anomaly score distributions, ROC curves with AUC metrics, and performs a statistical bump-hunt on the invariant mass spectrum.

```bash
python scripts/evaluate.py \
    --checkpoint outputs/models/best_model.pt \
    --model-type part_ae \
    --data data/raw/events_anomalydetection.h5
```

**Produced Artefacts:**
- `outputs/figures/[model]_anomaly_scores.png` — score distributions for signal vs. background
- `outputs/figures/[model]_roc_curve.png` — ROC curve with AUC
- `outputs/figures/mass_distribution.png` — invariant-mass spectrum
- **Bump-hunt results** printed to stdout (Z-score, p-value, signal count, background estimate)

---

## Configuration

Edit `configs/config.yaml` to change model architecture, training hyper-parameters, or data paths:

```yaml
model:
  type: part_ae          # autoencoder | vae | classifier | part_ae
  n_particles: 200       # for part_ae
  embed_dim: 128
  latent_dim: 16         # for AE/VAE
  beta: 1.0              # VAE KL weight

train:
  batch_size: 512
  lr: 0.001
  epochs: 10
  device: cpu            # cuda | mps | cpu

preprocessing:
  jet_algorithm: antikt
  jet_radius: 0.4
```

---

## Project Structure

```
LHC_Olympics_2020_ML/
├── configs/
│   ├── config.yaml          # Main configuration
│   └── sweep.yaml           # W&B sweep definition
├── scripts/
│   ├── train.py             # Training entry point
│   └── evaluate.py          # Evaluation and bump-hunt
├── src/
│   ├── analysis/
│   │   ├── bump_hunt.py     # Polynomial background fit + Z-score
│   │   └── plotting.py      # CMS-style figures via mplhep
│   ├── data/
│   │   ├── clustering.py    # Anti-kT jet clustering (pyjet)
│   │   └── dataset.py       # HDF5 loader + synthetic fallback
│   ├── models/
│   │   ├── autoencoder.py   # SimpleAutoencoder + VAE
│   │   ├── classifier.py    # MLPClassifier
│   │   └── particle_transformer.py # Particle Transformer (ParT-AE)
│   ├── training/
│   │   └── trainer.py       # Unified training loop
│   └── utils/
│       └── config.py        # YAML loader & model factory
├── tests/
│   └── test_smoke.py        # Pipeline smoke tests
├── outputs/                 # Generated artefacts (gitignored)
└── requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model training & inference |
| `numpy`, `scipy` | Numerical computing & statistical fitting |
| `h5py` | HDF5 data handling |
| `matplotlib`, `mplhep` | Publication-quality plotting |
| `scikit-learn` | Metrics (ROC/AUC) |
| `pyjet` | *(Optional)* Jet clustering |
| `wandb` | *(Optional)* Experiment tracking |

---

## Notes

- **Synthetic Data:** If no data file is provided, the scripts automatically generate synthetic data to test the pipeline end-to-end.
- **Jet Clustering:** `pyjet` is required for clustering raw particle data. If missing, the pipeline falls back to pre-processed features if available.
- **W&B Integration:** Use `--wandb-project <name>` in `train.py` to enable real-time logging and hyperparameter sweeps.
