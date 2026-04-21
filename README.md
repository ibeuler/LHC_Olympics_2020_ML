# LHC Olympics 2020 — ML Anomaly Detection

A PyTorch-based machine learning pipeline for the [LHC Olympics 2020](https://lhco2020.github.io/homepage/) anomaly detection challenge. Trains either an **autoencoder** (unsupervised) or an **MLP classifier** (supervised) on collider event data, then scores events for new-physics signatures using reconstruction loss or class probability — followed by a bump-hunt statistical test on the invariant-mass spectrum.

---

## Overview

| Component | Description |
|-----------|-------------|
| `src/data/` | HDF5 dataset loader, `BackgroundOnlyDataset` wrapper, synthetic fallback, anti-kT jet clustering (pyjet) |
| `src/models/` | `SimpleAutoencoder`, `VariationalAutoencoder` (beta-VAE), `MLPClassifier` |
| `src/training/` | Unified training loop with W&B logging (MSE/ELBO for AE/VAE, CrossEntropy for classifier) |
| `src/analysis/` | Anomaly score plots, ROC/AUC curves, mass distributions, sliding-window bump-hunt |
| `src/utils/` | YAML config loading, model factory |
| `scripts/` | `train.py` and `evaluate.py` entry points |
| `configs/` | `config.yaml` (main config), `sweep.yaml` (W&B hyperparameter sweep) |
| `notebooks/` | Data exploration, preprocessing tests, model prototyping |

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
python scripts/train.py --config configs/config.yaml

# 5. Evaluate a saved checkpoint
python scripts/evaluate.py \
    --checkpoint outputs/models/model_final.pt \
    --config configs/config.yaml
```

> **Real data:** Download `events_anomalydetection.h5` from the [LHC Olympics Zenodo record](https://zenodo.org/record/6547837) and pass it via `--data path/to/events_anomalydetection.h5`.

---

## Training

```bash
# VAE (recommended) — train on background only so signal = high reconstruction loss
python scripts/train.py \
    --config configs/config.yaml \
    --data data/raw/events_anomalydetection.h5 \
    --model-type vae \
    --background-only \
    --epochs 50

# Plain autoencoder
python scripts/train.py \
    --config configs/config.yaml \
    --data data/raw/events_anomalydetection.h5 \
    --model-type autoencoder \
    --background-only \
    --epochs 50

# MLP classifier
python scripts/train.py \
    --config configs/config.yaml \
    --data data/raw/events_anomalydetection.h5 \
    --model-type classifier \
    --lr 3e-4 \
    --batch-size 1024

# With W&B experiment tracking
python scripts/train.py \
    --config configs/config.yaml \
    --data data/raw/events_anomalydetection.h5 \
    --model-type vae \
    --background-only \
    --wandb-project lhc-olympics
```

Checkpoints are saved to `outputs/models/`. Loss curves are saved to `outputs/figures/loss_curves.png`.

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint outputs/models/model_final.pt \
    --config configs/config.yaml \
    --data data/raw/events_anomalydetection.h5 \
    --model-type autoencoder \
    --output outputs
```

Produces:
- `outputs/figures/anomaly_scores.png` — score distributions for signal vs. background
- `outputs/figures/roc_curve.png` — ROC curve with AUC
- `outputs/figures/mass_distribution.png` — invariant-mass spectrum
- Bump-hunt results printed to stdout (Z-score, p-value, signal count, background estimate)

---

## Configuration

Edit `configs/config.yaml` to change model architecture, training hyper-parameters, or data paths:

```yaml
model:
  type: vae             # autoencoder | vae | classifier
  input_dim: 128
  latent_dim: 16
  hidden_dim: 256
  beta: 1.0             # VAE KL weight; try 0.1–4.0

train:
  batch_size: 512
  lr: 0.001
  epochs: 10
  device: cpu           # cuda | mps | cpu

preprocessing:
  jet_algorithm: antikt
  jet_radius: 0.4
  jet_pt_min: 20.0
```

### Hyperparameter Sweep (W&B)

```bash
wandb sweep configs/sweep.yaml
wandb agent <sweep-id>
```

The sweep searches over `lr` (log-uniform), `batch_size` {256, 512, 1024}, and `latent_dim` {8, 16, 32}, minimising `val_loss`.

---

## Project Structure

```
LHC_Olympics_2020_ML/
├── configs/
│   ├── config.yaml          # Main config
│   └── sweep.yaml           # W&B sweep definition
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_test.ipynb
│   └── 03_model_prototyping.ipynb
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── analysis/
│   │   ├── bump_hunt.py     # Polynomial background fit + Z-score
│   │   └── plotting.py      # CMS-style figures via mplhep
│   ├── data/
│   │   ├── clustering.py    # Anti-kT jet clustering (pyjet)
│   │   └── dataset.py       # HDF5 loader + synthetic fallback
│   ├── models/
│   │   ├── autoencoder.py   # SimpleAutoencoder + VariationalAutoencoder (beta-VAE)
│   │   └── classifier.py    # MLPClassifier
│   ├── training/
│   │   └── trainer.py       # Training loop, CSV loss log
│   └── utils/
│       └── config.py        # YAML loader, model factory
├── tests/
│   └── test_smoke.py
├── outputs/                 # Generated artefacts (gitignored)
└── requirements.txt
```

---

## Dependencies

Core dependencies (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---------|---------|
| `torch` | Model training |
| `numpy`, `scipy` | Numerical computing, bump-hunt fitting |
| `h5py` | HDF5 data file reading |
| `matplotlib`, `mplhep` | CMS/ATLAS-style plotting |
| `pyyaml` | Config loading |
| `scikit-learn` | ROC/AUC metrics |
| `pyjet` *(optional)* | Anti-kT jet clustering |
| `wandb` *(optional)* | Experiment tracking and sweeps |

---

## Notes

- If `pyjet` is not installed, jet clustering is disabled but all other functionality works.
- If `mplhep` is not installed, plots fall back to matplotlib defaults.
- Without a data file, both scripts use a **synthetic dataset** for testing the pipeline end-to-end.
- All outputs (models, figures, logs) are written under `outputs/` which is gitignored.
