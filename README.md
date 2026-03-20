# LHC Olympics 2020 — ML Project

Repository for anomaly detection on the LHC Olympics 2020 dataset using machine learning.

## Structure
- `data/`: raw/processed/external data (git-ignored)
- `notebooks/`: exploration + prototyping notebooks
- `src/`: reusable core code (data, models, training, analysis)
  - `src/models/`: model implementations — see [src/models/README.md](src/models/README.md) for details
- `configs/`: YAML configs (single-run + sweep)
- `outputs/`: models/logs/figures (git-ignored)
- `scripts/`: CLI entry points
- `docs/plans/`: design documents

## Available Models

| Model | Type | Config | Use Case |
|-------|------|--------|----------|
| `SimpleAutoencoder` | autoencoder | `configs/config.yaml` | Baseline unsupervised anomaly detection |
| `MLPClassifier` | classifier | `configs/config.yaml` | Baseline supervised classification |
| `ParTAutoencoder` | part_autoencoder | `configs/part_autoencoder.yaml` | ParT-based unsupervised anomaly detection |
| `ParTClassifier` | part_classifier | `configs/part_classifier.yaml` | ParT-based supervised transfer learning |

## Quickstart

1. Put the official challenge HDF5 files in `data/raw/`.
2. Create an environment and install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Train a model:
   ```bash
   # Baseline autoencoder (synthetic data, no HDF5 needed)
   python scripts/train.py --epochs 5

   # ParT autoencoder on real data
   python scripts/train.py --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5

   # ParT classifier (supervised, requires labeled data)
   python scripts/train.py --config configs/part_classifier.yaml --data data/raw/events_LHCO2020_RnD.h5
   ```

## Tests

```bash
python tests/test_smoke.py        # baseline models
python tests/test_part_smoke.py   # Particle Transformer models
```

## Notes
- Jet clustering is stubbed in `src/data/clustering.py` (intended for FastJet/PyJet).
- The Particle Transformer implementation is extracted from [weaver-core](https://github.com/hqucms/weaver-core) (paper: [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)).
