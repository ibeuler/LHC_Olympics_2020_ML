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
| `ParTAutoencoder` (no U) | part_autoencoder | `configs/part_autoencoder_no_pairwise.yaml` | Measures the impact of the pairwise attention bias |

## Quickstart

1. Download the dataset files into `data/raw/` (or use the automatic helper script):
   ```bash
   python scripts/download_data.py --dataset rnd        # R&D Dataset (~1.5GB)
   python scripts/download_data.py --dataset background # Pythia Background Dataset
   ```
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

## Physics-Aware Evaluation and ParT Ablation

The evaluation pipeline reports the metrics commonly used in collider analyses:
AUC, best-threshold accuracy, maximum SIC, and background rejection at signal
efficiencies of 0.2, 0.3, and 0.5. If no background event survives a cut, the
calculation uses the measurable limit `1 / N_background` instead of reporting
infinite rejection. The CSV output records this limit explicitly.

To compare the baseline autoencoder with ParTAE, with and without pairwise
kinematics, run:

```bash
python scripts/evaluate.py \
  --model SimpleAE configs/autoencoder_lhco.yaml outputs/simple_ae.pt \
  --model ParTAE-no-U configs/part_autoencoder_no_pairwise.yaml outputs/part_no_u.pt \
  --model ParTAE-with-U configs/part_autoencoder.yaml outputs/part_with_u.pt \
  --lhc-background-data data/raw/events_LHCO2020_backgroundMC_Pythia.h5
```

The command evaluates every model on the same synthetic sample and on real LHC
background. Results are written to `report/tables/synthetic_validation.csv`,
`report/tables/lhc_background_evaluation.csv`, and the cumulative
`report/tables/results_summary.csv`. The report also includes parameter counts,
ROC and score comparisons, and `report/plots/interpretability_features.png`.

## Tests

```bash
python tests/test_smoke.py        # baseline models
python tests/test_part_smoke.py   # Particle Transformer models
```

## Notes
- Jet clustering is stubbed in `src/data/clustering.py` (intended for FastJet/PyJet).
- The Particle Transformer implementation is extracted from [weaver-core](https://github.com/hqucms/weaver-core) (paper: [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)).
