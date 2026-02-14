# LHC Olympics 2020 — ML Project

Repository skeleton for working with the LHC Olympics 2020 dataset.

## Structure
- `data/`: raw/processed/external data (git-ignored)
- `notebooks/`: exploration + prototyping notebooks
- `src/`: reusable core code (data, models, training, analysis)
- `configs/`: YAML configs (single-run + sweep)
- `outputs/`: models/logs/figures (git-ignored)
- `scripts/`: CLI entry points

## Quickstart
1. Put the official challenge HDF5 files in `data/raw/`.
2. Create an environment and install deps:
   - `pip install -r requirements.txt`
3. Run (placeholder until you implement `train()`):
   - `python scripts/train.py --epochs 1`

## Notes
- Jet clustering is stubbed in `src/data/clustering.py` (intended for FastJet/PyJet).
- Training loop is stubbed in `src/training/trainer.py`.
