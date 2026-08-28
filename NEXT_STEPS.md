# Project Roadmap

This note summarizes the current state of the LHCO 2020 project and the most
useful next steps.

## Current Status

- **Data:** The pipeline has been run on one million Pythia background events
  from `events_LHCO2020_backgroundMC_Pythia.h5`.
- **Model:** A Particle Transformer autoencoder has been trained for 20 epochs.
- **Evaluation:** GPU inference, anomaly scoring, and bump-hunt analysis are in
  place. The evaluation pipeline now also reports SIC and background rejection.

## Recommended Next Steps

### Improve Training Stability and Throughput

- Lower the learning rate from `0.001` to `0.0003`; the larger value produced
  NaN losses in some epochs.
- Add `clip_grad_norm_` to `trainer.py` if gradient spikes continue.
- Try `batch_size: 256` and `num_workers: 4` to improve GPU utilization, subject
  to available memory and host bandwidth.

### Validate on the Labelled R&D Sample

Download the sample:

  ```bash
  python scripts/download_data.py --dataset rnd
  ```

Then measure ROC AUC, SIC, and background rejection on
`events_LHCO2020_RnD.h5`:

  ```bash
  python scripts/evaluate.py --checkpoint outputs/models/<model_checkpoint>.pt --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
  ```

### Train the Supervised ParT Classifier

Use the labelled R&D sample to train a direct signal/background classifier:

  ```bash
  python scripts/train.py --config configs/part_classifier.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
  ```

## Quick Command Reference

```bash
# Download the labelled R&D sample
python scripts/download_data.py --dataset rnd

# Train the autoencoder with the more stable learning rate
python scripts/train.py --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 --lr 0.0003 --batch-size 256

# Evaluate a checkpoint on the GPU
python scripts/evaluate.py --checkpoint outputs/models/<model_name>.pt --config configs/part_autoencoder.yaml --data data/raw/events_LHCO2020_RnD.h5 --device cuda
```
