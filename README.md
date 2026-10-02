# LHC Olympics 2020 — ML Project

Repository for unsupervised anomaly detection and supervised classification on the LHC Olympics 2020 dataset using Deep Learning and Particle Transformer (ParT).

---

## Directory Structure

```text
├── configs/            # YAML configurations (ParT, Baseline AE, Classifier)
├── data/
│   └── raw/            # HDF5 dataset files (events_LHCO2020_*.h5)
├── notebooks/          # Exploratory data analysis & prototyping
├── outputs/            # Training artifacts (models .pt, loss curves, metadata)
│   ├── figures/        # Train/val loss curve plots
│   ├── logs/           # Run metadata and effective configs
│   └── models/         # Best and final model checkpoints
├── report/             # Evaluation results, tables, and physics figures
│   ├── plots/          # ROC curves, anomaly score histograms, interpretability features
│   └── tables/         # HEP metric summary tables (AUC, SIC, quantile thresholds)
├── scripts/            # CLI entry points (train.py, evaluate.py, download_data.py)
├── src/                # Modular core package
│   ├── analysis/       # HEP metrics, plotting, interpretability, physics observables
│   ├── data/           # HDF5 dataset loader (LHCDataset) & batching
│   ├── models/         # Particle Transformer (ParT), Preprocessing, Autoencoders
│   ├── training/       # Training loop, AMP (FP16), checkpointing, validation
│   └── utils/          # Config parser and helpers
├── tests/              # Smoke tests for pipelines and models
└── requirements.txt    # Python dependencies
```

---

## Implemented Models

| Model | Type | Architecture | Config | Primary Use Case |
|---|---|---|---|---|
| `ParTAutoencoder` | `part_autoencoder` | Particle Transformer + Pairwise Kinematics + MLP Decoder | `configs/part_autoencoder.yaml` | **Unsupervised Anomaly Detection** (SOTA) |
| `ParTAutoencoder (no U)` | `part_autoencoder` | ParT without pairwise attention bias | `configs/part_autoencoder_no_pairwise.yaml` | Ablation study on the pairwise interaction matrix |
| `ParTClassifier` | `part_classifier` | Particle Transformer + Classification Head | `configs/part_classifier.yaml` | Supervised classification / transfer learning |
| `SimpleAutoencoder` | `autoencoder` | Dense MLP Autoencoder (2 layers) | `configs/config.yaml` | Baseline unsupervised benchmark |
| `MLPClassifier` | `classifier` | Dense MLP Classifier (3 layers) | `configs/config.yaml` | Baseline supervised benchmark |

### Particle Transformer (ParT) Optimizations:
- **Kinematic Preprocessing & Particle Pruning:** Events are sorted by transverse momentum ($p_T$) and truncated to the top **128 particles**. This retains **97.35%** of the event energy while reducing the pairwise attention matrix from $700 \times 700$ to $128 \times 128$ (~**30x computational speedup**).
- **Automatic Mixed Precision (AMP):** Utilizes PyTorch `torch.amp.autocast('cuda')` with FP16 and `GradScaler` for full tensor-core acceleration on modern GPUs (e.g., NVIDIA RTX 40-series).
- **Pairwise Kinematic Embeddings ($U$ matrix):** Computes pairwise physics observables ($k_T$, momentum fraction $z$, $\Delta R$, and invariant mass $m^2$) directly from Lorentz 4-vectors to bias multi-head self-attention.

---

## Quickstart

### 1. Environment Setup

Clone the repository and activate your Python environment (Python 3.10 - 3.12 recommended):

```powershell
# Activate virtual environment
.\.venv\Scripts\activate

# Install PyTorch with CUDA support (e.g., CUDA 12.1 for NVIDIA RTX GPUs)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

Verify GPU availability:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

### 2. Dataset Preparation

Download the official LHC Olympics 2020 datasets into `data/raw/`:

* **Background MC Pythia (1M background events, ~2.7 GB):**
  ```powershell
  python scripts/download_data.py --dataset background
  ```
* **R&D Dataset (1.1M events: 1M background + 100k signal):**
  * Direct Zenodo link: [events_anomalydetection_v2.h5](https://zenodo.org/records/6466204/files/events_anomalydetection_v2.h5)
  * Save to: `data/raw/events_LHCO2020_RnD.h5`
* **Black Box 1 (Unlabeled challenge data):**
  ```powershell
  python scripts/download_data.py --dataset blackbox1
  ```

---

### 3. Model Training

Train the **Particle Transformer Autoencoder** on the 1M Pythia background sample:

```powershell
python scripts/train.py `
  --config configs/part_autoencoder.yaml `
  --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 `
  --batch-size 256 `
  --epochs 20 `
  --device cuda
```

Training outputs will be saved automatically:
* Best model checkpoint: `outputs/models/best_model_<run_tag>.pt`
* Loss progression curve: `outputs/figures/loss_curves_<run_tag>.png`
* Metadata & effective config: `outputs/logs/run_meta_<run_tag>.json`

---

### 4. Evaluation & Anomaly Detection

Evaluate the trained checkpoint to compute reconstruction MSE anomaly scores, identify high-anomaly tail events, and generate HEP interpretability figures:

```powershell
python scripts/evaluate.py `
  --checkpoint outputs/models/best_model_parT_AE_ep20_bs256_lr1e-03_seed42_cuda_20260927_132059.pt `
  --config configs/part_autoencoder.yaml `
  --data data/raw/events_LHCO2020_backgroundMC_Pythia.h5 `
  --model-type part_autoencoder `
  --tag partAE_eval_bg `
  --device cuda
```

Generated evaluation outputs in `report/`:
* `report/plots/lhc_background_scores_ParTAE.png`: Anomaly score (MSE) distribution over 1M events.
* `report/plots/interpretability_features.png`: Physical feature comparisons ($p_T, \eta, \phi, m_{jj}$) between normal and top-1% anomalous events.
* `report/tables/lhc_background_evaluation.csv`: Summary metrics including mean score, median, 95% and 99% quantile thresholds.

To evaluate on labeled R&D data for ROC, AUC, and SIC metrics:
```powershell
python scripts/evaluate.py `
  --checkpoint outputs/models/best_model_parT_AE_ep20_bs256_lr1e-03_seed42_cuda_20260927_132059.pt `
  --config configs/part_autoencoder.yaml `
  --data data/raw/events_LHCO2020_RnD.h5 `
  --model-type part_autoencoder `
  --tag partAE_eval_rnd `
  --device cuda
```

---

## Smoke Tests

Run smoke tests to verify model definitions and data pipelines:

```powershell
python tests/test_smoke.py        # Baseline models
python tests/test_part_smoke.py   # Particle Transformer models
```

---

## References & Citations

1. **LHC Olympics 2020 Challenge:** Kasieczka, G. et al., *The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics*, [arXiv:2101.08320](https://arxiv.org/abs/2101.08320).
2. **Particle Transformer (ParT):** Qu, H., Li, C., & Qian, S., *Particle Transformer for Jet Tagging*, [arXiv:2202.03772](https://arxiv.org/abs/2202.03772), [weaver-core repository](https://github.com/hqucms/weaver-core).
