"""Smoke test: end-to-end training and evaluation with synthetic data.

Run from the repository root:
    python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataset import SyntheticLHCDataset, build_dataloaders
from src.models.autoencoder import SimpleAutoencoder
from src.models.classifier import MLPClassifier
from src.training.trainer import TrainConfig, train
from src.analysis.bump_hunt import bump_hunt
from src.analysis.plotting import plot_loss_curves, plot_anomaly_scores, plot_roc_curve
from src.utils.config import load_config, get_model

import numpy as np


TEST_OUTPUT = Path("outputs/_smoke_test")


def test_autoencoder_training():
    """Train a SimpleAutoencoder for 2 epochs on synthetic data."""
    print("=" * 60)
    print("TEST: Autoencoder Training")
    print("=" * 60)

    input_dim = 64
    dataset = SyntheticLHCDataset(n_samples=2_000, input_dim=input_dim)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=256)

    model = SimpleAutoencoder(input_dim=input_dim, latent_dim=8)

    cfg = TrainConfig(
        batch_size=256, lr=1e-3, epochs=2,
        device="cpu", model_type="autoencoder",
    )

    out = TEST_OUTPUT / "autoencoder"
    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=out,
    )

    # Assertions
    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert (out / "loss_log.csv").exists(), "loss_log.csv not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries in loss log"
    print("✓ Autoencoder training PASSED\n")
    return trained_model, dataset


def test_classifier_training():
    """Train an MLPClassifier for 2 epochs on synthetic data."""
    print("=" * 60)
    print("TEST: Classifier Training")
    print("=" * 60)

    input_dim = 64
    dataset = SyntheticLHCDataset(n_samples=2_000, input_dim=input_dim)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=256)

    model = MLPClassifier(input_dim=input_dim, hidden_dim=64, num_classes=2)

    cfg = TrainConfig(
        batch_size=256, lr=1e-3, epochs=2,
        device="cpu", model_type="classifier",
    )

    out = TEST_OUTPUT / "classifier"
    trained_model, loss_log = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        output_dir=out,
    )

    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries"
    print("✓ Classifier training PASSED\n")
    return trained_model, dataset


def test_plotting(loss_log_ae):
    """Test plotting functions."""
    print("=" * 60)
    print("TEST: Plotting")
    print("=" * 60)

    figs = TEST_OUTPUT / "figures"

    # Loss curves
    train_l = [e["train_loss"] for e in loss_log_ae]
    val_l = [e["val_loss"] for e in loss_log_ae]
    plot_loss_curves(train_l, val_l, figs / "loss_curves.png")
    assert (figs / "loss_curves.png").exists(), "Loss curve plot not saved!"
    print("  ✓ Loss curves OK")

    # Anomaly scores
    scores = np.random.exponential(1.0, size=1000).astype(np.float32)
    labels = np.random.randint(0, 2, size=1000)
    plot_anomaly_scores(scores, labels, figs / "anomaly_scores.png")
    assert (figs / "anomaly_scores.png").exists(), "Anomaly score plot not saved!"
    print("  ✓ Anomaly scores OK")

    # ROC curve
    y_true = np.random.randint(0, 2, size=500)
    y_score = np.random.rand(500)
    auc_val = plot_roc_curve(y_true, y_score, figs / "roc_curve.png")
    assert (figs / "roc_curve.png").exists(), "ROC plot not saved!"
    print(f"  ✓ ROC curve OK (AUC={auc_val:.4f})")

    print("✓ Plotting tests PASSED\n")


def test_bump_hunt():
    """Test bump hunt on synthetic mass distribution."""
    print("=" * 60)
    print("TEST: Bump Hunt")
    print("=" * 60)

    # Background: exponential + small Gaussian signal
    bg = np.random.exponential(500, size=5000)
    sig = np.random.normal(750, 30, size=200)
    masses = np.concatenate([bg, sig])

    result = bump_hunt(masses, mass_window=(700, 800), num_bins=60)
    print(f"  Z-score:    {result.z_score}")
    print(f"  p-value:    {result.p_value}")
    print(f"  Signal:     {result.signal_count}")
    print(f"  Background: {result.background_estimate}")
    assert result.z_score is not None, "Z-score is None"
    print("✓ Bump hunt PASSED\n")


def test_config_loader():
    """Test config loading and model factory."""
    print("=" * 60)
    print("TEST: Config Loader & Model Factory")
    print("=" * 60)

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        model = get_model(cfg)
        print(f"  Model: {model.__class__.__name__}")
        assert model is not None
        print("✓ Config loader PASSED\n")
    else:
        print("  ⚠ configs/config.yaml not found, skipping")
        print("  Testing with inline config instead...")
        cfg = {"model": {"type": "autoencoder", "input_dim": 32, "latent_dim": 8}}
        model = get_model(cfg)
        assert isinstance(model, SimpleAutoencoder)
        print("✓ Config loader PASSED (inline)\n")


def main():
    print("\n🚀 LHC Olympics 2020 ML — Smoke Test Suite\n")

    # Clean previous test outputs
    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)

    # Run tests
    test_config_loader()

    ae_model, ae_dataset = test_autoencoder_training()
    clf_model, clf_dataset = test_classifier_training()

    # Use autoencoder loss log for plotting test
    loss_log = [{"train_loss": 0.5, "val_loss": 0.6}, {"train_loss": 0.3, "val_loss": 0.4}]
    test_plotting(loss_log)

    test_bump_hunt()

    print("=" * 60)
    print("🎉 ALL SMOKE TESTS PASSED")
    print("=" * 60)

    # Cleanup
    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)
        print("Cleaned up test outputs.\n")


if __name__ == "__main__":
    main()
