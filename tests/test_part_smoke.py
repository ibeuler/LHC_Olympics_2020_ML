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
INPUT_DIM = 63  # 21 particles x 3

def test_part_autoencoder_training():
    print("=" * 60)
    print("TEST: ParT Autoencoder Training (Version A)")
    print("=" * 60)

    dataset = SyntheticLHCDataset(n_samples=200, input_dim=INPUT_DIM)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=32)

    model = ParTAutoencoder(
        input_dim=INPUT_DIM, n_particles=21,
        embed_dims=[32, 32], pair_embed_dims=[16, 16],
        num_heads=2, num_layers=2, num_cls_layers=1,
        decoder_hidden_dim=64,
    )

    cfg = TrainConfig(
        batch_size=32, lr=1e-3, epochs=2,
        device="cpu", model_type="part_autoencoder",
    )

    out = TEST_OUTPUT / "part_autoencoder"
    trained_model, loss_log = train(
        model=model, train_loader=train_loader,
        val_loader=val_loader, config=cfg, output_dir=out,
    )

    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert (out / "loss_log.csv").exists(), "loss_log.csv not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries"
    print("PASSED: ParT Autoencoder training\n")

def test_part_classifier_training():
    print("=" * 60)
    print("TEST: ParT Classifier Training (Version C)")
    print("=" * 60)

    dataset = SyntheticLHCDataset(n_samples=200, input_dim=INPUT_DIM)
    train_loader, val_loader = build_dataloaders(dataset, batch_size=32)

    model = ParTClassifier(
        input_dim=INPUT_DIM, n_particles=21, num_classes=2,
        embed_dims=[32, 32], pair_embed_dims=[16, 16],
        num_heads=2, num_layers=2, num_cls_layers=1,
    )

    cfg = TrainConfig(
        batch_size=32, lr=1e-3, epochs=2,
        device="cpu", model_type="part_classifier",
    )

    out = TEST_OUTPUT / "part_classifier"
    trained_model, loss_log = train(
        model=model, train_loader=train_loader,
        val_loader=val_loader, config=cfg, output_dir=out,
    )

    assert (out / "best_model.pt").exists(), "best_model.pt not saved!"
    assert len(loss_log) == 2, "Expected 2 epoch entries"
    print("PASSED: ParT Classifier training\n")

def test_config_factory_integration():
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
