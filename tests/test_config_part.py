"""Test config factory recognizes new model types."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_model
from src.models.part_autoencoder import ParTAutoencoder
from src.models.part_classifier import ParTClassifier

def test_part_autoencoder_factory():
    cfg = {
        "model": {
            "type": "part_autoencoder",
            "input_dim": 63,
            "n_particles": 21,
            "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16],
            "num_heads": 2,
            "num_layers": 2,
            "num_cls_layers": 1,
            "decoder_hidden_dim": 64,
        }
    }
    model = get_model(cfg)
    assert isinstance(model, ParTAutoencoder), f"Got {type(model)}"
    print("PASSED: test_part_autoencoder_factory")

def test_part_classifier_factory():
    cfg = {
        "model": {
            "type": "part_classifier",
            "input_dim": 63,
            "n_particles": 21,
            "num_classes": 2,
            "embed_dims": [32, 32],
            "pair_embed_dims": [16, 16],
            "num_heads": 2,
            "num_layers": 2,
            "num_cls_layers": 1,
        }
    }
    model = get_model(cfg)
    assert isinstance(model, ParTClassifier), f"Got {type(model)}"
    print("PASSED: test_part_classifier_factory")

if __name__ == "__main__":
    test_part_autoencoder_factory()
    test_part_classifier_factory()
    print("\nAll config factory tests PASSED")
