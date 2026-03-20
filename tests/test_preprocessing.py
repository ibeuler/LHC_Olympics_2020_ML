"""Tests for LHCOPreprocessor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.preprocessing import LHCOPreprocessor

def test_preprocessor_shapes():
    n_particles = 21
    input_dim = n_particles * 3
    batch_size = 4
    prep = LHCOPreprocessor(n_particles=n_particles)
    x = torch.randn(batch_size, input_dim)
    features, vectors, mask = prep(x)
    assert features.shape == (batch_size, 3, n_particles), f"features: {features.shape}"
    assert vectors.shape == (batch_size, 4, n_particles), f"vectors: {vectors.shape}"
    assert mask.shape == (batch_size, 1, n_particles), f"mask: {mask.shape}"
    print("PASSED: test_preprocessor_shapes")

def test_preprocessor_mask():
    n_particles = 10
    input_dim = n_particles * 3
    prep = LHCOPreprocessor(n_particles=n_particles)
    x = torch.zeros(2, input_dim)
    for i in range(3):
        x[0, i * 3] = 5.0
        x[1, i * 3] = 3.0
    _, _, mask = prep(x)
    assert mask[0, 0, :3].sum() == 3, "First 3 should be real"
    assert mask[0, 0, 3:].sum() == 0, "Rest should be padded"
    print("PASSED: test_preprocessor_mask")

def test_preprocessor_4vectors():
    n_particles = 1
    prep = LHCOPreprocessor(n_particles=n_particles)
    x = torch.tensor([[10.0, 0.0, 0.0]])
    _, vectors, _ = prep(x)
    assert torch.allclose(vectors[0, 0, 0], torch.tensor(10.0), atol=1e-5), "px"
    assert torch.allclose(vectors[0, 1, 0], torch.tensor(0.0), atol=1e-5), "py"
    assert torch.allclose(vectors[0, 2, 0], torch.tensor(0.0), atol=1e-5), "pz"
    assert torch.allclose(vectors[0, 3, 0], torch.tensor(10.0), atol=1e-5), "E"
    print("PASSED: test_preprocessor_4vectors")

if __name__ == "__main__":
    test_preprocessor_shapes()
    test_preprocessor_mask()
    test_preprocessor_4vectors()
    print("\nAll preprocessing tests PASSED")
