"""Tests for ParTAutoencoder (Version A)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.part_autoencoder import ParTAutoencoder

TINY_CFG = dict(
    input_dim=63,
    n_particles=21,
    embed_dims=[32, 32],
    pair_embed_dims=[16, 16],
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
    decoder_hidden_dim=64,
)

def test_forward_shape():
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    x_hat, z = model(x)
    assert x_hat.shape == (4, 63), f"x_hat: {x_hat.shape}"
    embed_dim = TINY_CFG["embed_dims"][-1]
    assert z.shape == (4, embed_dim), f"z: {z.shape}"
    print("PASSED: test_forward_shape")

def test_gradient_flow():
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    x_hat, z = model(x)
    loss = torch.nn.functional.mse_loss(x_hat, x)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "No gradients!"
    print(f"PASSED: test_gradient_flow ({n_grad} params have gradients)")

def test_autoencoder_contract():
    model = ParTAutoencoder(**TINY_CFG)
    x = torch.randn(4, 63)
    result = model(x)
    assert isinstance(result, tuple) and len(result) == 2, "Must return (x_hat, z)"
    x_hat, z = result
    loss = torch.nn.functional.mse_loss(x_hat, x)
    assert loss.requires_grad, "Loss must be differentiable"
    print("PASSED: test_autoencoder_contract")

if __name__ == "__main__":
    test_forward_shape()
    test_gradient_flow()
    test_autoencoder_contract()
    print("\nAll ParTAutoencoder tests PASSED")
