"""Tests for ParTClassifier (Version C)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.part_classifier import ParTClassifier

TINY_CFG = dict(
    input_dim=63,
    n_particles=21,
    num_classes=2,
    embed_dims=[32, 32],
    pair_embed_dims=[16, 16],
    num_heads=2,
    num_layers=2,
    num_cls_layers=1,
)

def test_forward_shape():
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    assert logits.shape == (4, 2), f"logits: {logits.shape}"
    print("PASSED: test_forward_shape")

def test_gradient_flow():
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 0, "No gradients!"
    print(f"PASSED: test_gradient_flow ({n_grad} params have gradients)")

def test_classifier_contract():
    model = ParTClassifier(**TINY_CFG)
    x = torch.randn(4, 63)
    logits = model(x)
    assert isinstance(logits, torch.Tensor), "Must return a single tensor"
    assert logits.dim() == 2, f"Expected 2D, got {logits.dim()}D"
    assert logits.size(1) == 2, f"Expected 2 classes, got {logits.size(1)}"
    print("PASSED: test_classifier_contract")

if __name__ == "__main__":
    test_forward_shape()
    test_gradient_flow()
    test_classifier_contract()
    print("\nAll ParTClassifier tests PASSED")
