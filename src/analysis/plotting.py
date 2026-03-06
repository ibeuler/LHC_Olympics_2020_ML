"""Plotting utilities for LHC Olympics 2020 analysis.

All functions save figures to disk and optionally display them.
Uses mplhep for CMS/ATLAS-style aesthetics when available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt

# Try to import mplhep for physics-style plots
try:
    import mplhep
    _MPLHEP_AVAILABLE = True
except ImportError:
    mplhep = None  # type: ignore
    _MPLHEP_AVAILABLE = False


def set_style() -> None:
    """Apply CMS-style plotting via mplhep (falls back to default)."""
    if _MPLHEP_AVAILABLE:
        mplhep.style.use("CMS")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")


def savefig(path: Path, *, dpi: int = 150) -> None:
    """Save current figure, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()



def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Path,
) -> None:
    """Plot training and validation loss curves."""
    set_style()
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "o-", label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, "s-", label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    savefig(save_path)



def plot_mass_distribution(
    masses: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    *,
    num_bins: int = 50,
    title: str = "Invariant Mass Distribution",
) -> None:
    """Plot a histogrammed mass distribution, optionally coloured by prediction."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if predictions is not None:
        mask_bg = predictions == 0
        mask_sig = predictions == 1
        ax.hist(
            masses[mask_bg], bins=num_bins, alpha=0.7,
            label="Background", color="steelblue",
        )
        ax.hist(
            masses[mask_sig], bins=num_bins, alpha=0.7,
            label="Signal", color="crimson",
        )
        ax.legend()
    else:
        ax.hist(masses, bins=num_bins, alpha=0.7, color="steelblue")

    ax.set_xlabel("Invariant Mass [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(title)

    if save_path:
        savefig(save_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    save_path: Path,
) -> float:
    """Plot ROC curve and return AUC.

    Parameters
    ----------
    y_true : array of int
        Ground truth labels (0 = background, 1 = signal).
    y_score : array of float
        Model scores / probabilities for the positive class.
    save_path : Path
        Where to save the figure.

    Returns
    -------
    float
        Area under the ROC curve.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    set_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    savefig(save_path)
    return roc_auc


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    *,
    num_bins: int = 60,
    title: str = "Anomaly Score Distribution",
) -> None:
    """Plot distribution of anomaly scores (reconstruction loss)."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    if labels is not None:
        mask_bg = labels == 0
        mask_sig = labels == 1
        ax.hist(
            scores[mask_bg], bins=num_bins, alpha=0.7, density=True,
            label="Background", color="steelblue",
        )
        ax.hist(
            scores[mask_sig], bins=num_bins, alpha=0.7, density=True,
            label="Signal", color="crimson",
        )
        ax.legend()
    else:
        ax.hist(scores, bins=num_bins, alpha=0.7, density=True, color="steelblue")

    ax.set_xlabel("Anomaly Score (Reconstruction Loss)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        savefig(save_path)
