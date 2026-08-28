"""Collider-physics metrics for binary classifier and anomaly scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve


DEFAULT_SIGNAL_EFFICIENCIES = (0.2, 0.3, 0.5)


@dataclass(frozen=True)
class HEPMetrics:
    """A compact summary of the ROC quantities used in HEP studies.

    A test sample can contain a cut with no surviving background events. This
    does not prove zero background efficiency, so rejection and SIC use the
    sample's measurable limit, ``1 / n_background``. The limit is stored with
    the result so downstream tables remain unambiguous.
    """

    auc: float
    accuracy: float
    optimal_threshold: float
    max_sic: float
    signal_efficiency_at_max_sic: float
    background_efficiency_at_max_sic: float
    background_efficiency_floor: float
    background_rejection: dict[float, float]
    background_efficiency: dict[float, float]

    def as_dict(self) -> dict[str, float]:
        row: dict[str, float] = {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "optimal_threshold": self.optimal_threshold,
            "max_sic": self.max_sic,
            "signal_efficiency_at_max_sic": self.signal_efficiency_at_max_sic,
            "background_efficiency_at_max_sic": self.background_efficiency_at_max_sic,
            "background_efficiency_floor": self.background_efficiency_floor,
        }
        for efficiency in sorted(self.background_rejection):
            token = _efficiency_token(efficiency)
            row[f"background_rejection_at_eff_sig_{token}"] = self.background_rejection[efficiency]
            row[f"background_efficiency_at_eff_sig_{token}"] = self.background_efficiency[efficiency]
        return row


def _efficiency_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate_binary_inputs(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.size != scores.size:
        raise ValueError("y_true and y_score must contain the same number of events")
    if labels.size == 0:
        raise ValueError("Cannot compute metrics for an empty sample")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score contains NaN or infinite values")
    unique = np.unique(labels)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError(f"Both binary classes 0 and 1 are required; found {unique.tolist()}")
    return labels.astype(np.int64), scores


def background_efficiency_at_signal_efficiency(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_efficiency: float,
) -> float:
    """Interpolate background efficiency at a chosen signal efficiency.

    When several thresholds give the same signal efficiency, the one with the
    lowest background efficiency is used.
    """
    if not 0.0 <= target_efficiency <= 1.0:
        raise ValueError("target_efficiency must be between 0 and 1")
    order = np.argsort(tpr, kind="stable")
    tpr_sorted = np.asarray(tpr, dtype=float)[order]
    fpr_sorted = np.asarray(fpr, dtype=float)[order]
    unique_tpr = np.unique(tpr_sorted)
    best_fpr = np.array([np.min(fpr_sorted[tpr_sorted == value]) for value in unique_tpr])
    return float(np.interp(target_efficiency, unique_tpr, best_fpr))


def significance_improvement_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Build the SIC curve and return it with the ROC points and FPR limit."""
    labels, scores = _validate_binary_inputs(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    efficiency_floor = 1.0 / int(np.count_nonzero(labels == 0))
    sic = tpr / np.sqrt(np.maximum(fpr, efficiency_floor))
    return fpr, tpr, sic, thresholds, efficiency_floor


def compute_hep_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    signal_efficiencies: Iterable[float] = DEFAULT_SIGNAL_EFFICIENCIES,
) -> HEPMetrics:
    """Calculate AUC, best accuracy, background rejection, and maximum SIC."""
    labels, scores = _validate_binary_inputs(y_true, y_score)
    targets = tuple(float(value) for value in signal_efficiencies)
    if not targets:
        raise ValueError("At least one signal efficiency must be requested")

    fpr, tpr, sic, thresholds, efficiency_floor = significance_improvement_curve(labels, scores)
    n_background = int(np.count_nonzero(labels == 0))
    regularized_fpr = np.maximum(fpr, efficiency_floor)
    max_index = int(np.nanargmax(sic))

    # Report the threshold with the highest event-level accuracy on this sample.
    positives = int(np.count_nonzero(labels == 1))
    negatives = n_background
    accuracies = (tpr * positives + (1.0 - fpr) * negatives) / labels.size
    accuracy_index = int(np.nanargmax(accuracies))
    threshold = float(thresholds[accuracy_index])
    predictions = (scores >= threshold).astype(np.int64)

    efficiencies: dict[float, float] = {}
    rejections: dict[float, float] = {}
    for target in targets:
        raw_efficiency = background_efficiency_at_signal_efficiency(fpr, tpr, target)
        regularized_efficiency = max(raw_efficiency, efficiency_floor)
        efficiencies[target] = regularized_efficiency
        rejections[target] = 1.0 / regularized_efficiency

    return HEPMetrics(
        auc=float(auc(fpr, tpr)),
        accuracy=float(accuracy_score(labels, predictions)),
        optimal_threshold=threshold,
        max_sic=float(sic[max_index]),
        signal_efficiency_at_max_sic=float(tpr[max_index]),
        background_efficiency_at_max_sic=float(regularized_fpr[max_index]),
        background_efficiency_floor=efficiency_floor,
        background_rejection=rejections,
        background_efficiency=efficiencies,
    )
