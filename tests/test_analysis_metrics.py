"""Check the HEP metrics and percentile-based anomaly selections."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.metrics import compute_hep_metrics, significance_improvement_curve
from src.analysis.interpretability import select_anomaly_score_regions


def test_perfect_classifier_metrics_are_finite():
    labels = np.array([0] * 10 + [1] * 10)
    scores = np.array([0.01] * 10 + [0.99] * 10)
    metrics = compute_hep_metrics(labels, scores)
    assert metrics.auc == 1.0
    assert metrics.accuracy == 1.0
    assert metrics.background_efficiency_floor == 0.1
    assert metrics.background_rejection[0.3] == 10.0
    assert np.isfinite(metrics.max_sic)
    fpr, tpr, sic, thresholds, floor = significance_improvement_curve(labels, scores)
    assert fpr.shape == tpr.shape == sic.shape == thresholds.shape
    assert floor == 0.1


def test_anomaly_regions_have_exact_rank_counts():
    scores = np.arange(100, dtype=float)
    regions = select_anomaly_score_regions(scores)
    assert regions["top_1_percent"].sum() == 1
    assert regions["top_5_percent"].sum() == 5
    assert regions["bottom_50_percent"].sum() == 50
    assert regions["top_1_percent"][99]
    assert regions["bottom_50_percent"][0]
