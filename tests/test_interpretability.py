"""Check particle-level observable extraction and the summary plot."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.analysis.interpretability import (
    group_physics_features,
    plot_interpretability_features,
)


def _toy_events(n_events: int = 100) -> np.ndarray:
    particles = np.zeros((n_events, 6, 3), dtype=np.float32)
    particles[:, 0] = (100.0, 0.2, 0.0)
    particles[:, 1] = (40.0, 0.25, 0.12)
    particles[:, 2] = (20.0, 0.1, -0.15)
    particles[:, 3] = (90.0, -0.2, np.pi)
    particles[:, 4] = (35.0, -0.25, np.pi - 0.1)
    particles[:, 5] = (15.0, -0.1, -np.pi + 0.12)
    return particles.reshape(n_events, -1)


def test_physics_feature_groups_and_plot(tmp_path):
    events = _toy_events()
    scores = np.linspace(0.0, 1.0, len(events))
    grouped = group_physics_features(events, scores, n_particles=6)
    assert len(grouped["top_1_percent"]["m_jj"]) == 1
    assert len(grouped["top_5_percent"]["m_j1"]) == 5
    assert len(grouped["bottom_50_percent"]["constituent_multiplicity"]) == 50
    assert np.all(grouped["top_5_percent"]["m_jj"] > 0)
    output = tmp_path / "interpretability_features.png"
    plot_interpretability_features(grouped, output, bins=10)
    assert output.exists()
