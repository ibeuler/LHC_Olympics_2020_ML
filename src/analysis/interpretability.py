"""Connect event anomaly scores to familiar collider observables.

LHCO tensors store massless constituents as ``(pT, eta, phi)``. To keep this
analysis dependency-free, the module reconstructs two seeded cone jets and
computes masses, leading-jet N-subjettiness, and constituent multiplicity. The
axis choices are fixed and documented so results can be reproduced exactly.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.plotting import savefig, set_style


FEATURE_LABELS = {
    "m_jj": r"$m_{jj}$ [GeV]",
    "m_j1": r"$m_{j1}$ [GeV]",
    "m_j2": r"$m_{j2}$ [GeV]",
    "tau21": r"Leading jet $\tau_{21}$",
    "tau32": r"Leading jet $\tau_{32}$",
    "constituent_multiplicity": "Constituent multiplicity",
}


def select_anomaly_score_regions(scores: np.ndarray) -> dict[str, np.ndarray]:
    """Select the top 1%, top 5%, and bottom 50% by anomaly-score rank.

    The top 1% is intentionally part of the top 5%. Stable sorting gives tied
    scores a deterministic ordering.
    """
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must contain at least one event")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores contains NaN or infinite values")
    order = np.argsort(values, kind="stable")
    n_events = values.size
    n_top_1 = max(1, int(np.ceil(0.01 * n_events)))
    n_top_5 = max(1, int(np.ceil(0.05 * n_events)))
    n_bottom_50 = max(1, int(np.ceil(0.50 * n_events)))
    regions = {
        "top_1_percent": np.zeros(n_events, dtype=bool),
        "top_5_percent": np.zeros(n_events, dtype=bool),
        "bottom_50_percent": np.zeros(n_events, dtype=bool),
    }
    regions["top_1_percent"][order[-n_top_1:]] = True
    regions["top_5_percent"][order[-n_top_5:]] = True
    regions["bottom_50_percent"][order[:n_bottom_50]] = True
    return regions


def _delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _delta_r2(eta: np.ndarray, phi: np.ndarray, axis_eta: np.ndarray, axis_phi: np.ndarray):
    return (eta - axis_eta) ** 2 + _delta_phi(phi, axis_phi) ** 2


def _weighted_axis(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, membership: np.ndarray):
    weight = np.where(membership, pt, 0.0)
    total = weight.sum(axis=1)
    safe = np.maximum(total, 1e-12)
    axis_eta = (weight * eta).sum(axis=1) / safe
    sin_phi = (weight * np.sin(phi)).sum(axis=1)
    cos_phi = (weight * np.cos(phi)).sum(axis=1)
    axis_phi = np.arctan2(sin_phi, cos_phi)
    return axis_eta, axis_phi


def _four_vector_sum(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, membership: np.ndarray):
    selected_pt = np.where(membership, pt, 0.0)
    px = (selected_pt * np.cos(phi)).sum(axis=1)
    py = (selected_pt * np.sin(phi)).sum(axis=1)
    pz = (selected_pt * np.sinh(np.clip(eta, -10.0, 10.0))).sum(axis=1)
    energy = (selected_pt * np.cosh(np.clip(eta, -10.0, 10.0))).sum(axis=1)
    transverse = np.hypot(px, py)
    mass2 = energy * energy - px * px - py * py - pz * pz
    return px, py, pz, energy, transverse, np.sqrt(np.maximum(mass2, 0.0))


def _tau_n(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    membership: np.ndarray,
    n_axes: int,
    jet_radius: float,
) -> np.ndarray:
    """Compute N-subjettiness with the N hardest constituents as fixed axes."""
    n_events, n_particles = pt.shape
    n_axes = min(n_axes, n_particles)
    ranked_pt = np.where(membership, pt, -np.inf)
    indices = np.argsort(ranked_pt, axis=1, kind="stable")[:, -n_axes:]
    axis_eta = np.take_along_axis(eta, indices, axis=1)
    axis_phi = np.take_along_axis(phi, indices, axis=1)
    axis_valid = np.take_along_axis(membership, indices, axis=1)
    distances = np.sqrt(
        (eta[:, None, :] - axis_eta[:, :, None]) ** 2
        + _delta_phi(phi[:, None, :], axis_phi[:, :, None]) ** 2
    )
    distances = np.where(axis_valid[:, :, None], distances, np.inf)
    min_distance = distances.min(axis=1)
    weighted_distance = np.zeros_like(pt, dtype=float)
    np.multiply(pt, min_distance, out=weighted_distance, where=membership)
    numerator = weighted_distance.sum(axis=1)
    denominator = np.where(membership, pt, 0.0).sum(axis=1) * jet_radius
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_events, np.nan, dtype=float),
        where=denominator > 0,
    )


def compute_physics_features(
    events: np.ndarray,
    *,
    n_particles: int | None = None,
    jet_radius: float = 1.0,
) -> dict[str, np.ndarray]:
    """Calculate event observables from flattened massless constituents.

    The two jet seeds are the hardest constituent and the hardest constituent
    outside radius ``R``; the second-hardest is used as a fallback. One pT-weighted
    refinement assigns each constituent to its nearest axis. N-subjettiness uses
    the N hardest constituents of the leading jet as fixed axes with beta=1.
    """
    values = np.asarray(events, dtype=float)
    if values.ndim != 2:
        raise ValueError("events must have shape (n_events, n_features)")
    particles = n_particles if n_particles is not None else values.shape[1] // 3
    if particles <= 0 or values.shape[1] != particles * 3:
        raise ValueError(
            f"Expected n_particles*3 features; got {values.shape[1]} features "
            f"for n_particles={particles}"
        )
    if jet_radius <= 0:
        raise ValueError("jet_radius must be positive")

    constituents = values.reshape(values.shape[0], particles, 3)
    pt = constituents[:, :, 0]
    eta = constituents[:, :, 1]
    phi = constituents[:, :, 2]
    valid = (pt > 0) & np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi)
    pt = np.where(valid, pt, 0.0)
    eta = np.where(valid, eta, 0.0)
    phi = np.where(valid, phi, 0.0)
    row = np.arange(values.shape[0])

    seed1 = np.argmax(pt, axis=1)
    seed1_eta = eta[row, seed1][:, None]
    seed1_phi = phi[row, seed1][:, None]
    outside = valid & (_delta_r2(eta, phi, seed1_eta, seed1_phi) > jet_radius**2)
    seed2 = np.argmax(np.where(outside, pt, -1.0), axis=1)
    no_outside = ~outside.any(axis=1)
    if particles > 1 and np.any(no_outside):
        second_hardest = np.argsort(pt, axis=1, kind="stable")[:, -2]
        seed2[no_outside] = second_hardest[no_outside]

    axis1_eta = eta[row, seed1]
    axis1_phi = phi[row, seed1]
    axis2_eta = eta[row, seed2]
    axis2_phi = phi[row, seed2]
    distance1 = _delta_r2(eta, phi, axis1_eta[:, None], axis1_phi[:, None])
    distance2 = _delta_r2(eta, phi, axis2_eta[:, None], axis2_phi[:, None])
    jet1 = valid & (distance1 <= distance2)
    jet2 = valid & ~jet1

    axis1_eta, axis1_phi = _weighted_axis(pt, eta, phi, jet1)
    axis2_eta, axis2_phi = _weighted_axis(pt, eta, phi, jet2)
    distance1 = _delta_r2(eta, phi, axis1_eta[:, None], axis1_phi[:, None])
    distance2 = _delta_r2(eta, phi, axis2_eta[:, None], axis2_phi[:, None])
    jet1 = valid & (distance1 <= distance2)
    jet2 = valid & ~jet1

    vec1 = _four_vector_sum(pt, eta, phi, jet1)
    vec2 = _four_vector_sum(pt, eta, phi, jet2)
    leading_is_1 = vec1[4] >= vec2[4]
    leading_membership = np.where(leading_is_1[:, None], jet1, jet2)
    leading_mass = np.where(leading_is_1, vec1[5], vec2[5])
    subleading_mass = np.where(leading_is_1, vec2[5], vec1[5])

    total_px = vec1[0] + vec2[0]
    total_py = vec1[1] + vec2[1]
    total_pz = vec1[2] + vec2[2]
    total_energy = vec1[3] + vec2[3]
    dijet_mass2 = total_energy**2 - total_px**2 - total_py**2 - total_pz**2

    tau1 = _tau_n(pt, eta, phi, leading_membership, 1, jet_radius)
    tau2 = _tau_n(pt, eta, phi, leading_membership, 2, jet_radius)
    tau3 = _tau_n(pt, eta, phi, leading_membership, 3, jet_radius)
    tau21 = np.divide(tau2, tau1, out=np.full_like(tau1, np.nan), where=tau1 > 1e-12)
    tau32 = np.divide(tau3, tau2, out=np.full_like(tau2, np.nan), where=tau2 > 1e-12)

    return {
        "m_jj": np.sqrt(np.maximum(dijet_mass2, 0.0)),
        "m_j1": leading_mass,
        "m_j2": subleading_mass,
        "tau21": tau21,
        "tau32": tau32,
        "constituent_multiplicity": valid.sum(axis=1).astype(float),
    }


def group_physics_features(
    events: np.ndarray,
    scores: np.ndarray,
    *,
    n_particles: int | None = None,
    jet_radius: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Calculate the observables once, then split them by score percentile."""
    regions = select_anomaly_score_regions(scores)
    features = compute_physics_features(events, n_particles=n_particles, jet_radius=jet_radius)
    return {
        region: {name: values[mask] for name, values in features.items()}
        for region, mask in regions.items()
    }


def plot_interpretability_features(
    grouped_features: Mapping[str, Mapping[str, np.ndarray]],
    save_path: str | Path = "report/plots/interpretability_features.png",
    *,
    bins: int = 50,
) -> Path:
    """Plot normalized observable distributions for each score percentile."""
    required_regions = ("bottom_50_percent", "top_5_percent", "top_1_percent")
    missing = [name for name in required_regions if name not in grouped_features]
    if missing:
        raise ValueError(f"Missing anomaly-score regions: {missing}")
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    styles = {
        "bottom_50_percent": ("Bottom 50%", "steelblue", "stepfilled", 0.28),
        "top_5_percent": ("Top 5%", "darkorange", "step", 0.95),
        "top_1_percent": ("Top 1%", "crimson", "step", 0.95),
    }

    for axis, feature in zip(axes.flat, FEATURE_LABELS):
        all_values = []
        for region in required_regions:
            values = np.asarray(grouped_features[region][feature], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                all_values.append(values)
        if not all_values:
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
            axis.set_xlabel(FEATURE_LABELS[feature])
            continue
        combined = np.concatenate(all_values)
        low, high = np.quantile(combined, [0.005, 0.995])
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low, high = float(np.min(combined)), float(np.max(combined) + 1.0)
        edges = np.linspace(low, high, bins + 1)
        for region in required_regions:
            label, color, histtype, alpha = styles[region]
            values = np.asarray(grouped_features[region][feature], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                axis.hist(
                    values,
                    bins=edges,
                    density=True,
                    histtype=histtype,
                    linewidth=2.0,
                    alpha=alpha,
                    color=color,
                    label=label,
                )
        axis.set_xlabel(FEATURE_LABELS[feature])
        axis.set_ylabel("Normalized events")
        axis.grid(True, alpha=0.25)

    axes[0, 0].legend(frameon=False)
    fig.suptitle("Physics observables by anomaly-score percentile", y=1.01)
    fig.tight_layout()
    destination = Path(save_path)
    savefig(destination)
    return destination
