"""Bump-hunt analysis: background fitting and Z-score calculation.

Implements a simple polynomial background fit to an invariant-mass
distribution and computes a local significance (Z-score) for an excess
in a specified signal window.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class BumpHuntResult:
    """Container for bump-hunt output."""
    z_score: float
    p_value: float
    signal_count: float
    background_estimate: float
    fit_params: Tuple[float, ...] = ()


def _poly_background(x: np.ndarray, *params) -> np.ndarray:
    """Evaluate a polynomial background model."""
    return np.polyval(params, x)


def bump_hunt(
    masses: np.ndarray,
    *,
    mass_window: Optional[Tuple[float, float]] = None,
    num_bins: int = 50,
    poly_degree: int = 4,
) -> BumpHuntResult:
    """Run a bump-hunt on an invariant-mass distribution.

    Parameters
    ----------
    masses : np.ndarray
        1-D array of reconstructed invariant masses (GeV).
    mass_window : tuple[float, float] | None
        ``(low, high)`` edges of the signal window in GeV.  If *None* the
        window is centred on the bin with the largest excess over the smooth
        background and spans ±1 bin.
    num_bins : int
        Number of histogram bins.
    poly_degree : int
        Degree of the polynomial used for background fitting.

    Returns
    -------
    BumpHuntResult
        Z-score, p-value, observed signal count, and background estimate.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if len(masses) < 10:
        warnings.warn("Too few events for meaningful bump-hunt.", stacklevel=2)
        return BumpHuntResult(
            z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0
        )

    counts, bin_edges = np.histogram(masses, bins=num_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # Exclude signal window from the fit if specified
    if mass_window is not None:
        lo, hi = mass_window
        mask = (bin_centres < lo) | (bin_centres > hi)
    else:
        mask = np.ones(len(bin_centres), dtype=bool)

    try:
        fit_params = np.polyfit(
            bin_centres[mask], counts[mask].astype(float), deg=poly_degree
        )
    except (np.linalg.LinAlgError, ValueError):
        warnings.warn("Polynomial fit failed; returning null result.", stacklevel=2)
        return BumpHuntResult(
            z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0
        )

    bg_estimate_all = _poly_background(bin_centres, *fit_params)
    bg_estimate_all = np.maximum(bg_estimate_all, 0.0)  # prevent negative bg

    if mass_window is not None:
        lo, hi = mass_window
        win = (bin_centres >= lo) & (bin_centres <= hi)
    else:
        # Auto-detect: largest excess bin ± 1
        excess = counts - bg_estimate_all
        peak_bin = int(np.argmax(excess))
        lo_idx = max(peak_bin - 1, 0)
        hi_idx = min(peak_bin + 1, len(bin_centres) - 1)
        win = np.zeros(len(bin_centres), dtype=bool)
        win[lo_idx : hi_idx + 1] = True

    n_obs = float(counts[win].sum())
    n_bg = float(bg_estimate_all[win].sum())

    if n_bg <= 0:
        z_score = 0.0
        p_value = 1.0
    else:
        # Li-Ma-like approximation:  Z ≈ (n_obs - n_bg) / sqrt(n_bg)
        z_score = (n_obs - n_bg) / np.sqrt(n_bg)
        # Two-sided p-value from Gaussian
        p_value = float(stats.norm.sf(abs(z_score)) * 2)

    return BumpHuntResult(
        z_score=round(z_score, 4),
        p_value=round(p_value, 6),
        signal_count=round(n_obs, 1),
        background_estimate=round(n_bg, 1),
        fit_params=tuple(float(p) for p in fit_params),
    )
