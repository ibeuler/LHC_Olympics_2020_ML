"""Bump-hunt analysis: background fitting and Z-score calculation.

Implements a polynomial background fit to an invariant-mass distribution
and a sliding-window scan that finds the most significant local excess
without fixing the signal window in advance.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BumpHuntResult:
    """Container for bump-hunt output."""
    z_score: float
    p_value: float
    signal_count: float
    background_estimate: float
    best_window: Tuple[float, float] = (0.0, 0.0)
    fit_params: Tuple[float, ...] = ()


def _poly_background(x: np.ndarray, *params) -> np.ndarray:
    """Evaluate a polynomial background model."""
    return np.polyval(params, x)


def _fit_background(
    bin_centres: np.ndarray,
    counts: np.ndarray,
    exclude_mask: np.ndarray,
    poly_degree: int,
) -> Optional[np.ndarray]:
    """Fit polynomial background excluding bins in exclude_mask.
    Returns fitted background at all bin_centres, or None on failure.
    """
    mask = ~exclude_mask
    if mask.sum() < poly_degree + 1:
        return None
    try:
        params = np.polyfit(bin_centres[mask], counts[mask].astype(float), deg=poly_degree)
    except (np.linalg.LinAlgError, ValueError):
        return None
    bg = _poly_background(bin_centres, *params)
    return np.maximum(bg, 0.0)


def _z_score(n_obs: float, n_bg: float) -> float:
    if n_bg <= 0:
        return 0.0
    return (n_obs - n_bg) / np.sqrt(n_bg)


def bump_hunt(
    masses: np.ndarray,
    *,
    mass_window: Optional[Tuple[float, float]] = None,
    num_bins: int = 50,
    poly_degree: int = 4,
    min_window_bins: int = 2,
    max_window_bins: int = 10,
) -> BumpHuntResult:
    """Run a bump-hunt on an invariant-mass distribution.

    When *mass_window* is None (default) a sliding-window scan tests all
    window positions and widths between *min_window_bins* and *max_window_bins*
    bins wide and returns the result for the most significant window.  This is
    more robust than fixing the window or using a single ±1-bin heuristic.

    Parameters
    ----------
    masses : np.ndarray
        1-D array of reconstructed invariant masses (or anomaly scores used
        as a proxy).
    mass_window : tuple[float, float] | None
        If given, skip the scan and evaluate only this ``(low, high)`` window.
    num_bins : int
        Number of histogram bins.
    poly_degree : int
        Degree of the polynomial background model.
    min_window_bins, max_window_bins : int
        Range of window widths (in bins) to scan.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if len(masses) < 10:
        warnings.warn("Too few events for meaningful bump-hunt.", stacklevel=2)
        return BumpHuntResult(z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0)

    counts, bin_edges = np.histogram(masses, bins=num_bins)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centres)

    if mass_window is not None:
        lo, hi = mass_window
        win_mask = (bin_centres >= lo) & (bin_centres <= hi)
        bg = _fit_background(bin_centres, counts, win_mask, poly_degree)
        if bg is None:
            warnings.warn("Polynomial fit failed; returning null result.", stacklevel=2)
            return BumpHuntResult(z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0)
        n_obs = float(counts[win_mask].sum())
        n_bg = float(bg[win_mask].sum())
        z = _z_score(n_obs, n_bg)
        p_val = float(stats.norm.sf(abs(z)) * 2)
        return BumpHuntResult(
            z_score=round(z, 4),
            p_value=round(p_val, 6),
            signal_count=round(n_obs, 1),
            background_estimate=round(n_bg, 1),
            best_window=(lo, hi),
        )

    # Sliding-window scan
    best_z = 0.0
    best_result: Optional[BumpHuntResult] = None

    # Fit global background once (no window excluded) as fast initialisation
    global_bg = _fit_background(bin_centres, counts, np.zeros(n_bins, dtype=bool), poly_degree)
    if global_bg is None:
        warnings.warn("Global polynomial fit failed; returning null result.", stacklevel=2)
        return BumpHuntResult(z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0)

    for width in range(min_window_bins, min(max_window_bins + 1, n_bins)):
        for start in range(n_bins - width + 1):
            end = start + width
            win_mask = np.zeros(n_bins, dtype=bool)
            win_mask[start:end] = True

            # Re-fit background excluding this window for an unbiased estimate
            bg = _fit_background(bin_centres, counts, win_mask, poly_degree)
            if bg is None:
                bg = global_bg  # fall back to global fit

            n_obs = float(counts[win_mask].sum())
            n_bg = float(bg[win_mask].sum())
            z = _z_score(n_obs, n_bg)

            if abs(z) > abs(best_z):
                best_z = z
                p_val = float(stats.norm.sf(abs(z)) * 2)
                best_result = BumpHuntResult(
                    z_score=round(z, 4),
                    p_value=round(p_val, 6),
                    signal_count=round(n_obs, 1),
                    background_estimate=round(n_bg, 1),
                    best_window=(float(bin_edges[start]), float(bin_edges[end])),
                )

    if best_result is None:
        return BumpHuntResult(z_score=0.0, p_value=1.0, signal_count=0.0, background_estimate=0.0)
    return best_result
