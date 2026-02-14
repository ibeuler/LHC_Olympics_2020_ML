from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BumpHuntResult:
    z_score: float


def bump_hunt(masses, *, mass_window=None) -> BumpHuntResult:
    """Placeholder bump-hunt routine.

    Typical workflow: fit a smooth background model, compute local p-values and Z-scores.
    """
    raise NotImplementedError("Implement background fit and Z-score calculation.")
