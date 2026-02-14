from __future__ import annotations

from pathlib import Path
from typing import Optional


def set_style() -> None:
    """Set plotting style.

    If you use mplhep, call mplhep.style.use here.
    """
    try:
        import mplhep  # noqa: F401
    except Exception:
        return


def savefig(path: Path, *, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
