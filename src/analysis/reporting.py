"""Write evaluation results to CSV without silently dropping old columns."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Any


RESULT_COLUMNS = [
    "run_id", "dataset", "model", "model_type", "use_pairwise", "checkpoint",
    "n_events", "n_signal", "n_background", "total_parameters", "trainable_parameters",
    "auc", "accuracy", "optimal_threshold", "max_sic",
    "signal_efficiency_at_max_sic", "background_efficiency_at_max_sic",
    "background_efficiency_floor",
    "background_rejection_at_eff_sig_0p2", "background_rejection_at_eff_sig_0p3",
    "background_rejection_at_eff_sig_0p5",
    "background_efficiency_at_eff_sig_0p2", "background_efficiency_at_eff_sig_0p3",
    "background_efficiency_at_eff_sig_0p5",
    "score_mean", "score_std", "score_median", "score_q95", "score_q99",
    "z_score", "p_value", "signal_count", "background_estimate", "notes",
]


def write_results_table(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write the standard result columns first, followed by any custom fields."""
    materialized = [dict(row) for row in rows]
    extras = sorted({key for row in materialized for key in row if key not in RESULT_COLUMNS})
    fieldnames = RESULT_COLUMNS + extras
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return destination


def update_results_summary(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Add new rows to an existing summary and update its schema if needed."""
    destination = Path(path)
    existing: list[dict[str, Any]] = []
    if destination.exists():
        with destination.open("r", newline="", encoding="utf-8") as handle:
            existing.extend(csv.DictReader(handle))
    existing.extend(dict(row) for row in rows)
    return write_results_table(existing, destination)
