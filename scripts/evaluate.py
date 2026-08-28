"""Evaluate trained LHCO models and create a physics-focused report.

Use the original single-checkpoint interface::

    python scripts/evaluate.py --checkpoint best_model.pt --config configs/config.yaml

Or compare the full autoencoder ablation suite::

    python scripts/evaluate.py \
      --model SimpleAE configs/config.yaml outputs/simple.pt \
      --model ParTAE-no-U configs/part_autoencoder_no_pairwise.yaml outputs/no_u.pt \
      --model ParTAE-with-U configs/part_autoencoder.yaml outputs/with_u.pt \
      --lhc-background-data data/raw/events_LHCO2020_backgroundMC_Pythia.h5
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.interpretability import (
    compute_physics_features,
    plot_interpretability_features,
    select_anomaly_score_regions,
)
from src.analysis.metrics import compute_hep_metrics
from src.analysis.model_complexity import count_parameters, parameter_count_table
from src.analysis.plotting import (
    plot_anomaly_scores,
    plot_roc_comparison,
    plot_roc_curve,
    plot_score_comparison,
)
from src.analysis.reporting import update_results_summary, write_results_table
from src.data.dataset import LHCDataset, SyntheticLHCDataset, build_dataloaders
from src.utils.config import get_model, load_config


@dataclass
class ModelSpec:
    name: str
    config_path: Path | None
    checkpoint: Path
    config: dict[str, Any]
    model_type: str
    input_dim: int


def _sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _model_short_name(model_type: str) -> str:
    return {
        "autoencoder": "SimpleAE",
        "classifier": "MLPClassifier",
        "part_autoencoder": "ParTAE",
        "part_classifier": "ParTClassifier",
    }.get(model_type.lower(), model_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LHCO models with HEP metrics.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Single-model .pt/.pth checkpoint")
    parser.add_argument("--config", type=Path, default=None, help="Config for --checkpoint")
    parser.add_argument(
        "--model",
        nargs=3,
        action="append",
        metavar=("NAME", "CONFIG", "CHECKPOINT"),
        help="Repeat for an ablation suite: display name, YAML config, checkpoint",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Backward-compatible alias for --lhc-background-data",
    )
    parser.add_argument("--lhc-background-data", type=Path, default=None, help="Real LHCO background HDF5")
    parser.add_argument("--synthetic-samples", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("report"), help="Report root directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--model-type", type=str, default=None,
        choices=["autoencoder", "classifier", "part_autoencoder", "part_classifier"],
        help="Override model type in single-checkpoint mode",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional run identifier")
    parser.add_argument("--skip-interpretability", action="store_true")
    return parser.parse_args()


def _build_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    if args.model:
        if args.checkpoint is not None:
            raise ValueError("Use either repeated --model entries or --checkpoint, not both")
        for name, config_value, checkpoint_value in args.model:
            config_path = Path(config_value)
            config = load_config(config_path)
            model_cfg = config.get("model", {})
            model_type = str(model_cfg.get("type", "autoencoder")).lower()
            specs.append(ModelSpec(
                name=name,
                config_path=config_path,
                checkpoint=Path(checkpoint_value),
                config=config,
                model_type=model_type,
                input_dim=int(model_cfg.get("input_dim", 128)),
            ))
    else:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or at least one --model NAME CONFIG CHECKPOINT")
        config = load_config(args.config) if args.config is not None else {}
        model_cfg = config.get("model", {})
        model_type = (args.model_type or model_cfg.get("type", "autoencoder")).lower()
        specs.append(ModelSpec(
            name=_model_short_name(model_type),
            config_path=args.config,
            checkpoint=args.checkpoint,
            config=config,
            model_type=model_type,
            input_dim=int(model_cfg.get("input_dim", 128)),
        ))
    return specs


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, nn.Module):
            return candidate.state_dict()
        if isinstance(candidate, dict) and candidate:
            payload = candidate
            break
    if not payload or not all(isinstance(key, str) for key in payload):
        raise ValueError("Checkpoint does not contain a recognizable state_dict")
    state = dict(payload)
    for prefix in ("module.", "_orig_mod."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> nn.Module:
    """Load a raw or wrapped state dictionary from a ``.pt`` or ``.pth`` file."""
    checkpoint = Path(path)
    if checkpoint.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(f"Expected a .pt or .pth checkpoint, got: {checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    state = _extract_state_dict(payload)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as first_error:
        # Frameworks sometimes prefix every key with `model.`. Try the original
        # keys first because ParTClassifier itself has a child named `model`.
        if state and all(key.startswith("model.") for key in state):
            stripped = {key[len("model."):]: value for key, value in state.items()}
            try:
                model.load_state_dict(stripped, strict=True)
            except RuntimeError:
                raise RuntimeError(
                    f"Checkpoint {checkpoint} is incompatible with {model.__class__.__name__}. "
                    "Verify input_dim, architecture settings, and use_pairwise in the YAML config."
                ) from first_error
        else:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is incompatible with {model.__class__.__name__}. "
                "Verify input_dim, architecture settings, and use_pairwise in the YAML config."
            ) from first_error
    return model.to(device)


def _make_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    config = copy.deepcopy(spec.config)
    config.setdefault("model", {})
    config.setdefault("train", {})
    config["model"]["type"] = spec.model_type
    config["model"]["input_dim"] = spec.input_dim
    config["train"]["use_amp"] = device.type == "cuda" and config["train"].get("use_amp", True)
    return load_checkpoint(get_model(config), spec.checkpoint, device)


def collect_scores(
    model: nn.Module,
    dataloader,
    device: torch.device,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return scores and labels in DataLoader order."""
    model.eval()
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    is_autoencoder = model_type in {"autoencoder", "part_autoencoder"}
    with torch.no_grad():
        for x, labels in dataloader:
            x = x.to(device, non_blocking=True)
            if is_autoencoder:
                reconstruction, _ = model(x)
                scores = (reconstruction - x).square().flatten(1).mean(dim=1)
            else:
                logits = model(x)
                scores = torch.softmax(logits, dim=1)[:, 1]
            score_batches.append(scores.detach().cpu().numpy())
            label_batches.append(labels.detach().cpu().numpy())
    if not score_batches:
        raise ValueError("Evaluation dataset is empty")
    return np.concatenate(score_batches), np.concatenate(label_batches)


def _collect_grouped_features(
    dataloader,
    scores: np.ndarray,
    *,
    n_particles: int,
    jet_radius: float = 1.0,
) -> dict[str, dict[str, np.ndarray]]:
    regions = select_anomaly_score_regions(scores)
    grouped: dict[str, dict[str, list[np.ndarray]]] = {name: {} for name in regions}
    cursor = 0
    for x, _ in dataloader:
        batch = x.detach().cpu().numpy()
        stop = cursor + len(batch)
        for region_name, mask in regions.items():
            local_mask = mask[cursor:stop]
            if not np.any(local_mask):
                continue
            features = compute_physics_features(
                batch[local_mask], n_particles=n_particles, jet_radius=jet_radius
            )
            for feature_name, values in features.items():
                grouped[region_name].setdefault(feature_name, []).append(values)
        cursor = stop
    if cursor != len(scores):
        raise RuntimeError("Dataloader ordering/length changed between scoring and feature extraction")
    return {
        region: {
            feature: np.concatenate(parts) if parts else np.array([], dtype=float)
            for feature, parts in feature_map.items()
        }
        for region, feature_map in grouped.items()
    }


def _result_row(
    *, spec: ModelSpec, model: nn.Module, dataset_name: str,
    scores: np.ndarray, labels: np.ndarray, run_id: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model": spec.name,
        "model_type": spec.model_type,
        "use_pairwise": (
            spec.config.get("model", {}).get("use_pairwise", True)
            if spec.model_type.startswith("part_") else ""
        ),
        "checkpoint": str(spec.checkpoint),
        "n_events": len(scores),
        "n_signal": int(np.count_nonzero(labels == 1)),
        "n_background": int(np.count_nonzero(labels == 0)),
        **count_parameters(model),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_median": float(np.median(scores)),
        "score_q95": float(np.quantile(scores, 0.95)),
        "score_q99": float(np.quantile(scores, 0.99)),
    }
    if np.any(labels == 0) and np.any(labels == 1):
        row.update(compute_hep_metrics(labels, scores).as_dict())
        row["notes"] = "Binary signal/background sample; zero-FPR points use the recorded sample limit."
    else:
        row["notes"] = "Single-class sample; ROC, rejection, and SIC are undefined."
    return row


def main() -> None:
    args = parse_args()
    specs = _build_specs(args)
    if args.synthetic_samples < 2:
        raise ValueError("--synthetic-samples must be at least 2")
    device = torch.device(args.device)
    background_path = args.lhc_background_data or args.data
    report_root = args.output
    plots_dir = report_root / "plots"
    tables_dir = report_root / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_id = _sanitize_token(args.tag or datetime.now().strftime("evaluation_%Y%m%d_%H%M%S"))

    real_dataset = LHCDataset(background_path) if background_path is not None else None
    synthetic_rows: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    synthetic_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    synthetic_scores: dict[str, np.ndarray] = {}
    background_scores: dict[str, np.ndarray] = {}
    loaded_models: dict[str, nn.Module] = {}
    loaders: dict[tuple[str, str], Any] = {}

    for spec in specs:
        print(f"\nEvaluating {spec.name} from {spec.checkpoint}")
        model = _make_model(spec, device)
        loaded_models[spec.name] = model
        synthetic_dataset = SyntheticLHCDataset(
            n_samples=args.synthetic_samples,
            input_dim=spec.input_dim,
            seed=int(spec.config.get("seed", 42)),
        )
        _, synthetic_loader = build_dataloaders(
            synthetic_dataset, batch_size=args.batch_size, val_fraction=1.0,
            seed=int(spec.config.get("seed", 42)),
        )
        loaders[(spec.name, "synthetic")] = synthetic_loader
        scores, labels = collect_scores(model, synthetic_loader, device, spec.model_type)
        synthetic_rows.append(_result_row(
            spec=spec, model=model, dataset_name="synthetic", scores=scores,
            labels=labels, run_id=run_id,
        ))
        synthetic_curves[spec.name] = (labels, scores)
        synthetic_scores[spec.name] = scores
        tag = _sanitize_token(spec.name)
        plot_roc_curve(labels, scores, plots_dir / f"synthetic_roc_{tag}.png")
        plot_anomaly_scores(scores, labels, plots_dir / f"synthetic_scores_{tag}.png")

        if real_dataset is not None:
            if real_dataset.input_dim != spec.input_dim:
                raise ValueError(
                    f"{spec.name} expects input_dim={spec.input_dim}, but "
                    f"{background_path} provides {real_dataset.input_dim}."
                )
            _, real_loader = build_dataloaders(
                real_dataset, batch_size=args.batch_size, val_fraction=1.0,
                seed=int(spec.config.get("seed", 42)),
            )
            loaders[(spec.name, "lhc_background")] = real_loader
            real_scores, real_labels = collect_scores(model, real_loader, device, spec.model_type)
            background_rows.append(_result_row(
                spec=spec, model=model, dataset_name="lhc_background", scores=real_scores,
                labels=real_labels, run_id=run_id,
            ))
            background_scores[spec.name] = real_scores
            plot_anomaly_scores(
                real_scores, None, plots_dir / f"lhc_background_scores_{tag}.png",
                title=f"{spec.name}: LHC background score",
            )

    write_results_table(synthetic_rows, tables_dir / "synthetic_validation.csv")
    write_results_table(background_rows, tables_dir / "lhc_background_evaluation.csv")
    update_results_summary(synthetic_rows + background_rows, tables_dir / "results_summary.csv")
    parameter_count_table(loaded_models).to_csv(tables_dir / "model_parameter_counts.csv", index=False)
    plot_roc_comparison(
        synthetic_curves, plots_dir / "synthetic_roc_comparison.png",
        title="Synthetic validation: ROC comparison",
    )
    plot_score_comparison(
        synthetic_scores, plots_dir / "synthetic_score_comparison.png",
        title="Synthetic validation: score comparison",
    )
    if background_scores:
        plot_score_comparison(
            background_scores, plots_dir / "lhc_background_score_comparison.png",
            title="LHC background: score comparison",
        )

    if not args.skip_interpretability:
        candidates = [
            spec for spec in specs
            if spec.model_type.startswith("part_")
            and spec.input_dim % 3 == 0
            and spec.config.get("model", {}).get("use_pairwise", True)
        ]
        if not candidates:
            candidates = [spec for spec in specs if spec.input_dim % 3 == 0]
        if candidates:
            chosen = candidates[0]
            dataset_key = "lhc_background" if real_dataset is not None else "synthetic"
            loader = loaders[(chosen.name, dataset_key)]
            scores = background_scores.get(chosen.name, synthetic_scores[chosen.name])
            n_particles = int(chosen.config.get("model", {}).get("n_particles", chosen.input_dim // 3))
            jet_radius = float(chosen.config.get("preprocessing", {}).get("jet_radius", 1.0))
            grouped = _collect_grouped_features(
                loader, scores, n_particles=n_particles, jet_radius=jet_radius
            )
            plot_interpretability_features(grouped, plots_dir / "interpretability_features.png")
            print(f"Interpretability plot uses {chosen.name} on {dataset_key}.")
        else:
            print("Interpretability skipped: no model input is a flat (pT, eta, phi) tensor.")

    print(f"\nEvaluation complete. Tables: {tables_dir}; plots: {plots_dir}")


if __name__ == "__main__":
    main()
