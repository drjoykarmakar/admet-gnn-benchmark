#!/usr/bin/env python
"""Run the Random Forest benchmark on saved random and scaffold splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models_baseline import (  # noqa: E402
    build_candidate_grid,
    load_feature_archive,
    run_baseline_for_split,
)


def _resolve(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI options for the classical baseline."""
    parser = argparse.ArgumentParser(description="Run Random Forest ADMET baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML experiment configuration.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("random", "scaffold"),
        default=("random", "scaffold"),
        help="Split methods to evaluate.",
    )
    return parser


def main() -> None:
    """Train, select, evaluate, and save the classical benchmark outputs."""
    args = build_arg_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text())

    seed = int(config.get("seed", 42))
    data_config = config.get("data", {})
    baseline_config = config.get("baseline", {})
    results_config = config.get("results", {})

    features_path = _resolve(
        ROOT,
        data_config.get(
            "features_npz", "data/processed/solubility_aqsoldb_features.npz"
        ),
    )
    split_dir = _resolve(ROOT, data_config.get("split_dir", "data/processed/splits"))
    tables_dir = _resolve(ROOT, results_config.get("tables_dir", "results/tables"))
    predictions_dir = _resolve(
        ROOT, results_config.get("predictions_dir", "results/predictions")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    archive = load_feature_archive(features_path)
    candidates = build_candidate_grid(baseline_config)
    n_jobs = int(baseline_config.get("n_jobs", -1))

    summaries: list[dict[str, object]] = []
    tuning_frames: list[pd.DataFrame] = []

    for split_method in args.splits:
        split_path = split_dir / f"{split_method}.csv"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Run `python -m src.splits` first."
            )
        split_frame = pd.read_csv(split_path)
        result = run_baseline_for_split(
            archive,
            split_frame,
            split_method=split_method,
            candidates=candidates,
            seed=seed,
            n_jobs=n_jobs,
        )

        summaries.append(result.summary_row())
        tuning_frames.append(result.tuning_history)
        prediction_path = predictions_dir / f"baseline_{split_method}.csv"
        result.predictions.to_csv(prediction_path, index=False)

        print(
            f"{split_method:8s} | "
            f"test RMSE={result.test_metrics.rmse:.4f} "
            f"MAE={result.test_metrics.mae:.4f} "
            f"R2={result.test_metrics.r2:.4f} | "
            f"best={json.dumps(result.best_params.__dict__, sort_keys=True)}"
        )

    metrics = pd.DataFrame(summaries)
    metrics.to_csv(tables_dir / "baseline_metrics.csv", index=False)
    pd.concat(tuning_frames, ignore_index=True).to_csv(
        tables_dir / "baseline_tuning.csv", index=False
    )

    print(f"Wrote metrics to {tables_dir / 'baseline_metrics.csv'}")
    print(f"Wrote tuning history to {tables_dir / 'baseline_tuning.csv'}")
    print(f"Wrote predictions under {predictions_dir}")


if __name__ == "__main__":
    main()
