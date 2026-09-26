#!/usr/bin/env python
"""Train the small PyTorch GNN on saved random and scaffold splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import GraphFeatures  # noqa: E402
from src.train import GNNTrainConfig, run_gnn_for_split  # noqa: E402


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI options for GNN training."""
    parser = argparse.ArgumentParser(description="Run the small molecular GNN benchmark.")
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the configured maximum epoch count.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="CPU-friendly smoke run using at most the configured demo epoch count.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Override the configured device.",
    )
    return parser


def main() -> None:
    """Train, early-stop, evaluate, and save GNN outputs for each split."""
    args = build_arg_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text())

    seed = int(config.get("seed", 42))
    data_config = config.get("data", {})
    gnn_config = config.get("gnn", {})
    results_config = config.get("results", {})

    epochs_override = args.epochs
    if args.demo:
        demo_epochs = int(gnn_config.get("demo_epochs", 5))
        epochs_override = min(epochs_override or demo_epochs, demo_epochs)
        if args.device is None:
            args.device = "cpu"

    train_config = GNNTrainConfig.from_mapping(
        gnn_config,
        epochs_override=epochs_override,
        device_override=args.device,
    )

    cleaned_path = _resolve(
        ROOT,
        data_config.get("cleaned_csv", "data/processed/solubility_aqsoldb_clean.csv"),
    )
    split_dir = _resolve(ROOT, data_config.get("split_dir", "data/processed/splits"))
    tables_dir = _resolve(ROOT, results_config.get("tables_dir", "results/tables"))
    predictions_dir = _resolve(
        ROOT, results_config.get("predictions_dir", "results/predictions")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}. Run `python -m src.data` first."
        )
    cleaned_frame = pd.read_csv(cleaned_path)

    summaries: list[dict[str, object]] = []
    history_frames: list[pd.DataFrame] = []
    graph_cache: dict[str, GraphFeatures] = {}

    for split_method in args.splits:
        split_path = split_dir / f"{split_method}.csv"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_path}. Run `python -m src.splits` first."
            )
        split_frame = pd.read_csv(split_path)
        print(
            f"Training {split_method} GNN on {train_config.device} "
            f"(max_epochs={train_config.epochs})..."
        )
        result = run_gnn_for_split(
            cleaned_frame,
            split_frame,
            split_method=split_method,
            config=train_config,
            seed=seed,
            graph_cache=graph_cache,
        )

        summaries.append(result.summary_row())
        history_frames.append(result.history)
        prediction_path = predictions_dir / f"gnn_{split_method}.csv"
        result.predictions.to_csv(prediction_path, index=False)

        print(
            f"{split_method:8s} | "
            f"test RMSE={result.test_metrics.rmse:.4f} "
            f"MAE={result.test_metrics.mae:.4f} "
            f"R2={result.test_metrics.r2:.4f} | "
            f"best_epoch={result.best_epoch}/{result.epochs_trained} "
            f"parameters={result.n_parameters:,}"
        )

    pd.DataFrame(summaries).to_csv(tables_dir / "gnn_metrics.csv", index=False)
    pd.concat(history_frames, ignore_index=True).to_csv(
        tables_dir / "gnn_history.csv", index=False
    )

    print(f"Wrote metrics to {tables_dir / 'gnn_metrics.csv'}")
    print(f"Wrote training history to {tables_dir / 'gnn_history.csv'}")
    print(f"Wrote predictions under {predictions_dir}")


if __name__ == "__main__":
    main()
