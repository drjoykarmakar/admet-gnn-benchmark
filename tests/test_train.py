"""Tests for GNN training, early stopping, and artifact alignment."""

import numpy as np
import pandas as pd
import pytest

from src.train import GNNTrainConfig, run_gnn_for_split, validate_split_against_cleaned


def _tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    smiles = [
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CCO",
        "CCCO",
        "CCN",
        "CCCN",
        "CCCl",
        "CCBr",
        "c1ccccc1",
        "c1ccncc1",
    ]
    targets = np.linspace(-3.0, 1.0, len(smiles))
    cleaned = pd.DataFrame({"smiles": smiles, "target": targets})
    split = cleaned.copy()
    split["scaffold"] = "<ACYCLIC>"
    split.loc[10:, "scaffold"] = ["c1ccccc1", "c1ccncc1"]
    split["split"] = ["train"] * 8 + ["val"] * 2 + ["test"] * 2
    return cleaned, split


def test_validate_split_detects_stale_target() -> None:
    cleaned, split = _tiny_frames()
    split.loc[0, "target"] += 0.5
    with pytest.raises(ValueError, match="Targets in split table do not match"):
        validate_split_against_cleaned(cleaned, split)


def test_validate_split_requires_all_partitions() -> None:
    cleaned, split = _tiny_frames()
    split.loc[split["split"] == "test", "split"] = "train"
    with pytest.raises(ValueError, match="missing partitions"):
        validate_split_against_cleaned(cleaned, split)


def test_tiny_gnn_training_run_produces_metrics_and_predictions() -> None:
    cleaned, split = _tiny_frames()
    config = GNNTrainConfig(
        hidden_size=16,
        num_layers=2,
        batch_size=4,
        epochs=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        dropout=0.0,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
        num_workers=0,
        device="cpu",
    )
    result = run_gnn_for_split(
        cleaned,
        split,
        split_method="random",
        config=config,
        seed=7,
        graph_cache={},
    )

    assert 1 <= result.best_epoch <= result.epochs_trained <= 4
    assert result.n_train == 8
    assert result.n_val == 2
    assert result.n_test == 2
    assert np.isfinite(result.validation_metrics.rmse)
    assert np.isfinite(result.test_metrics.rmse)
    assert set(result.predictions["partition"]) == {"val", "test"}
    assert len(result.predictions) == 4
    assert len(result.history) == result.epochs_trained
