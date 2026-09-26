"""Tests for baseline selection and artifact alignment."""

import numpy as np
import pandas as pd
import pytest

from src.models_baseline import (
    BaselineParams,
    FeatureArchive,
    build_candidate_grid,
    run_baseline_for_split,
    validate_and_index_split,
)


def _archive() -> FeatureArchive:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 8)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.1 * rng.normal(size=30)).astype(
        np.float32
    )
    smiles = np.asarray([f"mol_{i}" for i in range(30)], dtype=str)
    names = np.asarray([f"f_{i}" for i in range(8)], dtype=str)
    return FeatureArchive(X=X, y=y, smiles=smiles, feature_names=names)


def _split_frame(archive: FeatureArchive) -> pd.DataFrame:
    split = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    return pd.DataFrame(
        {
            "smiles": archive.smiles,
            "target": archive.y,
            "split": split,
            "scaffold": [f"s_{i // 2}" for i in range(30)],
        }
    )


def test_validate_split_detects_target_mismatch() -> None:
    archive = _archive()
    split = _split_frame(archive)
    split.loc[0, "target"] += 1.0
    with pytest.raises(ValueError, match="Targets in split table do not match"):
        validate_and_index_split(archive, split)


def test_candidate_grid_expands_light_search() -> None:
    candidates = build_candidate_grid(
        {
            "n_estimators": 25,
            "max_features_candidates": ["sqrt", 0.5],
            "min_samples_leaf_candidates": [1, 2],
        }
    )
    assert len(candidates) == 4
    assert {candidate.n_estimators for candidate in candidates} == {25}


def test_baseline_run_returns_validation_and_test_predictions() -> None:
    archive = _archive()
    split = _split_frame(archive)
    candidates = [
        BaselineParams(n_estimators=20, max_features="sqrt", min_samples_leaf=1),
        BaselineParams(n_estimators=20, max_features=0.5, min_samples_leaf=2),
    ]
    result = run_baseline_for_split(
        archive,
        split,
        split_method="random",
        candidates=candidates,
        seed=7,
        n_jobs=1,
    )
    assert result.n_train == 20
    assert result.n_val == 5
    assert result.n_test == 5
    assert len(result.tuning_history) == 2
    assert set(result.predictions["partition"]) == {"val", "test"}
    assert len(result.predictions) == 10
    assert np.isfinite(result.test_metrics.rmse)
