"""Random Forest baseline training and validation-set model selection.

The baseline is intentionally strong but small enough to inspect. Hyperparameter
selection uses the validation partition only; the held-out test partition is
predicted exactly once after selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import RegressionMetrics, build_prediction_frame, regression_metrics


MODEL_NAME = "Random Forest"
REPRESENTATION_NAME = "Morgan + RDKit descriptors"
EXPECTED_PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class BaselineParams:
    """Random Forest hyperparameters exposed by the light search."""

    n_estimators: int = 300
    max_depth: int | None = None
    max_features: str | float = "sqrt"
    min_samples_leaf: int = 1


@dataclass(frozen=True)
class FeatureArchive:
    """Classical feature matrix plus identifiers saved by ``src.features``."""

    X: np.ndarray
    y: np.ndarray
    smiles: np.ndarray
    feature_names: np.ndarray


@dataclass
class BaselineRunResult:
    """Outputs from one random/scaffold baseline experiment."""

    split_method: str
    best_params: BaselineParams
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    n_train: int
    n_val: int
    n_test: int
    tuning_history: pd.DataFrame
    predictions: pd.DataFrame

    def summary_row(self) -> dict[str, Any]:
        """Return one flat row suitable for the benchmark result table."""
        row: dict[str, Any] = {
            "model": MODEL_NAME,
            "representation": REPRESENTATION_NAME,
            "split_method": self.split_method,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "val_rmse": self.validation_metrics.rmse,
            "val_mae": self.validation_metrics.mae,
            "val_r2": self.validation_metrics.r2,
            "test_rmse": self.test_metrics.rmse,
            "test_mae": self.test_metrics.mae,
            "test_r2": self.test_metrics.r2,
        }
        row.update({f"best_{key}": value for key, value in asdict(self.best_params).items()})
        return row


def load_feature_archive(path: Path | str) -> FeatureArchive:
    """Load and validate a compressed classical-feature archive."""
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Feature archive not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as data:
        required = {"X", "y", "smiles", "feature_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Feature archive is missing arrays: {sorted(missing)}")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32).reshape(-1)
        smiles = np.asarray(data["smiles"], dtype=str).reshape(-1)
        feature_names = np.asarray(data["feature_names"], dtype=str).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X.shape}.")
    if len(X) != len(y) or len(X) != len(smiles):
        raise ValueError("X, y, and smiles must contain the same number of rows.")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match X columns.")
    if len(set(smiles.tolist())) != len(smiles):
        raise ValueError("Feature archive contains duplicate standardized SMILES.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Feature archive contains non-finite numeric values.")

    return FeatureArchive(X=X, y=y, smiles=smiles, feature_names=feature_names)


def validate_and_index_split(
    archive: FeatureArchive,
    split_frame: pd.DataFrame,
    *,
    target_col: str = "target",
) -> pd.DataFrame:
    """Align a split CSV to the feature archive and detect stale artifacts.

    Alignment is by canonical SMILES rather than row position. If the split CSV
    carries targets, they must match the feature archive values within a tight
    floating-point tolerance.
    """
    required = {"smiles", "split"}
    missing = required.difference(split_frame.columns)
    if missing:
        raise ValueError(f"Split table is missing required columns: {sorted(missing)}")

    split = split_frame.copy()
    split["smiles"] = split["smiles"].astype(str)
    if split["smiles"].duplicated().any():
        raise ValueError("Split table contains duplicate standardized SMILES.")

    observed_partitions = set(split["split"].astype(str))
    unknown = observed_partitions.difference(EXPECTED_PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown split labels: {sorted(unknown)}")
    missing_partitions = set(EXPECTED_PARTITIONS).difference(observed_partitions)
    if missing_partitions:
        raise ValueError(f"Split table is missing partitions: {sorted(missing_partitions)}")

    feature_index = {smiles: index for index, smiles in enumerate(archive.smiles.tolist())}
    split_smiles = set(split["smiles"])
    feature_smiles = set(feature_index)
    if split_smiles != feature_smiles:
        missing_from_split = feature_smiles - split_smiles
        missing_from_features = split_smiles - feature_smiles
        raise ValueError(
            "Feature/split molecule sets differ: "
            f"{len(missing_from_split)} only in features, "
            f"{len(missing_from_features)} only in split table."
        )

    split["feature_index"] = split["smiles"].map(feature_index).astype(int)

    if target_col in split.columns:
        split_targets = pd.to_numeric(split[target_col], errors="raise").to_numpy(dtype=float)
        feature_targets = archive.y[split["feature_index"].to_numpy()]
        if not np.allclose(split_targets, feature_targets, rtol=1e-6, atol=1e-6):
            max_difference = float(np.max(np.abs(split_targets - feature_targets)))
            raise ValueError(
                "Targets in split table do not match the feature archive; "
                f"maximum absolute difference is {max_difference:.6g}."
            )

    return split


def _partition_arrays(
    archive: FeatureArchive,
    indexed_split: pd.DataFrame,
    partition: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract features, targets, and metadata for one named partition."""
    rows = indexed_split.loc[indexed_split["split"] == partition].copy()
    if rows.empty:
        raise ValueError(f"Partition {partition!r} is empty.")
    indices = rows["feature_index"].to_numpy(dtype=int)
    return archive.X[indices], archive.y[indices], rows


def build_candidate_grid(config: Mapping[str, Any]) -> list[BaselineParams]:
    """Expand the small validation-set search defined in YAML."""
    n_estimators = int(config.get("n_estimators", 300))
    max_depth_raw = config.get("max_depth", None)
    max_depth = None if max_depth_raw is None else int(max_depth_raw)
    max_features_values: Sequence[str | float] = config.get(
        "max_features_candidates", ["sqrt", 0.25]
    )
    min_leaf_values: Sequence[int] = config.get("min_samples_leaf_candidates", [1, 2])

    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if max_depth is not None and max_depth <= 0:
        raise ValueError("max_depth must be positive or null.")

    candidates: list[BaselineParams] = []
    for max_features, min_samples_leaf in product(max_features_values, min_leaf_values):
        if isinstance(max_features, str):
            if max_features not in {"sqrt", "log2"}:
                raise ValueError(
                    "max_features string values must be 'sqrt' or 'log2'; "
                    f"got {max_features!r}."
                )
        else:
            max_features = float(max_features)
            if not 0.0 < max_features <= 1.0:
                raise ValueError("Float max_features values must be in (0, 1].")

        min_samples_leaf = int(min_samples_leaf)
        if min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive.")

        candidates.append(
            BaselineParams(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
            )
        )

    if not candidates:
        raise ValueError("Baseline candidate grid is empty.")
    return candidates


def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    params: BaselineParams,
    seed: int,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Fit one deterministic Random Forest configuration."""
    model = RandomForestRegressor(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        max_features=params.max_features,
        min_samples_leaf=params.min_samples_leaf,
        criterion="squared_error",
        bootstrap=True,
        random_state=seed,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def run_baseline_for_split(
    archive: FeatureArchive,
    split_frame: pd.DataFrame,
    *,
    split_method: str,
    candidates: Iterable[BaselineParams],
    seed: int,
    n_jobs: int = -1,
) -> BaselineRunResult:
    """Tune on validation data, then evaluate one held-out test partition."""
    indexed = validate_and_index_split(archive, split_frame)
    X_train, y_train, train_rows = _partition_arrays(archive, indexed, "train")
    X_val, y_val, val_rows = _partition_arrays(archive, indexed, "val")
    X_test, y_test, test_rows = _partition_arrays(archive, indexed, "test")

    best_model: RandomForestRegressor | None = None
    best_params: BaselineParams | None = None
    best_metrics: RegressionMetrics | None = None
    tuning_rows: list[dict[str, Any]] = []

    for candidate_id, params in enumerate(candidates, start=1):
        model = fit_random_forest(
            X_train,
            y_train,
            params=params,
            seed=seed,
            n_jobs=n_jobs,
        )
        val_pred = model.predict(X_val)
        metrics = regression_metrics(y_val, val_pred)
        tuning_rows.append(
            {
                "split_method": split_method,
                "candidate_id": candidate_id,
                **asdict(params),
                "val_rmse": metrics.rmse,
                "val_mae": metrics.mae,
                "val_r2": metrics.r2,
            }
        )

        ranking = (metrics.rmse, metrics.mae)
        best_ranking = (
            (best_metrics.rmse, best_metrics.mae)
            if best_metrics is not None
            else (float("inf"), float("inf"))
        )
        if ranking < best_ranking:
            best_model = model
            best_params = params
            best_metrics = metrics

    if best_model is None or best_params is None or best_metrics is None:
        raise ValueError("At least one baseline candidate is required.")

    # Validation and test predictions are saved from the selected model. We do
    # not refit on train+validation here, which keeps training-set exposure
    # comparable with the later GNN that uses validation only for early stopping.
    selected_val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    selected_val_metrics = regression_metrics(y_val, selected_val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    val_predictions = build_prediction_frame(
        smiles=val_rows["smiles"],
        y_true=y_val,
        y_pred=selected_val_pred,
        partition="val",
        split_method=split_method,
        model_name=MODEL_NAME,
        scaffold=val_rows["scaffold"] if "scaffold" in val_rows.columns else None,
    )
    test_predictions = build_prediction_frame(
        smiles=test_rows["smiles"],
        y_true=y_test,
        y_pred=test_pred,
        partition="test",
        split_method=split_method,
        model_name=MODEL_NAME,
        scaffold=test_rows["scaffold"] if "scaffold" in test_rows.columns else None,
    )

    return BaselineRunResult(
        split_method=split_method,
        best_params=best_params,
        validation_metrics=selected_val_metrics,
        test_metrics=test_metrics,
        n_train=len(train_rows),
        n_val=len(val_rows),
        n_test=len(test_rows),
        tuning_history=pd.DataFrame(tuning_rows),
        predictions=pd.concat([val_predictions, test_predictions], ignore_index=True),
    )
