"""Plotting helpers for saved benchmark predictions and diagnostics.

Plots consume saved CSV artifacts rather than trained model objects. This keeps
report regeneration fast and prevents accidental retraining while polishing the
README or figures.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


REQUIRED_PREDICTION_COLUMNS = {
    "smiles",
    "y_true",
    "y_pred",
    "residual",
    "abs_error",
    "partition",
    "split_method",
    "model",
}


def _validated_test_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a finite test-only copy suitable for plotting."""
    missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    test = frame.loc[frame["partition"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError("Prediction table does not contain test-set rows.")

    for column in ("y_true", "y_pred", "residual", "abs_error"):
        values = pd.to_numeric(test[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction column {column!r} contains non-finite values.")
        test[column] = values
    return test


def _save_figure(fig: plt.Figure, output_path: Path | str) -> None:
    """Save a figure with consistent directory creation and resolution."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_parity_plot(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
) -> None:
    """Save observed-versus-predicted test values for one split."""
    test = _validated_test_predictions(predictions)
    fig, ax = plt.subplots(figsize=(6.2, 6.0))

    markers = ("o", "s", "^", "D")
    for marker, (model, rows) in zip(markers, test.groupby("model", sort=True)):
        ax.scatter(
            rows["y_true"],
            rows["y_pred"],
            alpha=0.58,
            s=28,
            marker=marker,
            label=str(model),
        )

    lower = float(min(test["y_true"].min(), test["y_pred"].min()))
    upper = float(max(test["y_true"].max(), test["y_pred"].max()))
    span = upper - lower
    padding = 0.04 * span if span > 0 else 0.5
    bounds = (lower - padding, upper + padding)
    ax.plot(bounds, bounds, linestyle="--", linewidth=1.2, label="Ideal")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_xlabel("Observed LogS")
    ax.set_ylabel("Predicted LogS")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def save_residual_plot(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
) -> None:
    """Save signed residuals versus observed LogS for one split."""
    test = _validated_test_predictions(predictions)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    markers = ("o", "s", "^", "D")
    for marker, (model, rows) in zip(markers, test.groupby("model", sort=True)):
        ax.scatter(
            rows["y_true"],
            rows["residual"],
            alpha=0.58,
            s=28,
            marker=marker,
            label=str(model),
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Observed LogS")
    ax.set_ylabel("Residual (prediction - observation)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def save_applicability_plot(
    predictions: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str,
) -> None:
    """Save absolute error versus nearest-training-set Tanimoto similarity."""
    test = _validated_test_predictions(predictions)
    if "nearest_train_tanimoto" not in test.columns:
        raise ValueError("Applicability plot requires nearest_train_tanimoto.")

    similarity = pd.to_numeric(test["nearest_train_tanimoto"], errors="raise")
    if not np.isfinite(similarity.to_numpy(dtype=float)).all():
        raise ValueError("nearest_train_tanimoto contains non-finite values.")
    if ((similarity < 0.0) | (similarity > 1.0)).any():
        raise ValueError("nearest_train_tanimoto values must lie in [0, 1].")
    test["nearest_train_tanimoto"] = similarity

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    markers = ("o", "s", "^", "D")
    for marker, (model, rows) in zip(markers, test.groupby("model", sort=True)):
        ax.scatter(
            rows["nearest_train_tanimoto"],
            rows["abs_error"],
            alpha=0.58,
            s=28,
            marker=marker,
            label=str(model),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Maximum Morgan Tanimoto to training set")
    ax.set_ylabel("Absolute error (LogS)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def save_gnn_learning_curve(
    history: pd.DataFrame,
    output_path: Path | str,
    *,
    split_method: str,
) -> None:
    """Save GNN train/validation loss history for one split when available."""
    required = {"split_method", "epoch", "train_scaled_mse", "val_scaled_mse"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"GNN history is missing columns: {sorted(missing)}")

    rows = history.loc[history["split_method"].astype(str) == split_method].copy()
    if rows.empty:
        raise ValueError(f"No GNN history rows found for split {split_method!r}.")

    rows = rows.sort_values("epoch")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(rows["epoch"], rows["train_scaled_mse"], label="Train MSE")
    ax.plot(rows["epoch"], rows["val_scaled_mse"], label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled-target MSE")
    ax.set_title(f"GNN training history - {split_method} split")
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def save_failure_molecule_grid(
    failure_cases: pd.DataFrame,
    output_path: Path | str,
    *,
    split_method: str = "scaffold",
    max_molecules: int = 6,
) -> None:
    """Draw a small grid of high-error molecules with factual legends."""
    required = {"smiles", "model", "abs_error", "y_true", "y_pred", "split_method"}
    missing = required.difference(failure_cases.columns)
    if missing:
        raise ValueError(f"Failure-case table is missing columns: {sorted(missing)}")
    if max_molecules <= 0:
        raise ValueError("max_molecules must be positive.")

    rows = failure_cases.loc[
        failure_cases["split_method"].astype(str) == split_method
    ].copy()
    rows = rows.sort_values("abs_error", ascending=False).head(max_molecules)
    if rows.empty:
        raise ValueError(f"No failure cases found for split {split_method!r}.")

    mols: list[Chem.Mol] = []
    legends: list[str] = []
    for row in rows.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.smiles))
        if mol is None:
            continue
        mols.append(mol)
        legends.append(
            f"{row.model}\nobs={float(row.y_true):.2f}  pred={float(row.y_pred):.2f}"
            f"\n|err|={float(row.abs_error):.2f}"
        )

    if not mols:
        raise ValueError("No failure-case SMILES could be rendered by RDKit.")

    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=3,
        subImgSize=(320, 240),
        legends=legends,
        useSVG=False,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output))
