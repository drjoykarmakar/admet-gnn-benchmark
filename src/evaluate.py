"""Evaluation and error-analysis helpers for regression benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    """Standard held-out regression metrics used throughout the benchmark."""

    n: int
    rmse: float
    mae: float
    r2: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON/CSV-friendly representation."""
        return asdict(self)


def _as_1d_finite(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    """Convert numeric input to a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def regression_metrics(
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
) -> RegressionMetrics:
    """Compute RMSE, MAE, and R2 for one prediction set."""
    true = _as_1d_finite(y_true, name="y_true")
    pred = _as_1d_finite(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have identical shape; got {true.shape} and {pred.shape}."
        )

    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    r2 = float(r2_score(true, pred))
    return RegressionMetrics(n=len(true), rmse=rmse, mae=mae, r2=r2)


def build_prediction_frame(
    *,
    smiles: Iterable[str],
    y_true: Iterable[float] | np.ndarray,
    y_pred: Iterable[float] | np.ndarray,
    partition: str,
    split_method: str,
    model_name: str,
    scaffold: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build an auditable prediction table with signed and absolute errors.

    ``residual`` is defined as prediction minus observation. Positive residuals
    therefore mean the model predicted a less negative / larger LogS value.
    """
    smiles_values = list(smiles)
    true = _as_1d_finite(y_true, name="y_true")
    pred = _as_1d_finite(y_pred, name="y_pred")
    if len(smiles_values) != len(true) or len(pred) != len(true):
        raise ValueError("SMILES, y_true, and y_pred must have the same length.")

    frame = pd.DataFrame(
        {
            "smiles": smiles_values,
            "y_true": true,
            "y_pred": pred,
            "residual": pred - true,
            "abs_error": np.abs(pred - true),
            "partition": partition,
            "split_method": split_method,
            "model": model_name,
        }
    )

    if scaffold is not None:
        scaffold_values = list(scaffold)
        if len(scaffold_values) != len(frame):
            raise ValueError("scaffold must have the same length as predictions.")
        frame.insert(1, "scaffold", scaffold_values)

    return frame


def benchmark_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recompute held-out metrics from saved prediction CSVs.

    Recomputing metrics for the report makes the prediction tables the source of
    truth and helps catch stale or hand-edited summary files.
    """
    required = {"model", "split_method", "partition", "y_true", "y_pred"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    test = predictions.loc[predictions["partition"].astype(str) == "test"].copy()
    if test.empty:
        raise ValueError("Prediction table contains no test-set rows.")

    rows: list[dict[str, object]] = []
    for (model, split_method), group in test.groupby(
        ["model", "split_method"], sort=True
    ):
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                "model": str(model),
                "split_method": str(split_method),
                **metrics.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def wide_benchmark_table(results: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-form test metrics into one recruiter-readable table."""
    required = {"model", "split_method", "rmse", "mae", "r2"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Benchmark results are missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for model, group in results.groupby("model", sort=True):
        by_split = group.set_index("split_method")
        row: dict[str, object] = {"model": str(model)}
        for split in ("random", "scaffold"):
            if split not in by_split.index:
                raise ValueError(f"Missing {split!r} result for model {model!r}.")
            values = by_split.loc[split]
            row[f"{split}_rmse"] = float(values["rmse"])
            row[f"{split}_mae"] = float(values["mae"])
            row[f"{split}_r2"] = float(values["r2"])
        row["scaffold_minus_random_rmse"] = (
            float(row["scaffold_rmse"]) - float(row["random_rmse"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def nearest_train_tanimoto(
    split_frame: pd.DataFrame,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """Find each test molecule's nearest training neighbor by Morgan Tanimoto.

    This is a structural-proximity diagnostic only. It is not a calibrated
    uncertainty estimate and no universal in-domain cutoff is imposed.
    """
    required = {"smiles", "split"}
    missing = required.difference(split_frame.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    split = split_frame.copy()
    split["smiles"] = split["smiles"].astype(str)
    train_smiles = split.loc[split["split"].astype(str) == "train", "smiles"].tolist()
    test_smiles = split.loc[split["split"].astype(str) == "test", "smiles"].tolist()
    if not train_smiles or not test_smiles:
        raise ValueError("Split table must contain non-empty train and test partitions.")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    def fp(smiles: str):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise ValueError(f"Could not parse split SMILES for Tanimoto analysis: {smiles!r}")
        return generator.GetFingerprint(mol)

    train_fps = [fp(smiles) for smiles in train_smiles]
    rows: list[dict[str, object]] = []
    for smiles in test_smiles:
        test_fp = fp(smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        best_index = int(np.argmax(similarities))
        rows.append(
            {
                "smiles": smiles,
                "nearest_train_tanimoto": float(similarities[best_index]),
                "nearest_train_smiles": train_smiles[best_index],
            }
        )
    return pd.DataFrame(rows)


def attach_applicability_domain(
    predictions: pd.DataFrame,
    similarity_table: pd.DataFrame,
) -> pd.DataFrame:
    """Attach nearest-neighbor similarity fields to test prediction rows."""
    required = {"smiles", "partition"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    sim_required = {"smiles", "nearest_train_tanimoto", "nearest_train_smiles"}
    sim_missing = sim_required.difference(similarity_table.columns)
    if sim_missing:
        raise ValueError(f"Similarity table is missing columns: {sorted(sim_missing)}")

    result = predictions.copy()
    test_mask = result["partition"].astype(str) == "test"
    test_smiles = set(result.loc[test_mask, "smiles"].astype(str))
    similarity_smiles = set(similarity_table["smiles"].astype(str))
    if test_smiles != similarity_smiles:
        raise ValueError(
            "Prediction test molecules and applicability-domain molecules differ."
        )

    merged_test = result.loc[test_mask].merge(
        similarity_table,
        on="smiles",
        how="left",
        validate="many_to_one",
    )
    non_test = result.loc[~test_mask].copy()
    non_test["nearest_train_tanimoto"] = np.nan
    non_test["nearest_train_smiles"] = ""
    return pd.concat([non_test, merged_test], ignore_index=True, sort=False)


def applicability_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize test error in broad nearest-neighbor similarity bins."""
    required = {
        "model",
        "split_method",
        "partition",
        "y_true",
        "y_pred",
        "nearest_train_tanimoto",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    test = predictions.loc[predictions["partition"].astype(str) == "test"].copy()
    bins = [-1e-12, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-12]
    labels = ["<0.30", "0.30-0.49", "0.50-0.69", "0.70-0.89", ">=0.90"]
    test["similarity_bin"] = pd.cut(
        pd.to_numeric(test["nearest_train_tanimoto"], errors="raise"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows: list[dict[str, object]] = []
    grouped = test.groupby(
        ["model", "split_method", "similarity_bin"],
        observed=True,
        sort=True,
    )
    for (model, split_method, similarity_bin), group in grouped:
        true = _as_1d_finite(group["y_true"], name="y_true")
        pred = _as_1d_finite(group["y_pred"], name="y_pred")
        rmse = float(np.sqrt(mean_squared_error(true, pred)))
        mae = float(mean_absolute_error(true, pred))
        r2 = float(r2_score(true, pred)) if len(true) >= 2 else np.nan
        rows.append(
            {
                "model": str(model),
                "split_method": str(split_method),
                "similarity_bin": str(similarity_bin),
                "n": len(true),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def chemistry_context(smiles: str) -> str:
    """Return a cautious, descriptor-based plain-language molecule description."""
    mol = Chem.MolFromSmiles(str(smiles), sanitize=True)
    if mol is None:
        raise ValueError(f"Could not parse SMILES for chemistry summary: {smiles!r}")

    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    charge_text = "neutral" if formal_charge == 0 else f"formal charge {formal_charge:+d}"
    return (
        f"{charge_text}; MW {mw:.1f}; cLogP {logp:.2f}; TPSA {tpsa:.1f}; "
        f"HBD/HBA {hbd}/{hba}; {rot} rotatable bonds; {aromatic_rings} aromatic rings. "
        "These 2D descriptors provide structural context but do not establish the cause of the error."
    )


def select_failure_cases(
    predictions: pd.DataFrame,
    *,
    top_n_per_model_split: int = 5,
) -> pd.DataFrame:
    """Select largest held-out absolute errors and add chemistry context."""
    if top_n_per_model_split <= 0:
        raise ValueError("top_n_per_model_split must be positive.")
    required = {
        "smiles",
        "model",
        "split_method",
        "partition",
        "y_true",
        "y_pred",
        "residual",
        "abs_error",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    test = predictions.loc[predictions["partition"].astype(str) == "test"].copy()
    test["abs_error"] = pd.to_numeric(test["abs_error"], errors="raise")
    selected = (
        test.sort_values("abs_error", ascending=False)
        .groupby(["model", "split_method"], sort=True, group_keys=False)
        .head(top_n_per_model_split)
        .copy()
    )
    selected["chemistry_context"] = selected["smiles"].map(chemistry_context)
    return selected.sort_values(
        ["split_method", "model", "abs_error"], ascending=[True, True, False]
    ).reset_index(drop=True)
