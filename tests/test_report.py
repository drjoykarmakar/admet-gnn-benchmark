"""Tests for applicability-domain analysis and report generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.make_report import generate_report
from src.evaluate import (
    attach_applicability_domain,
    benchmark_results,
    nearest_train_tanimoto,
    select_failure_cases,
    wide_benchmark_table,
)


def _split_frame(split_method: str) -> pd.DataFrame:
    smiles = [
        "CC",
        "CCC",
        "CCO",
        "CCN",
        "c1ccccc1",
        "c1ccncc1",
        "CC(=O)O",
        "CCCO",
    ]
    targets = np.linspace(-2.5, 0.5, len(smiles))
    frame = pd.DataFrame(
        {
            "smiles": smiles,
            "target": targets,
            "split": ["train"] * 4 + ["val"] * 2 + ["test"] * 2,
            "scaffold": ["<ACYCLIC>"] * 4
            + ["c1ccccc1", "c1ccncc1"]
            + ["<ACYCLIC>", "<ACYCLIC>"],
        }
    )
    if split_method == "scaffold":
        frame["scaffold"] = [f"s{i}" for i in range(len(frame))]
    return frame


def _prediction_frame(split: pd.DataFrame, split_method: str, model: str) -> pd.DataFrame:
    rows = split.loc[split["split"].isin(["val", "test"])].copy()
    model_offset = 0.12 if model == "Random Forest" else -0.18
    rows["y_true"] = rows["target"]
    rows["y_pred"] = rows["target"] + model_offset
    rows["residual"] = rows["y_pred"] - rows["y_true"]
    rows["abs_error"] = rows["residual"].abs()
    rows["partition"] = rows["split"]
    rows["split_method"] = split_method
    rows["model"] = model
    return rows[
        [
            "smiles",
            "scaffold",
            "y_true",
            "y_pred",
            "residual",
            "abs_error",
            "partition",
            "split_method",
            "model",
        ]
    ]


def test_nearest_train_tanimoto_attaches_to_predictions() -> None:
    split = _split_frame("random")
    similarities = nearest_train_tanimoto(split, radius=2, n_bits=128)
    assert set(similarities.columns) == {
        "smiles",
        "nearest_train_tanimoto",
        "nearest_train_smiles",
    }
    assert len(similarities) == 2
    assert similarities["nearest_train_tanimoto"].between(0.0, 1.0).all()

    predictions = _prediction_frame(split, "random", "Random Forest")
    enriched = attach_applicability_domain(predictions, similarities)
    test_rows = enriched.loc[enriched["partition"] == "test"]
    assert test_rows["nearest_train_tanimoto"].notna().all()
    assert test_rows["nearest_train_smiles"].str.len().gt(0).all()


def test_benchmark_and_failure_tables_are_derived_from_saved_predictions() -> None:
    frames = []
    for split_method in ("random", "scaffold"):
        split = _split_frame(split_method)
        similarity = nearest_train_tanimoto(split, radius=2, n_bits=128)
        for model in ("Random Forest", "Small GNN"):
            pred = _prediction_frame(split, split_method, model)
            frames.append(attach_applicability_domain(pred, similarity))
    predictions = pd.concat(frames, ignore_index=True)

    long = benchmark_results(predictions)
    wide = wide_benchmark_table(long)
    failures = select_failure_cases(predictions, top_n_per_model_split=1)

    assert len(long) == 4
    assert len(wide) == 2
    assert "scaffold_minus_random_rmse" in wide.columns
    assert len(failures) == 4
    assert failures["chemistry_context"].str.contains("do not establish").all()


def test_generate_report_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/processed/splits").mkdir(parents=True)
    (tmp_path / "results/predictions").mkdir(parents=True)
    (tmp_path / "results/tables").mkdir(parents=True)
    (tmp_path / "results/figures").mkdir(parents=True)

    config = {
        "data": {"split_dir": "data/processed/splits"},
        "features": {"morgan_radius": 2, "fingerprint_bits": 128},
        "results": {
            "tables_dir": "results/tables",
            "figures_dir": "results/figures",
            "predictions_dir": "results/predictions",
            "report_path": "results/report.md",
        },
        "report": {"failure_cases_per_model_split": 2, "failure_grid_size": 4},
    }
    (tmp_path / "configs/default.yaml").write_text(yaml.safe_dump(config))

    filenames = {
        ("Random Forest", "random"): "baseline_random.csv",
        ("Random Forest", "scaffold"): "baseline_scaffold.csv",
        ("Small GNN", "random"): "gnn_random.csv",
        ("Small GNN", "scaffold"): "gnn_scaffold.csv",
    }
    for split_method in ("random", "scaffold"):
        split = _split_frame(split_method)
        split.to_csv(tmp_path / f"data/processed/splits/{split_method}.csv", index=False)
        for model in ("Random Forest", "Small GNN"):
            pred = _prediction_frame(split, split_method, model)
            pred.to_csv(
                tmp_path / "results/predictions" / filenames[(model, split_method)],
                index=False,
            )

    history = pd.DataFrame(
        {
            "split_method": ["random", "random", "scaffold", "scaffold"],
            "epoch": [1, 2, 1, 2],
            "train_scaled_mse": [1.0, 0.8, 1.2, 0.9],
            "val_scaled_mse": [1.1, 0.9, 1.3, 1.0],
        }
    )
    history.to_csv(tmp_path / "results/tables/gnn_history.csv", index=False)

    outputs = generate_report("configs/default.yaml", root=tmp_path)
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    summary = pd.read_csv(tmp_path / "results/tables/benchmark_summary.csv")
    assert set(summary["model"]) == {"Random Forest", "Small GNN"}
    assert (tmp_path / "results/figures/residuals_scaffold.png").exists()
    assert (tmp_path / "results/figures/gnn_learning_curve_random.png").exists()
