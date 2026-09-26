"""Tests for saved-prediction plotting helpers."""

from pathlib import Path

import pandas as pd

from src.plots import (
    save_applicability_plot,
    save_failure_molecule_grid,
    save_parity_plot,
    save_residual_plot,
)


def _predictions() -> pd.DataFrame:
    rows = []
    smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    true = [-0.7, -0.2, -2.5, 0.1]
    for model, offset in (("Random Forest", 0.15), ("Small GNN", -0.10)):
        for i, (smi, observed) in enumerate(zip(smiles, true)):
            predicted = observed + offset * (i + 1) / 4
            rows.append(
                {
                    "smiles": smi,
                    "y_true": observed,
                    "y_pred": predicted,
                    "residual": predicted - observed,
                    "abs_error": abs(predicted - observed),
                    "partition": "test",
                    "split_method": "scaffold",
                    "model": model,
                    "nearest_train_tanimoto": 0.35 + 0.15 * i,
                }
            )
    return pd.DataFrame(rows)


def test_prediction_plots_write_nonempty_pngs(tmp_path: Path) -> None:
    frame = _predictions()
    outputs = [
        tmp_path / "parity.png",
        tmp_path / "residuals.png",
        tmp_path / "applicability.png",
    ]
    save_parity_plot(frame, outputs[0], title="Parity")
    save_residual_plot(frame, outputs[1], title="Residuals")
    save_applicability_plot(frame, outputs[2], title="Applicability")

    for output in outputs:
        assert output.exists()
        assert output.stat().st_size > 1000


def test_failure_grid_writes_rdkit_image(tmp_path: Path) -> None:
    frame = _predictions()
    output = tmp_path / "failures.png"
    save_failure_molecule_grid(frame, output, split_method="scaffold", max_molecules=4)
    assert output.exists()
    assert output.stat().st_size > 1000
