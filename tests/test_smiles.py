"""Focused tests for the Stage 1 SMILES cleaning policy."""

import pandas as pd

from src.data import clean_molecular_dataframe, standardize_smiles


def test_standardize_smiles_strips_simple_counterion() -> None:
    assert standardize_smiles("CCO.[Na+]") == "CCO"


def test_equivalent_smiles_collapse_after_standardization() -> None:
    frame = pd.DataFrame(
        {
            "Drug": ["CCO", "OCC"],
            "Y": [-0.30, -0.30],
        }
    )
    cleaned, report = clean_molecular_dataframe(frame)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "smiles"] == "CCO"
    assert report.standardized_duplicate_rows_collapsed == 1


def test_conflicting_duplicate_labels_are_dropped_by_default() -> None:
    frame = pd.DataFrame(
        {
            "Drug": ["c1ccccc1", "C1=CC=CC=C1"],
            "Y": [-2.0, -3.0],
        }
    )
    cleaned, report = clean_molecular_dataframe(frame, label_tolerance=0.01)
    assert cleaned.empty
    assert report.conflicting_standardized_label_groups == 1
    assert report.conflicting_rows_dropped == 2
