"""Tests for local dataset loading used by offline/reproducible workflows."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import load_tdc_export


def test_load_tdc_export_accepts_tsv(tmp_path):
    path = tmp_path / "aqsol.tsv"
    pd.DataFrame({"Drug": ["CCO", "CC"], "Y": [-1.0, -2.0]}).to_csv(
        path, sep="\t", index=False
    )
    loaded = load_tdc_export(path)
    assert list(loaded.columns) == ["Drug", "Y"]
    assert len(loaded) == 2


def test_load_tdc_export_rejects_wrong_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"], "target": [-1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_tdc_export(path)
