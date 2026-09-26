"""Unit tests for deterministic random and scaffold split logic."""

import pandas as pd

from src.splits import (
    assert_no_scaffold_overlap,
    bemis_murcko_scaffold,
    random_split,
    scaffold_overlap_counts,
    scaffold_split,
)


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": [
                "c1ccccc1O",
                "c1ccccc1N",
                "c1ccccc1Cl",
                "C1CCCCC1O",
                "C1CCCCC1N",
                "c1ccncc1",
                "c1ccncc1O",
                "CCO",
                "CCCO",
                "CCCCO",
            ],
            "target": list(range(10)),
        }
    )


def test_bemis_murcko_groups_same_ring_core() -> None:
    assert bemis_murcko_scaffold("c1ccccc1O") == bemis_murcko_scaffold("c1ccccc1N")
    assert bemis_murcko_scaffold("CCO") == "<ACYCLIC>"


def test_random_split_is_reproducible() -> None:
    frame = _toy_frame()
    first = random_split(frame, seed=7)
    second = random_split(frame, seed=7)
    assert first["split"].tolist() == second["split"].tolist()
    assert first["split"].value_counts().to_dict() == {"train": 8, "val": 1, "test": 1}


def test_scaffold_split_keeps_scaffolds_together() -> None:
    split = scaffold_split(_toy_frame(), seed=7)
    assert_no_scaffold_overlap(split)
    overlap = scaffold_overlap_counts(split)
    assert overlap == {"train_val": 0, "train_test": 0, "val_test": 0}

    per_scaffold_split_count = split.groupby("scaffold")["split"].nunique()
    assert (per_scaffold_split_count == 1).all()
    assert set(split["split"]) == {"train", "val", "test"}


def test_scaffold_split_is_reproducible() -> None:
    frame = _toy_frame()
    first = scaffold_split(frame, seed=11)
    second = scaffold_split(frame, seed=11)
    assert first["split"].tolist() == second["split"].tolist()
