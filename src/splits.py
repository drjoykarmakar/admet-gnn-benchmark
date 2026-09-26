"""Deterministic random and Bemis-Murcko scaffold splits.

The scaffold split keeps every Bemis-Murcko scaffold group in exactly one
partition. Molecules without a Murcko core share the explicit ``<ACYCLIC>``
group by default; this is conservative and can make partition sizes imperfect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


SplitName = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class SplitSummary:
    """Compact audit record for one split assignment."""

    method: str
    seed: int
    requested_train_fraction: float
    requested_val_fraction: float
    requested_test_fraction: float
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    n_unique_scaffolds: int
    train_val_scaffold_overlap: int
    train_test_scaffold_overlap: int
    val_test_scaffold_overlap: int


def _validate_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    """Validate split fractions with a small floating-point tolerance."""
    values = (train_fraction, val_fraction, test_fraction)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Split fractions must lie in [0, 1].")
    if not np.isclose(sum(values), 1.0, atol=1e-8):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.")
    if train_fraction <= 0.0:
        raise ValueError("train_fraction must be positive.")


def bemis_murcko_scaffold(smiles: str) -> str:
    """Return a canonical Bemis-Murcko scaffold or ``<ACYCLIC>``.

    Input is expected to be the cleaned canonical SMILES produced by ``src.data``.
    Acyclic molecules have no Murcko scaffold, so they are intentionally grouped
    together rather than assigned artificial per-molecule pseudo-scaffolds.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse cleaned SMILES: {smiles!r}")

    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False,
    )
    return scaffold if scaffold else "<ACYCLIC>"


def add_scaffold_column(
    frame: pd.DataFrame,
    *,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Return a copy with a canonical ``scaffold`` audit column."""
    if smiles_col not in frame.columns:
        raise ValueError(f"Missing SMILES column: {smiles_col!r}")
    result = frame.copy()
    result["scaffold"] = [bemis_murcko_scaffold(value) for value in result[smiles_col]]
    return result


def random_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Assign rows to deterministic random train/validation/test partitions."""
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    result = add_scaffold_column(frame, smiles_col=smiles_col)

    n_rows = len(result)
    permutation = np.random.default_rng(seed).permutation(n_rows)
    targets = _target_counts(n_rows, train_fraction, val_fraction, test_fraction)
    n_train = targets["train"]
    n_val = targets["val"]

    split_values = np.full(n_rows, "test", dtype=object)
    split_values[permutation[:n_train]] = "train"
    split_values[permutation[n_train : n_train + n_val]] = "val"
    result["split"] = split_values
    return result


def _group_indices_by_scaffold(scaffolds: pd.Series) -> list[list[int]]:
    """Collect positional row indices for each scaffold."""
    groups: dict[str, list[int]] = {}
    for position, scaffold in enumerate(scaffolds.astype(str).tolist()):
        groups.setdefault(scaffold, []).append(position)
    return list(groups.values())


def _target_counts(
    n_rows: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> dict[SplitName, int]:
    """Return integer target counts that sum exactly to ``n_rows``.

    When the dataset is large enough, every requested non-zero partition gets at
    least one row. This matters mostly for unit tests and tiny demos; AqSolDB is
    large enough that the adjustment does not affect the benchmark proportions.
    """
    fractions = np.asarray(
        [train_fraction, val_fraction, test_fraction],
        dtype=float,
    )
    raw = fractions * n_rows
    counts = np.floor(raw).astype(int)
    remainder = n_rows - int(counts.sum())

    # Give leftover rows to partitions with the largest fractional remainder.
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1

    positive = np.flatnonzero(fractions > 0.0)
    if n_rows >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donor_candidates = [
                donor
                for donor in positive
                if counts[donor] > 1
            ]
            if not donor_candidates:
                break
            donor = max(
                donor_candidates,
                key=lambda candidate: (counts[candidate] - raw[candidate], counts[candidate]),
            )
            counts[donor] -= 1
            counts[index] += 1

    return {
        "train": int(counts[0]),
        "val": int(counts[1]),
        "test": int(counts[2]),
    }


def scaffold_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    """Assign entire Bemis-Murcko scaffold groups to one partition.

    Groups are processed from largest to smallest. Equal-size groups are shuffled
    reproducibly before sorting, preventing an accidental lexical-order bias.
    Each group is then placed into the partition with the largest remaining
    requested capacity. Exact fractions are not guaranteed because scaffolds are
    indivisible; the saved split audit makes any imbalance visible.
    """
    _validate_fractions(train_fraction, val_fraction, test_fraction)
    result = add_scaffold_column(frame, smiles_col=smiles_col)
    n_rows = len(result)
    targets = _target_counts(n_rows, train_fraction, val_fraction, test_fraction)

    groups = _group_indices_by_scaffold(result["scaffold"])
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)  # stable: shuffled order breaks size ties

    assigned_counts: dict[SplitName, int] = {name: 0 for name in SPLIT_NAMES}
    split_values = np.empty(n_rows, dtype=object)

    for group_index, group in enumerate(groups):
        remaining = {
            name: targets[name] - assigned_counts[name]
            for name in SPLIT_NAMES
        }
        required_empty = [
            name
            for name in SPLIT_NAMES
            if targets[name] > 0 and assigned_counts[name] == 0
        ]
        groups_after_current = len(groups) - group_index - 1

        # Preserve at least one scaffold group for each requested non-empty
        # partition when the number of unique scaffolds makes that possible.
        # Without this guard, a small validation/test fraction can remain empty
        # even though suitable groups are available.
        if required_empty and groups_after_current < len(required_empty):
            destination = max(required_empty, key=lambda name: remaining[name])
        else:
            # Largest remaining absolute capacity wins. The fixed SPLIT_NAMES
            # order makes ties deterministic: train, then validation, then test.
            destination = max(SPLIT_NAMES, key=lambda name: remaining[name])

        split_values[group] = destination
        assigned_counts[destination] += len(group)

    result["split"] = split_values
    assert_no_scaffold_overlap(result)
    return result


def scaffold_overlap_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count scaffold identities shared by each pair of partitions."""
    required = {"split", "scaffold"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    scaffold_sets = {
        split: set(frame.loc[frame["split"] == split, "scaffold"].astype(str))
        for split in SPLIT_NAMES
    }
    return {
        "train_val": len(scaffold_sets["train"] & scaffold_sets["val"]),
        "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
        "val_test": len(scaffold_sets["val"] & scaffold_sets["test"]),
    }


def assert_no_scaffold_overlap(frame: pd.DataFrame) -> None:
    """Raise AssertionError if any scaffold appears in multiple partitions."""
    overlap = scaffold_overlap_counts(frame)
    if any(overlap.values()):
        raise AssertionError(f"Scaffold leakage detected: {overlap}")


def summarize_split(
    frame: pd.DataFrame,
    *,
    method: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> SplitSummary:
    """Build a reproducibility audit for a completed split assignment."""
    overlap = scaffold_overlap_counts(frame)
    counts = frame["split"].value_counts()
    return SplitSummary(
        method=method,
        seed=seed,
        requested_train_fraction=train_fraction,
        requested_val_fraction=val_fraction,
        requested_test_fraction=test_fraction,
        n_total=len(frame),
        n_train=int(counts.get("train", 0)),
        n_val=int(counts.get("val", 0)),
        n_test=int(counts.get("test", 0)),
        n_unique_scaffolds=int(frame["scaffold"].nunique()),
        train_val_scaffold_overlap=overlap["train_val"],
        train_test_scaffold_overlap=overlap["train_test"],
        val_test_scaffold_overlap=overlap["val_test"],
    )


def save_split(
    frame: pd.DataFrame,
    *,
    output_csv: Path | str,
    output_json: Path | str,
    method: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> SplitSummary:
    """Save split assignments plus a small JSON audit."""
    summary = summarize_split(
        frame,
        method=method,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    csv_path = Path(output_csv)
    json_path = Path(output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI arguments for saving both benchmark splits."""
    parser = argparse.ArgumentParser(description="Create random and scaffold splits.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/solubility_aqsoldb_clean.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Create and save random and scaffold split assignments."""
    args = build_arg_parser().parse_args()
    frame = pd.read_csv(args.input)
    kwargs = {
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
    }

    random_frame = random_split(frame, **kwargs)
    random_summary = save_split(
        random_frame,
        output_csv=args.output_dir / "random.csv",
        output_json=args.output_dir / "random.json",
        method="random",
        **kwargs,
    )

    scaffold_frame = scaffold_split(frame, **kwargs)
    scaffold_summary = save_split(
        scaffold_frame,
        output_csv=args.output_dir / "scaffold.csv",
        output_json=args.output_dir / "scaffold.json",
        method="scaffold",
        **kwargs,
    )

    print(
        "Random split: "
        f"{random_summary.n_train}/{random_summary.n_val}/{random_summary.n_test} "
        "train/val/test"
    )
    print(
        "Scaffold split: "
        f"{scaffold_summary.n_train}/{scaffold_summary.n_val}/{scaffold_summary.n_test} "
        "train/val/test; pairwise scaffold overlap = 0"
    )


if __name__ == "__main__":
    main()
