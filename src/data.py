"""Download and clean the main molecular property dataset.

The cleaning policy is deliberately explicit. The benchmark should never rely
on hidden upstream assumptions about salts, invalid SMILES, or duplicate labels.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

LOGGER = logging.getLogger(__name__)

DATASET_NAME = "Solubility_AqSolDB"
DATASET_SOURCE_URL = "https://tdcommons.ai/single_pred_tasks/adme/"
DATASET_LICENSE = "CC BY 4.0"
SMILES_COLUMN_TDC = "Drug"
TARGET_COLUMN_TDC = "Y"

ConflictPolicy = Literal["drop", "mean", "median"]


@dataclass(frozen=True)
class CleaningReport:
    """Counts describing each data-cleaning decision."""

    dataset_name: str
    source_url: str
    dataset_license: str
    input_rows: int
    missing_or_invalid_target_rows: int
    missing_or_blank_smiles_rows: int
    rdkit_parse_or_sanitize_failures: int
    standardization_failures: int
    exact_input_smiles_duplicate_rows: int
    standardized_duplicate_rows_collapsed: int
    conflicting_standardized_label_groups: int
    conflicting_rows_dropped: int
    output_rows: int
    conflict_policy: str
    label_tolerance: float

    @property
    def removed_rows(self) -> int:
        """Return total rows not represented one-for-one in the final table."""
        return self.input_rows - self.output_rows


def load_tdc_export(path: Path | str) -> pd.DataFrame:
    """Load a local TDC-format CSV/TSV export with ``Drug`` and ``Y`` columns."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input dataset not found: {source}")

    # ``sep=None`` lets pandas infer comma versus tab for small text exports.
    frame = pd.read_csv(source, sep=None, engine="python")
    required = {SMILES_COLUMN_TDC, TARGET_COLUMN_TDC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Unexpected local dataset schema. Missing columns: {sorted(missing)}; "
            f"available columns: {list(frame.columns)}"
        )
    return frame.copy()


def fetch_tdc_aqsol(cache_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Download `Solubility_AqSolDB` through PyTDC and return a DataFrame.

    PyTDC is imported inside the function so that local cleaning utilities can
    still be imported and unit-tested without triggering the dataset loader.
    """
    try:
        from tdc.single_pred import ADME
    except ImportError as exc:
        raise ImportError(
            "PyTDC is required to download the dataset. Install requirements.txt."
        ) from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading TDC dataset %s into %s", DATASET_NAME, cache_path)
    dataset = ADME(name=DATASET_NAME, path=str(cache_path))
    frame = dataset.get_data(format="df")

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("PyTDC did not return a pandas DataFrame.")

    required = {SMILES_COLUMN_TDC, TARGET_COLUMN_TDC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Unexpected TDC schema. Missing columns: {sorted(missing)}; "
            f"available columns: {list(frame.columns)}"
        )

    return frame.copy()


def parse_smiles(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string with RDKit sanitization enabled."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
    except (ValueError, RuntimeError):
        return None

    return mol


def standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Return a cleaned, parent-fragment, neutralized RDKit molecule.

    The operations are intentionally simple and reproducible:
    1. RDKit Cleanup normalizes common functional-group representations.
    2. FragmentParent removes counterions/salts and chooses the parent fragment.
    3. Uncharger neutralizes common ionizable forms where RDKit can do so
       without inventing a tautomer or changing stereochemistry deliberately.
    4. The molecule is sanitized again before canonical SMILES generation.

    This representation is useful for a controlled benchmark, but it can erase
    salt-form/protonation information that may matter experimentally. That is a
    documented limitation, not a claim that the parent form is "the true" form.
    """
    cleaned = rdMolStandardize.Cleanup(Chem.Mol(mol))
    parent = rdMolStandardize.FragmentParent(cleaned)
    uncharged = rdMolStandardize.Uncharger().uncharge(parent)
    Chem.SanitizeMol(uncharged)
    return uncharged


def standardize_smiles(smiles: str) -> str | None:
    """Parse and standardize SMILES, returning canonical isomeric SMILES."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None

    try:
        standardized = standardize_mol(mol)
        canonical = Chem.MolToSmiles(standardized, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return None

    return canonical or None


def _labels_conflict(values: np.ndarray, tolerance: float) -> bool:
    """Return True when duplicate labels differ by more than tolerance."""
    if values.size <= 1:
        return False
    return float(np.max(values) - np.min(values)) > tolerance


def _resolve_label(values: np.ndarray, policy: ConflictPolicy) -> float:
    """Resolve labels for one standardized structure after conflict handling."""
    if policy == "mean":
        return float(np.mean(values))
    if policy == "median":
        return float(np.median(values))
    # For non-conflicting duplicates under "drop", median avoids dependence on
    # row order while remaining equal to the common label up to tolerance.
    return float(np.median(values))


def clean_molecular_dataframe(
    frame: pd.DataFrame,
    *,
    smiles_col: str = SMILES_COLUMN_TDC,
    target_col: str = TARGET_COLUMN_TDC,
    conflict_policy: ConflictPolicy = "drop",
    label_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean a molecular regression table and return data plus an audit report.

    Duplicate handling occurs after structure standardization because two input
    SMILES can encode the same parent molecule. With the default ``drop`` policy,
    a standardized structure is excluded when its associated labels disagree by
    more than ``label_tolerance``. ``mean`` and ``median`` keep such groups and
    aggregate their labels, which is useful for a sensitivity analysis but is
    less conservative than the default benchmark policy.
    """
    if conflict_policy not in {"drop", "mean", "median"}:
        raise ValueError(f"Unsupported conflict_policy: {conflict_policy}")
    if label_tolerance < 0:
        raise ValueError("label_tolerance must be non-negative.")

    missing_columns = {smiles_col, target_col}.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    work = frame[[smiles_col, target_col]].copy()
    input_rows = len(work)

    numeric_target = pd.to_numeric(work[target_col], errors="coerce")
    invalid_target_mask = ~np.isfinite(numeric_target.to_numpy(dtype=float, na_value=np.nan))
    missing_or_invalid_target_rows = int(invalid_target_mask.sum())
    work = work.loc[~invalid_target_mask].copy()
    work[target_col] = numeric_target.loc[~invalid_target_mask].astype(float)

    smiles_as_string = work[smiles_col].astype("string")
    blank_smiles_mask = smiles_as_string.isna() | smiles_as_string.str.strip().eq("")
    missing_or_blank_smiles_rows = int(blank_smiles_mask.sum())
    work = work.loc[~blank_smiles_mask].copy()
    work[smiles_col] = work[smiles_col].astype(str).str.strip()

    exact_input_smiles_duplicate_rows = int(work.duplicated(subset=[smiles_col]).sum())

    canonical_smiles: list[str | None] = []
    parse_failures = 0
    standardization_failures = 0

    for smiles in work[smiles_col]:
        mol = parse_smiles(smiles)
        if mol is None:
            canonical_smiles.append(None)
            parse_failures += 1
            continue

        try:
            standardized = standardize_mol(mol)
            canonical = Chem.MolToSmiles(
                standardized, canonical=True, isomericSmiles=True
            )
        except (ValueError, RuntimeError):
            canonical_smiles.append(None)
            standardization_failures += 1
            continue

        if not canonical:
            canonical_smiles.append(None)
            standardization_failures += 1
            continue

        canonical_smiles.append(canonical)

    work["canonical_smiles"] = canonical_smiles
    work = work.loc[work["canonical_smiles"].notna()].copy()

    output_rows: list[dict[str, object]] = []
    standardized_duplicate_rows_collapsed = 0
    conflicting_groups = 0
    conflicting_rows_dropped = 0

    for canonical, group in work.groupby("canonical_smiles", sort=True):
        labels = group[target_col].to_numpy(dtype=float)
        conflict = _labels_conflict(labels, label_tolerance)

        if conflict:
            conflicting_groups += 1
            if conflict_policy == "drop":
                conflicting_rows_dropped += len(group)
                continue

        standardized_duplicate_rows_collapsed += len(group) - 1
        output_rows.append(
            {
                "smiles": canonical,
                "target": _resolve_label(labels, conflict_policy),
                "n_source_rows": int(len(group)),
                "label_min": float(np.min(labels)),
                "label_max": float(np.max(labels)),
            }
        )

    cleaned = pd.DataFrame(
        output_rows,
        columns=["smiles", "target", "n_source_rows", "label_min", "label_max"],
    )
    cleaned = cleaned.sort_values("smiles", kind="stable").reset_index(drop=True)

    report = CleaningReport(
        dataset_name=DATASET_NAME,
        source_url=DATASET_SOURCE_URL,
        dataset_license=DATASET_LICENSE,
        input_rows=input_rows,
        missing_or_invalid_target_rows=missing_or_invalid_target_rows,
        missing_or_blank_smiles_rows=missing_or_blank_smiles_rows,
        rdkit_parse_or_sanitize_failures=parse_failures,
        standardization_failures=standardization_failures,
        exact_input_smiles_duplicate_rows=exact_input_smiles_duplicate_rows,
        standardized_duplicate_rows_collapsed=standardized_duplicate_rows_collapsed,
        conflicting_standardized_label_groups=conflicting_groups,
        conflicting_rows_dropped=conflicting_rows_dropped,
        output_rows=len(cleaned),
        conflict_policy=conflict_policy,
        label_tolerance=label_tolerance,
    )
    return cleaned, report


def save_cleaned_dataset(
    cleaned: pd.DataFrame,
    report: CleaningReport,
    *,
    output_path: Path | str,
    report_path: Path | str,
) -> None:
    """Write cleaned CSV and JSON cleaning audit."""
    output = Path(output_path)
    report_output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(output, index=False)

    payload = asdict(report)
    payload["removed_rows"] = report.removed_rows
    report_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    LOGGER.info("Wrote %d cleaned molecules to %s", len(cleaned), output)
    LOGGER.info("Wrote cleaning report to %s", report_output)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct command-line arguments for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Download and clean TDC Solubility_AqSolDB."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory used by PyTDC for downloaded data.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help=(
            "Optional local TDC-format CSV/TSV with Drug and Y columns. "
            "When supplied, PyTDC download is skipped."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/solubility_aqsoldb_clean.csv"),
        help="Path for the cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/solubility_aqsoldb_cleaning.json"),
        help="Path for the JSON cleaning audit.",
    )
    parser.add_argument(
        "--conflict-policy",
        choices=("drop", "mean", "median"),
        default="drop",
        help="How to handle one standardized structure with conflicting labels.",
    )
    parser.add_argument(
        "--label-tolerance",
        type=float,
        default=0.01,
        help="Maximum duplicate-label range treated as agreement (LogS units).",
    )
    return parser


def main() -> None:
    """Download, clean, and save the benchmark dataset."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args()

    raw = load_tdc_export(args.input_file) if args.input_file else fetch_tdc_aqsol(args.cache_dir)
    cleaned, report = clean_molecular_dataframe(
        raw,
        conflict_policy=args.conflict_policy,
        label_tolerance=args.label_tolerance,
    )
    save_cleaned_dataset(
        cleaned,
        report,
        output_path=args.output,
        report_path=args.report,
    )

    LOGGER.info(
        "Cleaning complete: %d input rows -> %d unique standardized molecules",
        report.input_rows,
        report.output_rows,
    )


if __name__ == "__main__":
    main()
