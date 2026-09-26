"""Molecular representations used by the baseline and graph models.

The classical representation is a Morgan fingerprint concatenated with a small,
fixed set of interpretable RDKit descriptors. Graph featurization is deliberately
compact: categorical atom/bond chemistry is one-hot encoded with an explicit
"other" bucket so uncommon elements do not crash the pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors


DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "TPSA",
    "NumHBD",
    "NumHBA",
    "MolLogP",
    "NumRotatableBonds",
)

# A compact element vocabulary covering common drug-like chemistry. Anything
# outside this list maps to the final "other" bucket rather than being dropped.
ATOM_NUMBERS: tuple[int, ...] = (1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53)
ATOM_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
TOTAL_H_COUNTS: tuple[int, ...] = (0, 1, 2, 3, 4)
HYBRIDIZATIONS: tuple[Chem.rdchem.HybridizationType, ...] = (
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
BOND_TYPES: tuple[Chem.rdchem.BondType, ...] = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)


@dataclass(frozen=True)
class GraphFeatures:
    """Numpy representation of one molecular graph."""

    smiles: str
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


NODE_FEATURE_DIM = (
    len(ATOM_NUMBERS)
    + 1
    + len(ATOM_DEGREES)
    + 1
    + len(FORMAL_CHARGES)
    + 1
    + len(HYBRIDIZATIONS)
    + 1
    + len(TOTAL_H_COUNTS)
    + 1
    + 2  # aromatic, in-ring
)
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 2  # other bond type, conjugated, in-ring


def mol_from_clean_smiles(smiles: str) -> Chem.Mol:
    """Parse a cleaned SMILES string or raise a useful error."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")

    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit could not parse cleaned SMILES: {smiles!r}")
    return mol


def _one_hot_with_other(value: object, choices: Sequence[object]) -> list[float]:
    """One-hot encode a value with the final position reserved for unknowns."""
    encoded = [0.0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = len(choices)
    encoded[index] = 1.0
    return encoded


def descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    """Calculate the fixed six-descriptor vector used by the RF baseline."""
    values = (
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        Descriptors.MolLogP(mol),
        Lipinski.NumRotatableBonds(mol),
    )
    return np.asarray(values, dtype=np.float32)


def morgan_fingerprint(
    mol: Chem.Mol,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Return a binary Morgan fingerprint as a float32 numpy array."""
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )
    bit_vector = generator.GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array.astype(np.float32, copy=False)


def classical_feature_vector(
    smiles: str,
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Concatenate Morgan bits and RDKit descriptors for one molecule."""
    mol = mol_from_clean_smiles(smiles)
    fingerprint = morgan_fingerprint(mol, radius=radius, n_bits=n_bits)
    descriptors = descriptor_vector(mol)
    return np.concatenate((fingerprint, descriptors)).astype(np.float32, copy=False)


def classical_feature_names(n_bits: int = 2048) -> list[str]:
    """Return stable column names matching ``classical_feature_vector``."""
    if n_bits <= 0:
        raise ValueError("n_bits must be positive.")
    fingerprint_names = [f"morgan_{index}" for index in range(n_bits)]
    return fingerprint_names + list(DESCRIPTOR_NAMES)


def build_classical_matrix(
    smiles: Iterable[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Featurize an iterable of SMILES into a dense baseline design matrix."""
    rows = [
        classical_feature_vector(value, radius=radius, n_bits=n_bits)
        for value in smiles
    ]
    if not rows:
        return np.empty((0, n_bits + len(DESCRIPTOR_NAMES)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def atom_feature_vector(atom: Chem.Atom) -> np.ndarray:
    """Encode one atom using small categorical vocabularies and two flags."""
    features: list[float] = []
    features.extend(_one_hot_with_other(atom.GetAtomicNum(), ATOM_NUMBERS))
    features.extend(_one_hot_with_other(atom.GetDegree(), ATOM_DEGREES))
    features.extend(_one_hot_with_other(atom.GetFormalCharge(), FORMAL_CHARGES))
    features.extend(_one_hot_with_other(atom.GetHybridization(), HYBRIDIZATIONS))
    features.extend(_one_hot_with_other(atom.GetTotalNumHs(), TOTAL_H_COUNTS))
    features.append(float(atom.GetIsAromatic()))
    features.append(float(atom.IsInRing()))
    return np.asarray(features, dtype=np.float32)


def bond_feature_vector(bond: Chem.Bond) -> np.ndarray:
    """Encode one bond; direction is handled separately in ``edge_index``."""
    features = _one_hot_with_other(bond.GetBondType(), BOND_TYPES)
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    return np.asarray(features, dtype=np.float32)


def smiles_to_graph(smiles: str) -> GraphFeatures:
    """Convert one cleaned SMILES string to a bidirectional molecular graph."""
    mol = mol_from_clean_smiles(smiles)
    node_features = np.vstack([atom_feature_vector(atom) for atom in mol.GetAtoms()])

    directed_edges: list[tuple[int, int]] = []
    directed_edge_features: list[np.ndarray] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        features = bond_feature_vector(bond)
        directed_edges.extend(((begin, end), (end, begin)))
        directed_edge_features.extend((features, features.copy()))

    if directed_edges:
        edge_index = np.asarray(directed_edges, dtype=np.int64).T
        edge_features = np.vstack(directed_edge_features).astype(np.float32, copy=False)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, EDGE_FEATURE_DIM), dtype=np.float32)

    return GraphFeatures(
        smiles=smiles,
        node_features=node_features.astype(np.float32, copy=False),
        edge_index=edge_index,
        edge_features=edge_features,
    )


def graph_to_pyg_data(graph: GraphFeatures, target: float | None = None):
    """Convert ``GraphFeatures`` to ``torch_geometric.data.Data`` lazily.

    PyTorch Geometric is optional. The benchmark GNN itself uses a minimal
    pure-PyTorch implementation, while this helper remains available for readers
    who want to experiment with PyG in their own environment.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError(
            "PyTorch Geometric is required for graph_to_pyg_data(). "
            "Install the project requirements before GNN training."
        ) from exc

    kwargs = {
        "x": torch.from_numpy(graph.node_features),
        "edge_index": torch.from_numpy(graph.edge_index),
        "edge_attr": torch.from_numpy(graph.edge_features),
        "smiles": graph.smiles,
    }
    if target is not None:
        kwargs["y"] = torch.tensor([float(target)], dtype=torch.float32)
    return Data(**kwargs)


def save_classical_features(
    frame: pd.DataFrame,
    output_path: Path | str,
    *,
    smiles_col: str = "smiles",
    target_col: str = "target",
    radius: int = 2,
    n_bits: int = 2048,
) -> None:
    """Featurize a cleaned table and save a compressed, reproducible NPZ file."""
    missing = {smiles_col, target_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    smiles = frame[smiles_col].astype(str).tolist()
    targets = pd.to_numeric(frame[target_col], errors="raise").to_numpy(dtype=np.float32)
    matrix = build_classical_matrix(smiles, radius=radius, n_bits=n_bits)
    names = np.asarray(classical_feature_names(n_bits), dtype=str)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=matrix,
        y=targets,
        smiles=np.asarray(smiles, dtype=str),
        feature_names=names,
        radius=np.asarray(radius, dtype=np.int64),
        n_bits=np.asarray(n_bits, dtype=np.int64),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI arguments for baseline feature generation."""
    parser = argparse.ArgumentParser(description="Generate classical molecular features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/solubility_aqsoldb_clean.csv"),
        help="Cleaned CSV produced by src.data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/solubility_aqsoldb_features.npz"),
        help="Compressed NPZ containing X, y, SMILES, and feature names.",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser


def main() -> None:
    """Generate and save the baseline feature matrix."""
    args = build_arg_parser().parse_args()
    frame = pd.read_csv(args.input)
    save_classical_features(
        frame,
        args.output,
        radius=args.radius,
        n_bits=args.n_bits,
    )
    matrix_shape = (len(frame), args.n_bits + len(DESCRIPTOR_NAMES))
    print(f"Wrote classical features to {args.output} with shape {matrix_shape}")


if __name__ == "__main__":
    main()
