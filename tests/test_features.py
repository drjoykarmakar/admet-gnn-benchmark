"""Unit tests for molecular featurization."""

import numpy as np

from src.features import (
    DESCRIPTOR_NAMES,
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    build_classical_matrix,
    classical_feature_vector,
    smiles_to_graph,
)


def test_classical_feature_vector_has_expected_size() -> None:
    vector = classical_feature_vector("CCO", n_bits=128)
    assert vector.shape == (128 + len(DESCRIPTOR_NAMES),)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_classical_matrix_is_deterministic() -> None:
    smiles = ["CCO", "c1ccccc1"]
    first = build_classical_matrix(smiles, n_bits=64)
    second = build_classical_matrix(smiles, n_bits=64)
    np.testing.assert_array_equal(first, second)


def test_graph_is_bidirectional_and_dimensionally_stable() -> None:
    graph = smiles_to_graph("CCO")
    assert graph.node_features.shape == (3, NODE_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 4)  # two bonds, both directions
    assert graph.edge_features.shape == (4, EDGE_FEATURE_DIM)
    assert graph.node_features.dtype == np.float32


def test_single_atom_graph_has_empty_edge_arrays() -> None:
    graph = smiles_to_graph("[Na+]")
    assert graph.node_features.shape == (1, NODE_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_features.shape == (0, EDGE_FEATURE_DIM)
