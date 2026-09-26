"""Unit tests for the minimal pure-PyTorch molecular GNN."""

import torch

from src.features import NODE_FEATURE_DIM, smiles_to_graph
from src.models_gnn import MolecularGNN, count_trainable_parameters, global_mean_pool
from src.train import GraphSample, collate_graph_samples


def test_collate_offsets_edges_and_preserves_graph_membership() -> None:
    samples = [
        GraphSample(smiles_to_graph("CCO"), -1.0, "<ACYCLIC>"),
        GraphSample(smiles_to_graph("CC"), -2.0, "<ACYCLIC>"),
    ]
    batch = collate_graph_samples(samples)

    assert batch.x.shape == (5, NODE_FEATURE_DIM)
    assert batch.edge_index.shape == (2, 6)
    assert batch.batch_index.tolist() == [0, 0, 0, 1, 1]
    assert batch.y.tolist() == [-1.0, -2.0]
    assert int(batch.edge_index.max()) == 4


def test_gnn_forward_returns_one_finite_value_per_graph() -> None:
    samples = [
        GraphSample(smiles_to_graph("CCO"), -1.0, "<ACYCLIC>"),
        GraphSample(smiles_to_graph("c1ccccc1"), -2.0, "c1ccccc1"),
    ]
    batch = collate_graph_samples(samples)
    model = MolecularGNN(hidden_size=32, num_layers=2, dropout=0.0)
    prediction = model(batch.x, batch.edge_index, batch.batch_index)

    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()
    assert count_trainable_parameters(model) > 0


def test_gnn_handles_graph_with_no_bonds() -> None:
    batch = collate_graph_samples(
        [GraphSample(smiles_to_graph("[Na+]"), 0.0, "<ACYCLIC>")]
    )
    model = MolecularGNN(hidden_size=16, num_layers=1, dropout=0.0)
    prediction = model(batch.x, batch.edge_index, batch.batch_index)
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()


def test_global_mean_pool_matches_manual_means() -> None:
    x = torch.tensor([[1.0], [3.0], [10.0], [14.0]])
    batch_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    pooled = global_mean_pool(x, batch_index)
    torch.testing.assert_close(pooled, torch.tensor([[2.0], [12.0]]))
