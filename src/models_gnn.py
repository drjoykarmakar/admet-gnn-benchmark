"""Small pure-PyTorch graph neural network used by the benchmark.

PyTorch Geometric is intentionally not required. The model implements a readable
mean-neighbor message-passing layer with ``index_add_`` and global mean pooling.
This keeps the portfolio project easy to install while preserving the key graph
learning mechanics a reviewer should be able to inspect directly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.features import NODE_FEATURE_DIM


MODEL_NAME = "Small GNN"
REPRESENTATION_NAME = "Molecular graph"


class MeanMessagePassing(nn.Module):
    """Aggregate mean neighbor states, then combine them with each atom state."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.neighbor_linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return one message-passing update for all atoms in a graph batch."""
        if x.ndim != 2:
            raise ValueError(f"x must have shape [n_nodes, n_features]; got {tuple(x.shape)}.")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, n_edges]; "
                f"got {tuple(edge_index.shape)}."
            )

        neighbor_mean = torch.zeros_like(x)
        if edge_index.shape[1] > 0:
            source, destination = edge_index
            neighbor_sum = torch.zeros_like(x)
            neighbor_sum.index_add_(0, destination, x[source])

            degree = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            degree.index_add_(
                0,
                destination,
                torch.ones(destination.shape[0], dtype=x.dtype, device=x.device),
            )
            neighbor_mean = neighbor_sum / degree.clamp_min(1.0).unsqueeze(1)

        return self.self_linear(x) + self.neighbor_linear(neighbor_mean)


def global_mean_pool(
    x: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    num_graphs: int | None = None,
) -> torch.Tensor:
    """Pool node embeddings to one mean vector per molecular graph."""
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if batch_index.ndim != 1 or len(batch_index) != len(x):
        raise ValueError("batch_index must contain one graph index per node.")
    if len(x) == 0:
        raise ValueError("Cannot pool an empty node tensor.")

    inferred = int(batch_index.max().item()) + 1
    n_graphs = inferred if num_graphs is None else int(num_graphs)
    if n_graphs < inferred:
        raise ValueError("num_graphs is smaller than the graph IDs in batch_index.")

    pooled = torch.zeros((n_graphs, x.shape[1]), dtype=x.dtype, device=x.device)
    pooled.index_add_(0, batch_index, x)
    counts = torch.bincount(batch_index, minlength=n_graphs).to(dtype=x.dtype)
    if torch.any(counts == 0):
        raise ValueError("Every graph in the batch must contain at least one atom.")
    return pooled / counts.unsqueeze(1)


class MolecularGNN(nn.Module):
    """Small residual mean-aggregation GNN for molecular regression."""

    def __init__(
        self,
        *,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_size < 4:
            raise ValueError("hidden_size must be at least 4.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")

        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.message_layers = nn.ModuleList(
            MeanMessagePassing(hidden_size) for _ in range(num_layers)
        )
        self.norm_layers = nn.ModuleList(
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        )

        head_size = max(hidden_size // 2, 4)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one standardized regression target per molecular graph."""
        h = F.relu(self.input_projection(x))
        for message_layer, norm_layer in zip(self.message_layers, self.norm_layers):
            update = message_layer(h, edge_index)
            update = norm_layer(update)
            update = F.relu(update)
            update = F.dropout(update, p=self.dropout, training=self.training)
            h = h + update

        pooled = global_mean_pool(h, batch_index)
        return self.regression_head(pooled).squeeze(-1)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
