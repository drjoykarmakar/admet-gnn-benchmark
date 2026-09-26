"""Training utilities for the small molecular GNN.

The validation partition controls early stopping. Test labels are not evaluated
until the best validation checkpoint has been selected and restored in memory.
Targets are standardized using training-set statistics only.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.evaluate import RegressionMetrics, build_prediction_frame, regression_metrics
from src.features import GraphFeatures, NODE_FEATURE_DIM, smiles_to_graph
from src.models_gnn import (
    MODEL_NAME,
    REPRESENTATION_NAME,
    MolecularGNN,
    count_trainable_parameters,
)


EXPECTED_PARTITIONS = ("train", "val", "test")


@dataclass(frozen=True)
class GraphSample:
    """One molecular graph plus target and audit metadata."""

    graph: GraphFeatures
    target: float
    scaffold: str


@dataclass
class GraphBatch:
    """Concatenated graph tensors for one mini-batch."""

    x: torch.Tensor
    edge_index: torch.Tensor
    batch_index: torch.Tensor
    y: torch.Tensor
    smiles: tuple[str, ...]
    scaffolds: tuple[str, ...]

    def to(self, device: torch.device) -> "GraphBatch":
        """Move tensor fields to a device while preserving string metadata."""
        return GraphBatch(
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            batch_index=self.batch_index.to(device),
            y=self.y.to(device),
            smiles=self.smiles,
            scaffolds=self.scaffolds,
        )


class GraphRegressionDataset(Dataset[GraphSample]):
    """Small in-memory dataset backed by cached RDKit graph featurization."""

    def __init__(self, samples: Sequence[GraphSample]) -> None:
        if not samples:
            raise ValueError("GraphRegressionDataset requires at least one sample.")
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        return self.samples[index]


@dataclass(frozen=True)
class TargetScaler:
    """Training-set target standardization parameters."""

    mean: float
    std: float

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


@dataclass(frozen=True)
class GNNTrainConfig:
    """Configuration for one GNN training run."""

    hidden_size: int = 128
    num_layers: int = 3
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.10
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-5
    num_workers: int = 0
    device: str = "auto"

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        epochs_override: int | None = None,
        device_override: str | None = None,
    ) -> "GNNTrainConfig":
        """Create a validated config from the YAML ``gnn`` section."""
        epochs = int(values.get("epochs", 100))
        if epochs_override is not None:
            epochs = int(epochs_override)
        device = str(values.get("device", "auto"))
        if device_override is not None:
            device = device_override

        config = cls(
            hidden_size=int(values.get("hidden_size", 128)),
            num_layers=int(values.get("num_layers", 3)),
            batch_size=int(values.get("batch_size", 64)),
            epochs=epochs,
            learning_rate=float(values.get("learning_rate", 1e-3)),
            weight_decay=float(values.get("weight_decay", 1e-5)),
            dropout=float(values.get("dropout", 0.10)),
            early_stopping_patience=int(values.get("early_stopping_patience", 15)),
            early_stopping_min_delta=float(values.get("early_stopping_min_delta", 1e-5)),
            num_workers=int(values.get("num_workers", 0)),
            device=device,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail early on nonsensical training settings."""
        if self.hidden_size < 4:
            raise ValueError("hidden_size must be at least 4.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps.")


@dataclass
class GNNRunResult:
    """Outputs from one random/scaffold GNN experiment."""

    split_method: str
    config: GNNTrainConfig
    validation_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    n_train: int
    n_val: int
    n_test: int
    best_epoch: int
    epochs_trained: int
    target_mean: float
    target_std: float
    n_parameters: int
    device: str
    history: pd.DataFrame
    predictions: pd.DataFrame

    def summary_row(self) -> dict[str, Any]:
        """Return a flat row for CSV benchmark tables."""
        return {
            "model": MODEL_NAME,
            "representation": REPRESENTATION_NAME,
            "split_method": self.split_method,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "best_epoch": self.best_epoch,
            "epochs_trained": self.epochs_trained,
            "n_parameters": self.n_parameters,
            "device": self.device,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "val_rmse": self.validation_metrics.rmse,
            "val_mae": self.validation_metrics.mae,
            "val_r2": self.validation_metrics.r2,
            "test_rmse": self.test_metrics.rmse,
            "test_mae": self.test_metrics.mae,
            "test_r2": self.test_metrics.r2,
            **{f"gnn_{key}": value for key, value in asdict(self.config).items()},
        }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable CPU experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` or validate an explicitly requested accelerator."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def collate_graph_samples(samples: Sequence[GraphSample]) -> GraphBatch:
    """Concatenate variable-size molecular graphs without PyTorch Geometric."""
    if not samples:
        raise ValueError("Cannot collate an empty graph batch.")

    node_tensors: list[torch.Tensor] = []
    edge_tensors: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    targets: list[float] = []
    smiles: list[str] = []
    scaffolds: list[str] = []
    node_offset = 0

    for graph_index, sample in enumerate(samples):
        graph = sample.graph
        nodes = torch.from_numpy(graph.node_features).to(dtype=torch.float32)
        edges = torch.from_numpy(graph.edge_index).to(dtype=torch.long)
        if edges.shape[1] > 0:
            edges = edges + node_offset
        node_tensors.append(nodes)
        edge_tensors.append(edges)
        batch_parts.append(torch.full((len(nodes),), graph_index, dtype=torch.long))
        targets.append(float(sample.target))
        smiles.append(graph.smiles)
        scaffolds.append(sample.scaffold)
        node_offset += len(nodes)

    nonempty_edges = [edge for edge in edge_tensors if edge.shape[1] > 0]
    if nonempty_edges:
        edge_index = torch.cat(nonempty_edges, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return GraphBatch(
        x=torch.cat(node_tensors, dim=0),
        edge_index=edge_index,
        batch_index=torch.cat(batch_parts, dim=0),
        y=torch.tensor(targets, dtype=torch.float32),
        smiles=tuple(smiles),
        scaffolds=tuple(scaffolds),
    )


def validate_split_against_cleaned(
    cleaned_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    *,
    target_col: str = "target",
) -> pd.DataFrame:
    """Verify that saved split assignments match the cleaned dataset exactly."""
    required_cleaned = {"smiles", target_col}
    required_split = {"smiles", target_col, "split"}
    missing_cleaned = required_cleaned.difference(cleaned_frame.columns)
    missing_split = required_split.difference(split_frame.columns)
    if missing_cleaned:
        raise ValueError(f"Cleaned table is missing columns: {sorted(missing_cleaned)}")
    if missing_split:
        raise ValueError(f"Split table is missing columns: {sorted(missing_split)}")

    cleaned = cleaned_frame.loc[:, ["smiles", target_col]].copy()
    split = split_frame.copy()
    cleaned["smiles"] = cleaned["smiles"].astype(str)
    split["smiles"] = split["smiles"].astype(str)
    cleaned[target_col] = pd.to_numeric(cleaned[target_col], errors="raise")
    split[target_col] = pd.to_numeric(split[target_col], errors="raise")

    if cleaned["smiles"].duplicated().any():
        raise ValueError("Cleaned table contains duplicate standardized SMILES.")
    if split["smiles"].duplicated().any():
        raise ValueError("Split table contains duplicate standardized SMILES.")
    if not np.isfinite(cleaned[target_col].to_numpy(dtype=float)).all():
        raise ValueError("Cleaned targets contain non-finite values.")
    if not np.isfinite(split[target_col].to_numpy(dtype=float)).all():
        raise ValueError("Split targets contain non-finite values.")

    observed = set(split["split"].astype(str))
    unknown = observed.difference(EXPECTED_PARTITIONS)
    if unknown:
        raise ValueError(f"Unknown split labels: {sorted(unknown)}")
    missing_partitions = set(EXPECTED_PARTITIONS).difference(observed)
    if missing_partitions:
        raise ValueError(f"Split table is missing partitions: {sorted(missing_partitions)}")

    clean_smiles = set(cleaned["smiles"])
    split_smiles = set(split["smiles"])
    if clean_smiles != split_smiles:
        raise ValueError(
            "Cleaned/split molecule sets differ: "
            f"{len(clean_smiles - split_smiles)} only in cleaned data, "
            f"{len(split_smiles - clean_smiles)} only in split table."
        )

    target_lookup = cleaned.set_index("smiles")[target_col]
    expected_targets = split["smiles"].map(target_lookup).to_numpy(dtype=float)
    observed_targets = split[target_col].to_numpy(dtype=float)
    if not np.allclose(expected_targets, observed_targets, rtol=1e-6, atol=1e-6):
        max_difference = float(np.max(np.abs(expected_targets - observed_targets)))
        raise ValueError(
            "Targets in split table do not match cleaned data; "
            f"maximum absolute difference is {max_difference:.6g}."
        )

    if "scaffold" not in split.columns:
        split["scaffold"] = ""
    split["scaffold"] = split["scaffold"].fillna("").astype(str)
    return split


def _build_dataset(
    split: pd.DataFrame,
    partition: str,
    graph_cache: MutableMapping[str, GraphFeatures],
) -> GraphRegressionDataset:
    """Create one partition dataset, reusing graph features across split runs."""
    rows = split.loc[split["split"] == partition]
    if rows.empty:
        raise ValueError(f"Partition {partition!r} is empty.")

    samples: list[GraphSample] = []
    for row in rows.itertuples(index=False):
        smiles = str(row.smiles)
        graph = graph_cache.get(smiles)
        if graph is None:
            graph = smiles_to_graph(smiles)
            graph_cache[smiles] = graph
        samples.append(
            GraphSample(
                graph=graph,
                target=float(row.target),
                scaffold=str(row.scaffold),
            )
        )
    return GraphRegressionDataset(samples)


def _target_scaler(dataset: GraphRegressionDataset) -> TargetScaler:
    """Estimate target mean/std from the training partition only."""
    targets = np.asarray([sample.target for sample in dataset.samples], dtype=float)
    mean = float(targets.mean())
    std = float(targets.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Training targets must be finite.")
    if std < 1e-8:
        std = 1.0
    return TargetScaler(mean=mean, std=std)


def _make_loader(
    dataset: GraphRegressionDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[GraphSample]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _train_epoch(
    model: MolecularGNN,
    loader: DataLoader[GraphSample],
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
) -> float:
    model.train()
    loss_function = nn.MSELoss()
    total_squared_loss = 0.0
    n_examples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(batch.x, batch.edge_index, batch.batch_index)
        targets = scaler.transform_tensor(batch.y)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        total_squared_loss += float(loss.detach().item()) * len(batch.y)
        n_examples += len(batch.y)

    return total_squared_loss / n_examples


@torch.no_grad()
def _predict_loader(
    model: MolecularGNN,
    loader: DataLoader[GraphSample],
    scaler: TargetScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], float]:
    """Predict in original target units and return scaled MSE for early stopping."""
    model.eval()
    true_values: list[np.ndarray] = []
    predicted_values: list[np.ndarray] = []
    smiles: list[str] = []
    scaffolds: list[str] = []
    scaled_squared_error = 0.0
    n_examples = 0

    for batch in loader:
        batch = batch.to(device)
        scaled_predictions = model(batch.x, batch.edge_index, batch.batch_index)
        scaled_targets = scaler.transform_tensor(batch.y)
        scaled_squared_error += float(
            torch.sum((scaled_predictions - scaled_targets) ** 2).item()
        )
        n_examples += len(batch.y)

        predictions = scaler.inverse_tensor(scaled_predictions)
        true_values.append(batch.y.detach().cpu().numpy())
        predicted_values.append(predictions.detach().cpu().numpy())
        smiles.extend(batch.smiles)
        scaffolds.extend(batch.scaffolds)

    y_true = np.concatenate(true_values).astype(float, copy=False)
    y_pred = np.concatenate(predicted_values).astype(float, copy=False)
    return y_true, y_pred, smiles, scaffolds, scaled_squared_error / n_examples


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy a model state to CPU so later epochs cannot mutate the best state."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def run_gnn_for_split(
    cleaned_frame: pd.DataFrame,
    split_frame: pd.DataFrame,
    *,
    split_method: str,
    config: GNNTrainConfig,
    seed: int,
    graph_cache: MutableMapping[str, GraphFeatures] | None = None,
) -> GNNRunResult:
    """Train with validation early stopping, then evaluate the held-out test set."""
    config.validate()
    seed_everything(seed)
    device = resolve_device(config.device)
    split = validate_split_against_cleaned(cleaned_frame, split_frame)
    cache: MutableMapping[str, GraphFeatures] = graph_cache if graph_cache is not None else {}

    train_dataset = _build_dataset(split, "train", cache)
    val_dataset = _build_dataset(split, "val", cache)
    test_dataset = _build_dataset(split, "test", cache)
    scaler = _target_scaler(train_dataset)

    train_loader = _make_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed,
        num_workers=config.num_workers,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=seed,
        num_workers=config.num_workers,
    )

    model = MolecularGNN(
        input_dim=NODE_FEATURE_DIM,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scaler, device)
        val_true, val_pred, _, _, val_loss = _predict_loader(
            model, val_loader, scaler, device
        )
        val_metrics = regression_metrics(val_true, val_pred)

        improved = val_loss < best_val_loss - config.early_stopping_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = _clone_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "split_method": split_method,
                "epoch": epoch,
                "train_scaled_mse": train_loss,
                "val_scaled_mse": val_loss,
                "val_rmse": val_metrics.rmse,
                "val_mae": val_metrics.mae,
                "val_r2": val_metrics.r2,
                "is_best": improved,
            }
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("GNN training did not produce a valid validation checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    val_true, val_pred, val_smiles, val_scaffolds, _ = _predict_loader(
        model, val_loader, scaler, device
    )
    test_true, test_pred, test_smiles, test_scaffolds, _ = _predict_loader(
        model, test_loader, scaler, device
    )
    validation_metrics = regression_metrics(val_true, val_pred)
    test_metrics = regression_metrics(test_true, test_pred)

    val_predictions = build_prediction_frame(
        smiles=val_smiles,
        y_true=val_true,
        y_pred=val_pred,
        partition="val",
        split_method=split_method,
        model_name=MODEL_NAME,
        scaffold=val_scaffolds,
    )
    test_predictions = build_prediction_frame(
        smiles=test_smiles,
        y_true=test_true,
        y_pred=test_pred,
        partition="test",
        split_method=split_method,
        model_name=MODEL_NAME,
        scaffold=test_scaffolds,
    )

    return GNNRunResult(
        split_method=split_method,
        config=config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        n_train=len(train_dataset),
        n_val=len(val_dataset),
        n_test=len(test_dataset),
        best_epoch=best_epoch,
        epochs_trained=len(history_rows),
        target_mean=scaler.mean,
        target_std=scaler.std,
        n_parameters=count_trainable_parameters(model),
        device=str(device),
        history=pd.DataFrame(history_rows),
        predictions=pd.concat([val_predictions, test_predictions], ignore_index=True),
    )
