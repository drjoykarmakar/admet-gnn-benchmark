"""Tests for regression evaluation helpers."""

import numpy as np

from src.evaluate import build_prediction_frame, regression_metrics


def test_regression_metrics_match_simple_example() -> None:
    metrics = regression_metrics([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
    assert np.isclose(metrics.rmse, np.sqrt(2.0 / 3.0))
    assert np.isclose(metrics.mae, 2.0 / 3.0)
    assert np.isclose(metrics.r2, 0.0)
    assert metrics.n == 3


def test_prediction_frame_uses_prediction_minus_observation_residual() -> None:
    frame = build_prediction_frame(
        smiles=["CCO"],
        y_true=[-1.5],
        y_pred=[-1.0],
        partition="test",
        split_method="random",
        model_name="RF",
    )
    assert np.isclose(frame.loc[0, "residual"], 0.5)
    assert np.isclose(frame.loc[0, "abs_error"], 0.5)
