"""Evaluation metrics for regression experiments.

Standard regression metrics used to compare aware-kernel against baselines.
All metrics operate on flat NumPy arrays of true and predicted values.

Metrics provided:

* **RMSE** (Root Mean Squared Error): Penalizes large errors quadratically.
  ``sqrt(mean((y_true - y_pred)^2))``
* **MAE** (Mean Absolute Error): Robust to outliers.  ``mean(|y_true - y_pred|)``
* **R^2** (Coefficient of Determination): Proportion of variance explained.
  Capped at 1.0 to handle numerical overshoot.
* **Max Absolute Error**: Worst-case prediction error.

``compute_all_metrics`` returns all metrics in a single dict for convenience.
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_rmse(y_true: Array, y_pred: Array) -> float:
    """Compute root mean squared error.

    Args:
        y_true: Ground truth targets of shape (n,).
        y_pred: Predicted targets of shape (n,).

    Returns:
        RMSE value.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: Array, y_pred: Array) -> float:
    """Compute mean absolute error.

    Args:
        y_true: Ground truth targets of shape (n,).
        y_pred: Predicted targets of shape (n,).

    Returns:
        MAE value.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true: Array, y_pred: Array) -> float:
    """Compute coefficient of determination (R^2).

    R^2 measures the proportion of variance in ``y_true`` explained by the
    predictions.  Defined as ``1 - SS_res / SS_tot`` where ``SS_res`` is the
    residual sum of squares and ``SS_tot`` is the total sum of squares.

    A value of 1.0 indicates perfect prediction; 0.0 means the model is no
    better than predicting the mean; negative values indicate worse-than-mean
    performance.  The result is capped at 1.0 to guard against numerical
    overshoot from floating-point arithmetic.

    Args:
        y_true: Ground truth targets of shape ``(n,)``.
        y_pred: Predicted targets of shape ``(n,)``.

    Returns:
        R^2 value, capped at 1.0.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(min(r2, 1.0))


def compute_max_abs_error(y_true: Array, y_pred: Array) -> float:
    """Compute maximum absolute error.

    Args:
        y_true: Ground truth targets of shape (n,).
        y_pred: Predicted targets of shape (n,).

    Returns:
        Maximum absolute error.
    """
    return float(np.max(np.abs(y_true - y_pred)))


def compute_all_metrics(y_true: Array, y_pred: Array) -> dict:
    """Compute all available metrics.

    Args:
        y_true: Ground truth targets of shape (n,).
        y_pred: Predicted targets of shape (n,).

    Returns:
        Dictionary with keys "rmse", "mae", "r2", "max_abs_error".
    """
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "r2": compute_r2(y_true, y_pred),
        "max_abs_error": compute_max_abs_error(y_true, y_pred),
    }
