"""Normal equation accumulation helpers.

Implements Section 9:
    S = Phi^T Phi
    b = Phi^T y
"""

import numpy as np

from aware_kernel.aware.types import Array


def accumulate_normal_matrix(phi_batch: Array) -> Array:
    """Compute S_batch = Phi_batch^T Phi_batch.

    Args:
        phi_batch: Feature batch of shape (batch_size, m).

    Returns:
        Normal matrix batch contribution of shape (m, m).
    """
    return phi_batch.T @ phi_batch


def accumulate_normal_vector(phi_batch: Array, y_batch: Array) -> Array:
    """Compute b_batch = Phi_batch^T y_batch.

    Args:
        phi_batch: Feature batch of shape (batch_size, m).
        y_batch: Target batch of shape (batch_size,).

    Returns:
        Normal vector batch contribution of shape (m,).
    """
    return phi_batch.T @ y_batch


def build_normal_equations(
    phi: Array,
    y: Array,
    lambda_reg: float,
) -> tuple[Array, Array]:
    """Build full normal equations with ridge regularization.

    Args:
        phi: Feature matrix of shape (n, m).
        y: Target vector of shape (n,).
        lambda_reg: Ridge regularization parameter.

    Returns:
        Tuple of (S, b) where S has shape (m, m) and b has shape (m,).
    """
    s = phi.T @ phi + lambda_reg * np.eye(phi.shape[1])
    b = phi.T @ y
    return s, b
