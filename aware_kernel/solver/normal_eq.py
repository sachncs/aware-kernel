"""Normal equation accumulation helpers.

Implements Section 9 of the method blueprint: assembling the normal
equations for ridge regression.

The normal equations for ridge regression are:

    ``S = Phi^T Phi + lambda * I``
    ``b = Phi^T y``

These are the sufficient statistics for the ridge regression problem.
They can be computed either from the full feature matrix (cached mode)
or accumulated incrementally from mini-batches (streamed mode).

Design rationale
----------------
Separating normal equation assembly from the solver allows the memory
accumulators (cached/streamed) to construct ``S`` and ``b`` independently
of the solver strategy (direct/iterative).  This clean separation means
the memory mode and solver can be mixed freely.
"""

import numpy as np

from aware_kernel.aware.types import Array


def accumulate_normal_matrix(phi_batch: Array) -> Array:
    """Compute ``S_batch = Phi_batch^T Phi_batch``.

    This is the batch contribution to the normal matrix.  For the
    streamed accumulator, these contributions are summed over all
    mini-batches to form the full ``S``.

    Args:
        phi_batch: Feature batch of shape ``(batch_size, m)``.

    Returns:
        Normal matrix batch contribution of shape ``(m, m)``.
    """
    return phi_batch.T @ phi_batch


def accumulate_normal_vector(phi_batch: Array, y_batch: Array) -> Array:
    """Compute ``b_batch = Phi_batch^T y_batch``.

    This is the batch contribution to the normal vector.  For the
    streamed accumulator, these contributions are summed over all
    mini-batches to form the full ``b``.

    Args:
        phi_batch: Feature batch of shape ``(batch_size, m)``.
        y_batch: Target batch of shape ``(batch_size,)``.

    Returns:
        Normal vector batch contribution of shape ``(m,)``.
    """
    return phi_batch.T @ y_batch


def build_normal_equations(
    phi: Array,
    y: Array,
    lambda_reg: float,
) -> tuple[Array, Array]:
    """Build full normal equations with ridge regularization.

    Constructs the system ``(Phi^T Phi + lambda I) w = Phi^T y`` from
    the complete feature matrix and target vector.  This is used by
    the cached memory mode and by solvers that receive the full data.

    Args:
        phi: Feature matrix of shape ``(n, m)``.
        y: Target vector of shape ``(n,)``.
        lambda_reg: Ridge regularization parameter.

    Returns:
        Tuple of ``(S, b)`` where ``S`` has shape ``(m, m)`` and
        ``b`` has shape ``(m,)``.
    """
    s = phi.T @ phi + lambda_reg * np.eye(phi.shape[1])
    b = phi.T @ y
    return s, b
