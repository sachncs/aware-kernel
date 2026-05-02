"""Numerical utilities for eigenvalue clipping, soft truncation, and conditioning checks.

Implements Sections 4 (soft-truncated whitening) and 6 (numerical stabilization)
from the method blueprint.
"""

from typing import Tuple

import numpy as np

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.aware.types import Array


def eigenvalue_clip(eigenvalues: Array, min_val: float = 0.0) -> Array:
    """Clip eigenvalues below a threshold to enforce numerical PSDness.

    Args:
        eigenvalues: 1-D array of eigenvalues.
        min_val: Floor value; eigenvalues below this are clamped.

    Returns:
        Clipped eigenvalues.
    """
    return np.maximum(eigenvalues, min_val)


def soft_spectral_truncate(
    eigenvalues: Array,
    tau: float,
    epsilon: float,
) -> Array:
    """Compute soft-truncated eigenvalues for whitening.

    For each eigenvalue lambda_i:
        tilde_lambda_i = (lambda_i + epsilon)^(-1/2) * lambda_i / (lambda_i + tau)

    This avoids hard discontinuities when eigenvalues cross the threshold.

    Args:
        eigenvalues: Clipped eigenvalues (>= 0).
        tau: Soft-truncation threshold (tau_eig).
        epsilon: Stabilization epsilon (alpha_epsilon * tr(W) / m_g).

    Returns:
        Soft-truncated whitening eigenvalues.
    """
    safe = eigenvalues + epsilon
    scale = eigenvalues / (eigenvalues + tau)
    return scale / np.sqrt(safe)


def compute_epsilon(trace_w: float, m_g: int, alpha_epsilon: float) -> float:
    """Dataset-scale invariant epsilon for Nystr\"om whitening.

    Args:
        trace_w: Trace of the kernel-on-landmarks matrix W.
        m_g: Number of landmarks.
        alpha_epsilon: Scale factor.

    Returns:
        Epsilon value.
    """
    if m_g <= 0:
        raise ValueError("m_g must be positive")
    return alpha_epsilon * (trace_w / m_g)


def retained_indices(eigenvalues: Array, tau: float) -> Tuple[Array, int]:
    """Determine which eigenvalues exceed the retention threshold.

    Args:
        eigenvalues: Clipped eigenvalues.
        tau: Retention threshold.

    Returns:
        Tuple of (indices array, rank).
    """
    indices = np.where(eigenvalues >= tau)[0]
    rank = int(indices.size)
    return indices, rank


def check_conditioning(matrix: Array, threshold: float, name: str = "matrix") -> float:
    """Check condition number and raise if it exceeds threshold.

    Args:
        matrix: Square matrix to check.
        threshold: Maximum acceptable condition number.
        name: Name of matrix for error messages.

    Returns:
        Computed condition number.

    Raises:
        ConditioningError: If condition number exceeds threshold.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    cond = float(np.linalg.cond(matrix))
    if cond > threshold:
        raise ConditioningError(
            f"{name} condition number {cond:.3e} exceeds threshold {threshold:.3e}"
        )
    return cond


def safe_cholesky(matrix: Array, jitter: float = 1e-10) -> Array:
    """Compute Cholesky decomposition with jitter fallback.

    Args:
        matrix: Symmetric positive-definite matrix.
        jitter: Increment added to diagonal if initial decomposition fails.

    Returns:
        Lower-triangular Cholesky factor.

    Raises:
        ConditioningError: If decomposition fails even with jitter.
    """
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        jittered = matrix + jitter * np.eye(matrix.shape[0])
        try:
            return np.linalg.cholesky(jittered)
        except np.linalg.LinAlgError as exc:
            raise ConditioningError(
                f"Cholesky failed even with jitter {jitter}: {exc}"
            ) from exc
