"""Numerical utilities for eigenvalue clipping, soft truncation, and conditioning checks.

Implements Sections 4 (soft-truncated whitening) and 6 (numerical
stabilization) from the method blueprint.

These utilities provide the low-level numerical building blocks used by
the whitening pipeline, solver, and orthogonalization stages.  They
enforce the numerical invariants that make the method robust across a
range of dataset scales and feature dimensions.

Key invariants enforced:

* Eigenvalues are non-negative after clipping (PSD guarantee).
* The whitening eigenvalues are well-defined (no division by zero).
* The condition number of the normal equations matrix is bounded.
* The Cholesky decomposition succeeds (with jitter fallback if needed).

Complexity
----------
All functions operate on matrices of size ``m x m`` where ``m`` is the
feature dimension.  Most are O(m^2) or O(m^3) depending on whether they
involve matrix operations or elementwise operations.
"""

import numpy as np

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.aware.types import Array


def eigenvalue_clip(eigenvalues: Array, min_val: float = 0.0) -> Array:
    """Clip eigenvalues below a threshold to enforce numerical PSDness.

    Rounding errors in eigendecomposition can produce tiny negative
    eigenvalues for matrices that are theoretically positive
    semi-definite.  This function clamps them to a minimum value.

    Args:
        eigenvalues: 1-D array of eigenvalues.
        min_val: Floor value.  Eigenvalues below this are clamped.
            Default ``0.0`` for PSD enforcement.

    Returns:
        Clipped eigenvalues with ``min(eigenvalues) >= min_val``.
    """
    return np.maximum(eigenvalues, min_val)


def soft_spectral_truncate(
    eigenvalues: Array,
    tau: float,
    epsilon: float,
) -> Array:
    """Compute soft-truncated eigenvalues for whitening.

    For each eigenvalue ``lambda_i``, the whitened eigenvalue is:

        ``tilde_lambda_i = lambda_i / (lambda_i + tau) * (lambda_i + epsilon)^{-1/2}``

    This formula combines two operations:

    * **Soft truncation** (``lambda / (lambda + tau)``): Smoothly
      down-weights eigenvalues near the threshold ``tau``, avoiding
      hard discontinuities that could cause non-smooth gradients.
    * **Whitening** (``(lambda + epsilon)^{-1/2}``): Normalizes the
      eigenvalue to approximately unity, improving conditioning of the
      downstream ridge regression.

    The ``epsilon`` term prevents division by zero for eigenvalues
    near zero.

    Args:
        eigenvalues: Clipped eigenvalues (>= 0).
        tau: Soft-truncation threshold (``tau_eig``).
        epsilon: Stabilization epsilon (dataset-scale invariant).

    Returns:
        Soft-truncated whitening eigenvalues.
    """
    safe = eigenvalues + epsilon
    scale = eigenvalues / (eigenvalues + tau)
    result: Array = scale / np.sqrt(safe)
    return result


def compute_epsilon(trace_w: float, m_g: int, alpha_epsilon: float) -> float:
    """Dataset-scale invariant epsilon for Nystr\"om whitening.

    Computes ``epsilon = alpha_epsilon * tr(W) / m_g``, which makes the
    stabilization epsilon proportional to the average eigenvalue of the
    kernel matrix.  This adapts to the scale of the data without manual
    tuning.

    Args:
        trace_w: Trace of the kernel-on-landmarks matrix ``W``.
        m_g: Number of landmarks.
        alpha_epsilon: Scale factor (typically ``1e-5``).

    Returns:
        Epsilon value proportional to the average eigenvalue.

    Raises:
        ValueError: If ``m_g <= 0``.
    """
    if m_g <= 0:
        raise ValueError("m_g must be positive")
    return alpha_epsilon * (trace_w / m_g)


def retained_indices(eigenvalues: Array, tau: float) -> tuple[Array, int]:
    """Determine which eigenvalues exceed the retention threshold.

    Eigenvalues above ``tau`` are retained; those below are discarded.
    The retained rank ``r_g`` is the count of retained eigenvalues.

    Args:
        eigenvalues: Clipped eigenvalues (>= 0).
        tau: Retention threshold (``tau_eig``).

    Returns:
        Tuple of ``(indices array, rank)`` where ``indices`` is a 1-D
        array of retained indices and ``rank`` is the count.
    """
    indices = np.where(eigenvalues >= tau)[0]
    rank = int(indices.size)
    return indices, rank


def check_conditioning(matrix: Array, threshold: float, name: str = "matrix") -> float:
    """Check condition number and raise if it exceeds threshold.

    The condition number ``kappa = lambda_max / lambda_min`` measures
    how sensitive the matrix inverse is to perturbations in the input.
    A high condition number indicates numerical instability.

    Args:
        matrix: Square matrix to check.
        threshold: Maximum acceptable condition number.
        name: Name of matrix for error messages (used in
            ``ConditioningError``).

    Returns:
        Computed condition number.

    Raises:
        ValueError: If the matrix is not square.
        ConditioningError: If the condition number exceeds threshold.
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

    Attempts a standard Cholesky decomposition.  If it fails (e.g.,
    due to near-singular or numerically indefinite matrices), adds a
    small jitter to the diagonal and retries.

    The jitter is a standard technique in Gaussian process literature
    for handling numerical indefiniteness.  The fallback ensures the
    solver remains robust even when the normal equations matrix is
    marginally non-positive-definite.

    Args:
        matrix: Symmetric positive-definite matrix.
        jitter: Increment added to the diagonal on fallback.  Default
            ``1e-10``.

    Returns:
        Lower-triangular Cholesky factor ``L`` such that
        ``L @ L^T ≈ matrix``.

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
