"""Whitening map construction for the global Nystr\"om basis.

Implements Section 4 of the method blueprint: soft-truncated whitening of
the kernel-on-landmarks matrix.

The whitening map ``M_g`` serves two purposes:

1. **Dimensionality reduction**: Retains only the ``r_g`` eigenvectors
   whose eigenvalues exceed the truncation threshold ``tau_eig``.
2. **Covariance normalization**: Scales the retained eigenvectors so that
   the whitened features have approximately unit covariance, improving
   the conditioning of the downstream ridge regression.

The soft-truncation formula avoids hard discontinuities when eigenvalues
cross the threshold, which is important for gradient-based optimization
of the projection matrix ``R`` (the outer-loop objective).

Algorithm
---------
Given the kernel-on-landmarks matrix ``W = k(Z, Z)``:

1. Eigendecompose ``W = U Lambda U^T``.
2. Clip eigenvalues to enforce numerical PSDness.
3. Compute dataset-scale epsilon: ``epsilon = alpha_epsilon * tr(W) / m_g``.
4. Apply soft spectral truncation:
   ``tilde_lambda_i = lambda_i / (lambda_i + tau) * (lambda_i + epsilon)^{-1/2}``
5. Retain indices where ``lambda_i >= tau_eig``.
6. Build ``M_g = U_{I_g} @ diag(tilde_Lambda_{I_g})``.

Complexity
----------
O(m_g^3) for the eigendecomposition, dominated by the symmetric eigenvalue
computation on the ``m_g x m_g`` kernel matrix.
"""

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array
from aware_kernel.utils.numerics import (
    compute_epsilon,
    eigenvalue_clip,
    retained_indices,
    soft_spectral_truncate,
)


def build_whitening_map(
    W: Array,
    config: NumericsConfig,
) -> tuple[Array, Array, Array, int]:
    """Build the soft-truncated whitening map ``M_g``.

    Given the kernel-on-landmarks matrix ``W = k(Z, Z)``, computes the
    full whitening pipeline:

    1. Eigendecomposition ``W = U Lambda U^T``.
    2. Eigenvalue clipping to enforce numerical PSDness.
    3. Dataset-scale epsilon computation.
    4. Soft spectral truncation.
    5. Retained index set ``I_g`` and rank ``r_g``.

    The resulting ``M_g`` is the whitening map that maps kernel vectors
    to whitened features: ``phi_g = k(u, Z) @ M_g``.

    Args:
        W: Kernel-on-landmarks matrix of shape ``(m_g, m_g)``.  Must be
            square and symmetric.
        config: Numerical stability configuration controlling
            ``tau_eig``, ``alpha_epsilon``, and other thresholds.

    Returns:
        Tuple of ``(U, Lambda_clipped, M_g, r_g)`` where:
            ``U``: Eigenvectors of shape ``(m_g, m_g)``.
            ``Lambda_clipped``: Clipped eigenvalues of shape ``(m_g,)``.
            ``M_g``: Whitening map of shape ``(m_g, r_g)``.
            ``r_g``: Retained rank.

    Raises:
        ValueError: If ``W`` is not square.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")

    # Step 1: Eigendecomposition of the symmetric kernel matrix.
    # np.linalg.eigh is used (not eig) because W is guaranteed symmetric
    # PSD (up to numerical errors), and eigh is both faster and more
    # numerically stable for symmetric matrices.
    eigenvalues, U = np.linalg.eigh(W)

    # Step 2: Clip eigenvalues to enforce numerical PSDness.
    # Rounding errors can produce tiny negative eigenvalues; clipping to
    # zero ensures the matrix is treated as positive semi-definite.
    clipped = eigenvalue_clip(eigenvalues, min_val=0.0)

    # Step 3: Dataset-scale epsilon makes the stabilization epsilon
    # proportional to the average eigenvalue, so it adapts to the
    # scale of the kernel matrix without manual tuning.
    trace_w = float(np.sum(clipped))
    m_g = W.shape[0]
    epsilon = compute_epsilon(
        trace_w=trace_w,
        m_g=m_g,
        alpha_epsilon=config.alpha_epsilon,
    )

    # Step 4: Soft spectral truncation applies a smooth down-weighting
    # to eigenvalues near the threshold, avoiding hard discontinuities
    # that could cause non-smooth gradients in the outer objective.
    truncated = soft_spectral_truncate(
        clipped,
        tau=config.tau_eig,
        epsilon=epsilon,
    )

    # Step 5: Determine which eigenvalues are large enough to retain.
    indices, r_g = retained_indices(clipped, tau=config.tau_eig)

    if r_g == 0:
        # Degenerate case: no components retained.  Return a zero matrix
        # to avoid downstream shape errors.  The caller should detect
        # r_g == 0 and handle this gracefully.
        M_g = np.zeros((m_g, 1), dtype=W.dtype)
        return U, clipped, M_g, 0

    # Step 6: Build M_g = U_{I_g} @ diag(tilde_Lambda_{I_g}).
    # This projects onto the retained eigenspace and applies the
    # whitening scaling in a single matrix multiplication.
    M_g = U[:, indices] @ np.diag(truncated[indices])

    return U, clipped, M_g, r_g


def compute_kernel_on_landmarks(Z: Array, gamma: float = 1.0) -> Array:
    """Compute the RBF kernel matrix on landmarks.

    Computes ``W_{ij} = exp(-gamma * ||z_i - z_j||^2)`` for all pairs
    of landmarks.  The squared distances are computed via the identity:

        ``||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i^T z_j``

    which avoids materializing the full difference tensor.

    The default ``gamma = 1.0`` corresponds to a bandwidth of
    ``sigma = 1 / sqrt(2)`` in the standard RBF parameterization
    ``k(x, x') = exp(-||x - x'||^2 / (2 sigma^2))``.

    Args:
        Z: Landmarks of shape ``(m_g, d)``.
        gamma: RBF kernel bandwidth parameter.  Larger values produce
            tighter kernels (more local features).

    Returns:
        Kernel matrix ``W`` of shape ``(m_g, m_g)``.
    """
    sq_dists = (
        np.sum(Z**2, axis=1).reshape(-1, 1)
        + np.sum(Z**2, axis=1).reshape(1, -1)
        - 2.0 * (Z @ Z.T)
    )
    result: Array = np.exp(-gamma * sq_dists)
    return result
