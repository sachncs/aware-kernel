"""Whitening map construction for the global Nystr\"om basis.

Implements Section 4: soft-truncated whitening of the kernel-on-landmarks matrix.
"""

from typing import Tuple

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
) -> Tuple[Array, Array, Array, int]:
    """Build the soft-truncated whitening map M_g.

    Given the kernel-on-landmarks matrix W = k(Z,Z), computes:
      1. Eigendecomposition W = U Lambda U^T
      2. Eigenvalue clipping to enforce PSDness
      3. Dataset-scale epsilon
      4. Soft spectral truncation
      5. Retained index set I_g and rank r_g

    Args:
        W: Kernel-on-landmarks matrix of shape (m_g, m_g).
        config: Numerics configuration.

    Returns:
        Tuple of (U, Lambda_clipped, M_g, r_g) where:
            U: eigenvectors of shape (m_g, m_g).
            Lambda_clipped: clipped eigenvalues of shape (m_g,).
            M_g: whitening map of shape (m_g, r_g).
            r_g: retained rank.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")

    # 1. Eigendecomposition
    eigenvalues, U = np.linalg.eigh(W)

    # 2. Clip eigenvalues to enforce numerical PSDness
    clipped = eigenvalue_clip(eigenvalues, min_val=0.0)

    # 3. Dataset-scale epsilon
    trace_w = float(np.sum(clipped))
    m_g = W.shape[0]
    epsilon = compute_epsilon(
        trace_w=trace_w,
        m_g=m_g,
        alpha_epsilon=config.alpha_epsilon,
    )

    # 4. Soft spectral truncation
    truncated = soft_spectral_truncate(
        clipped,
        tau=config.tau_eig,
        epsilon=epsilon,
    )

    # 5. Retained indices
    indices, r_g = retained_indices(clipped, tau=config.tau_eig)

    if r_g == 0:
        # Degenerate case: no components retained
        M_g = np.zeros((m_g, 1), dtype=W.dtype)
        return U, clipped, M_g, 0

    # Build M_g = U_{I_g} diag(tilde_Lambda_{I_g})
    M_g = U[:, indices] @ np.diag(truncated[indices])

    return U, clipped, M_g, r_g


def compute_kernel_on_landmarks(Z: Array, gamma: float = 1.0) -> Array:
    """Compute the RBF kernel matrix on landmarks.

    Args:
        Z: Landmarks of shape (m_g, d).
        gamma: RBF kernel bandwidth parameter (1 / (2 * sigma^2)).

    Returns:
        Kernel matrix W of shape (m_g, m_g).
    """
    sq_dists = (
        np.sum(Z**2, axis=1).reshape(-1, 1)
        + np.sum(Z**2, axis=1).reshape(1, -1)
        - 2.0 * (Z @ Z.T)
    )
    return np.exp(-gamma * sq_dists)
