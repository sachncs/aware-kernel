"""Sparse local corrective features.

Implements Section 5:
    s_j(x) = exp(-||bar_e(x) - a_j||^2 / (2 tau_l^2)) * 1[j in N_{k_l}(x; A)]
    d_j = sum_i s_j(x_i)^2
    phi_l(x)_j = s_j(x) / sqrt(d_j + eta)
"""

from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from aware_kernel.aware.types import Array


def compute_sparse_features(
    embeddings: Array,
    anchors: Array,
    tau: float,
    k: int,
) -> Array:
    """Compute sparse radial features for local corrective basis.

    For each point, only the k nearest anchors get non-zero weights.

    Args:
        embeddings: Normalized embeddings of shape (n, d).
        anchors: Anchor points of shape (m_l, d).
        tau: Bandwidth parameter for RBF kernel.
        k: Number of nearest neighbors (must be <= m_l).

    Returns:
        Sparse feature matrix of shape (n, m_l).
    """
    n, d = embeddings.shape
    m_l = anchors.shape[0]

    if k > m_l:
        raise ValueError(f"k ({k}) cannot exceed m_l ({m_l})")

    # Build KD-tree for efficient k-NN
    tree = KDTree(anchors)
    distances, indices = tree.query(embeddings, k=k)

    # Compute RBF weights
    # distances shape: (n, k)
    # indices shape: (n, k)
    weights = np.exp(-(distances**2) / (2.0 * tau**2))

    # Scatter into sparse feature matrix
    s = np.zeros((n, m_l), dtype=embeddings.dtype)
    for i in range(n):
        s[i, indices[i]] = weights[i]

    return s


def compute_local_normalizers(s: Array, eta: float = 1e-8) -> Array:
    """Compute per-anchor normalization denominators d_j.

    Args:
        s: Sparse feature matrix of shape (n, m_l).
        eta: Stabilization constant.

    Returns:
        Normalization vector d of shape (m_l,).
    """
    return np.sum(s**2, axis=0) + eta


def build_local_features(
    embeddings: Array,
    anchors: Array,
    tau: float,
    k: int,
    eta: float = 1e-8,
) -> tuple[Array, Array]:
    """Build normalized local features and their normalizers.

    Args:
        embeddings: Normalized embeddings of shape (n, d).
        anchors: Anchor points of shape (m_l, d).
        tau: Bandwidth parameter.
        k: Number of nearest neighbors.
        eta: Stabilization constant.

    Returns:
        Tuple of (phi_l, d) where phi_l has shape (n, m_l) and d has shape (m_l,).
    """
    s = compute_sparse_features(embeddings, anchors, tau, k)
    d = compute_local_normalizers(s, eta)
    phi_l = s / np.sqrt(d)
    return phi_l, d
