"""Sparse local corrective features.

Implements Section 5 of the method blueprint: constructing k-NN sparse
radial basis function features for the local corrective basis.

The sparse feature map is:

    ``s_j(x) = exp(-||bar_e(x) - a_j||^2 / (2 tau_l^2)) * 1[j in N_{k_l}(x; A)]``

Only the ``k_local`` nearest anchors to each data point receive non-zero
weights, producing a sparse feature matrix.  This sparsity is crucial for
efficiency: the local basis has ``m_l`` anchors, but each data point
activates at most ``k_local`` of them.

After computing the raw sparse activations, per-anchor normalizers are
computed:

    ``d_j = sum_i s_j(x_i)^2 + eta``

and the normalized features are:

    ``phi_l(x)_j = s_j(x) / sqrt(d_j)``

The normalization ensures unit-variance local features, which is
important for the trace-based calibration in the fusion stage.

Algorithm
---------
1. Build a KD-tree from the anchor set ``A``.
2. For each data point, query the ``k`` nearest anchors.
3. Compute RBF weights for the ``k`` neighbors.
4. Scatter weights into a sparse feature matrix.
5. Compute per-anchor normalizers ``d_j``.
6. Normalize: ``phi_l = s / sqrt(d)``.

Complexity
----------
O(n * k * d) for the KD-tree query, O(n * m_l) for the sparse matrix
construction.  The KD-tree reduces the per-point cost from O(m_l * d)
to O(k * log(m_l) * d).
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
    """Compute sparse radial features for the local corrective basis.

    For each data point, only the ``k`` nearest anchors get non-zero
    weights.  The weights are RBF kernel values:

        ``w_ij = exp(-||e_i - a_j||^2 / (2 tau^2))``

    for the ``k`` nearest anchors, and zero elsewhere.

    Uses a KD-tree for efficient nearest-neighbor queries, reducing the
    per-point cost from O(m_l * d) to O(k * log(m_l) * d).

    Args:
        embeddings: Normalized embeddings of shape ``(n, d)``.
        anchors: Anchor points of shape ``(m_l, d)``.
        tau: Bandwidth parameter for the RBF kernel.  Smaller values
            produce sparser features.
        k: Number of nearest neighbors.  Must be ``<= m_l``.

    Returns:
        Sparse feature matrix of shape ``(n, m_l)``.

    Raises:
        ValueError: If ``k > m_l``.
    """
    n, d = embeddings.shape
    m_l = anchors.shape[0]

    if k > m_l:
        raise ValueError(f"k ({k}) cannot exceed m_l ({m_l})")

    # Build KD-tree for efficient k-NN queries.
    # The KD-tree provides O(log(m_l)) per query instead of O(m_l),
    # which is critical when m_l is large.
    tree = KDTree(anchors)
    distances, indices = tree.query(embeddings, k=k)

    # Compute RBF weights from the k-NN distances.
    # The bandwidth tau controls the locality: smaller tau means only
    # very close anchors contribute significantly.
    weights = np.exp(-(distances**2) / (2.0 * tau**2))

    # Scatter into a sparse feature matrix of shape (n, m_l).
    # Most entries are zero; only the k nearest anchors per point are
    # populated.
    s = np.zeros((n, m_l), dtype=embeddings.dtype)
    for i in range(n):
        s[i, indices[i]] = weights[i]

    return s


def compute_local_normalizers(s: Array, eta: float = 1e-8) -> Array:
    """Compute per-anchor normalization denominators ``d_j``.

    The normalizer ``d_j = sum_i s_j(x_i)^2 + eta`` ensures that
    each local feature has approximately unit variance across the
    dataset.  The stabilization constant ``eta`` prevents division by
    zero for anchors with no nearby data points.

    Args:
        s: Sparse feature matrix of shape ``(n, m_l)``.
        eta: Stabilization constant.  Default ``1e-8``.

    Returns:
        Normalization vector ``d`` of shape ``(m_l,)``.
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

    Convenience function that computes the sparse features, normalizers,
    and normalized features in one call.  This is the main entry point
    used by the refresh pipeline.

    Args:
        embeddings: Normalized embeddings of shape ``(n, d)``.
        anchors: Anchor points of shape ``(m_l, d)``.
        tau: Bandwidth parameter for the RBF kernel.
        k: Number of nearest neighbors.  Must be ``<= m_l``.
        eta: Stabilization constant for normalizers.

    Returns:
        Tuple of ``(phi_l, d)`` where ``phi_l`` has shape ``(n, m_l)``
        and ``d`` has shape ``(m_l,)``.
    """
    s = compute_sparse_features(embeddings, anchors, tau, k)
    d = compute_local_normalizers(s, eta)
    phi_l = s / np.sqrt(d)
    return phi_l, d
