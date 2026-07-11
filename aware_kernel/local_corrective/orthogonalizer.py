"""Residual orthogonalization against the global subspace.

Implements Section 6 of the method blueprint: projecting local features
into the nullspace of the global feature subspace.

The orthogonalization step ensures that local corrective features do not
redundantly encode information already captured by the global basis.  This
is achieved by computing the projection matrix onto the global subspace
and subtracting it from the local features:

    ``P_g = Phi_g (Phi_g^T Phi_g + eta_o I)^{-1} Phi_g^T``
    ``Phi_l_perp = (I - P_g) Phi_l``

The ridge term ``eta_o`` prevents singularity when ``Phi_g`` is
rank-deficient (which is common when ``r_g < n``).

Design rationale
----------------
Without orthogonalization, the local and global features would span
overlapping subspaces, making the fusion gate ``rho`` ineffective and
reducing the diversity of the feature representation.  Orthogonalization
ensures that adding local features always increases the total feature
rank, which is a key theoretical property of the method.

Complexity
----------
O(n * r_g^2) for the Gram matrix solve, O(n^2 * r_g) for the projection
computation.  When ``n >> r_g``, the bottleneck is the projection step.
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_orthogonalization_matrix(
    phi_g: Array,
    eta_o: float,
) -> Array:
    """Compute the projection matrix ``P_g`` onto the global subspace.

    Computes the ridge-regularized orthogonal projection:

        ``P_g = Phi_g (Phi_g^T Phi_g + eta_o I)^{-1} Phi_g^T``

    The ridge term ``eta_o`` ensures the inner matrix is invertible even
    when ``Phi_g`` is rank-deficient.  The projection is symmetric and
    idempotent (up to the ridge regularization).

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        eta_o: Ridge regularizer for projection.  Controls the
            trade-off between exact projection and numerical stability.

    Returns:
        Projection matrix ``P_g`` of shape ``(n, n)``.
    """
    gram = phi_g.T @ phi_g + eta_o * np.eye(phi_g.shape[1])
    # Solve (Phi_g^T Phi_g + eta_o I)^{-1} Phi_g^T without forming
    # the full inverse, which would be O(n^3) and numerically unstable.
    inv_gram_phi_g_t = np.linalg.solve(gram, phi_g.T)
    result: Array = phi_g @ inv_gram_phi_g_t
    return result


def orthogonalize_local_features(
    phi_g: Array,
    phi_l: Array,
    eta_o: float,
) -> Array:
    """Project local features orthogonal to the global subspace.

    Computes ``Phi_l_perp = (I - P_g) Phi_l`` where ``P_g`` is the
    ridge-regularized projection onto the global feature subspace.

    The result is a set of local features that carry only the information
    *not* already captured by the global basis, ensuring maximum feature
    diversity.

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        phi_l: Local features of shape ``(n, m_l)``.
        eta_o: Ridge regularizer for projection.

    Returns:
        Orthogonalized local features ``Phi_l_perp`` of shape
        ``(n, m_l)``.
    """
    P_g = compute_orthogonalization_matrix(phi_g, eta_o)
    result: Array = phi_l - P_g @ phi_l
    return result


def check_orthogonality(
    phi_g: Array,
    phi_l_perp: Array,
    tol: float = 1e-6,
) -> bool:
    """Check that ``Phi_g^T Phi_l_perp`` is near zero.

    This is a diagnostic function used in numerical tests to verify that
    the orthogonalization step is working correctly.  The Frobenius norm
    of the cross-covariance matrix should be small relative to the
    feature norms.

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        phi_l_perp: Orthogonalized local features of shape ``(n, m_l)``.
        tol: Tolerance for the Frobenius norm.  Default ``1e-6``.

    Returns:
        ``True`` if ``||Phi_g^T Phi_l_perp||_F < tol``.
    """
    cross = phi_g.T @ phi_l_perp
    norm = float(np.linalg.norm(cross, "fro"))
    return norm < tol
