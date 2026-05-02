"""Residual orthogonalization against global subspace.

Implements Section 6:
    P_g = Phi_g (Phi_g^T Phi_g + eta_o I)^{-1} Phi_g^T
    Phi_l_perp = (I - P_g) Phi_l
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_orthogonalization_matrix(
    phi_g: Array,
    eta_o: float,
) -> Array:
    """Compute the projection matrix P_g onto the global subspace.

    Args:
        phi_g: Global features of shape (n, r_g).
        eta_o: Ridge regularizer for projection.

    Returns:
        Projection matrix P_g of shape (n, n).
    """
    gram = phi_g.T @ phi_g + eta_o * np.eye(phi_g.shape[1])
    # Compute (Phi_g^T Phi_g + eta_o I)^{-1} Phi_g^T
    inv_gram_phi_g_t = np.linalg.solve(gram, phi_g.T)
    return phi_g @ inv_gram_phi_g_t


def orthogonalize_local_features(
    phi_g: Array,
    phi_l: Array,
    eta_o: float,
) -> Array:
    """Project local features orthogonal to the global subspace.

    Args:
        phi_g: Global features of shape (n, r_g).
        phi_l: Local features of shape (n, m_l).
        eta_o: Ridge regularizer for projection.

    Returns:
        Orthogonalized local features Phi_l_perp of shape (n, m_l).
    """
    P_g = compute_orthogonalization_matrix(phi_g, eta_o)
    return phi_l - P_g @ phi_l


def check_orthogonality(
    phi_g: Array,
    phi_l_perp: Array,
    tol: float = 1e-6,
) -> bool:
    """Check that Phi_g^T Phi_l_perp is near zero.

    Args:
        phi_g: Global features of shape (n, r_g).
        phi_l_perp: Orthogonalized local features of shape (n, m_l).
        tol: Tolerance for Frobenius norm.

    Returns:
        True if orthogonal within tolerance.
    """
    cross = phi_g.T @ phi_l_perp
    norm = float(np.linalg.norm(cross, "fro"))
    return norm < tol
