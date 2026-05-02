"""Training objectives for the bilevel outer loop.

Implements Section 10:
    L_outer + lambda_R ||R||_F^2 + lambda_orth ||R^T R - I||_F^2 + gamma_div D(Phi_g, Phi_l_perp)
"""

import numpy as np

from aware_kernel.aware.types import Array


def ridge_prediction_loss(y: Array, phi: Array, w: Array) -> float:
    """Compute ridge prediction loss ||y - Phi w||_2^2.

    Args:
        y: Targets of shape (n,).
        phi: Fused features of shape (n, m).
        w: Coefficients of shape (m,).

    Returns:
        Prediction loss.
    """
    residuals = y - phi @ w
    return float(np.sum(residuals**2))


def orthogonality_penalty(R: Array) -> float:
    """Compute orthogonality penalty ||R^T R - I||_F^2.

    Args:
        R: Projection matrix of shape (d, d).

    Returns:
        Orthogonality penalty.
    """
    d = R.shape[0]
    return float(np.sum((R.T @ R - np.eye(d))**2))


def diversity_penalty(
    phi_g: Array,
    phi_l_perp: Array,
) -> float:
    """Compute diversity penalty D(Phi_g, Phi_l_perp).

    D = ||Phi_g^T Phi_l_perp||_F^2 / (tr(Phi_g^T Phi_g) * tr((Phi_l_perp)^T Phi_l_perp))

    Args:
        phi_g: Global features of shape (n, r_g).
        phi_l_perp: Orthogonalized local features of shape (n, m_l).

    Returns:
        Diversity penalty.
    """
    cross = phi_g.T @ phi_l_perp
    num = float(np.sum(cross**2))
    denom = float(np.trace(phi_g.T @ phi_g) * np.trace(phi_l_perp.T @ phi_l_perp))
    if denom == 0.0:
        return 0.0
    return num / denom


def compute_outer_objective(
    y: Array,
    phi: Array,
    w: Array,
    R: Array,
    phi_g: Array,
    phi_l_perp: Array,
    lambda_r: float = 0.0,
    lambda_orth: float = 0.0,
    gamma_div: float = 0.0,
) -> float:
    """Compute the full outer objective.

    Args:
        y: Targets.
        phi: Fused features.
        w: Ridge coefficients.
        R: Projection matrix.
        phi_g: Global features.
        phi_l_perp: Orthogonalized local features.
        lambda_r: Weight for Frobenius norm of R.
        lambda_orth: Weight for orthogonality penalty.
        gamma_div: Weight for diversity penalty.

    Returns:
        Scalar outer objective.
    """
    loss = ridge_prediction_loss(y, phi, w)
    reg_r = lambda_r * float(np.sum(R**2))
    reg_orth = lambda_orth * orthogonality_penalty(R)
    div = gamma_div * diversity_penalty(phi_g, phi_l_perp)
    return loss + reg_r + reg_orth + div
