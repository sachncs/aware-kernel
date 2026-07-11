"""Training objectives for the bilevel outer loop.

Implements Section 10 of the method blueprint: the outer objective that
is optimized with respect to the projection matrix ``R``.

The full outer objective is:

    ``L_outer = ||y - Phi w||^2 + lambda_R ||R||_F^2``
    ``       + lambda_orth ||R^T R - I||_F^2``
    ``       + gamma_div D(Phi_g, Phi_l_perp)``

where:

* ``||y - Phi w||^2`` is the ridge prediction loss.
* ``lambda_R ||R||_F^2`` is a Frobenius regularizer encouraging
  small-norm projections.
* ``lambda_orth ||R^T R - I||_F^2`` penalizes deviation from
  orthogonality.
* ``gamma_div D(Phi_g, Phi_l_perp)`` encourages global and local
  features to span complementary subspaces.

Design rationale
----------------
The orthogonality penalty encourages ``R`` to be a rotation/reflection,
which preserves the norm of projected embeddings and maintains the
numerical stability of the whitening step.  The diversity penalty
prevents the global and local features from collapsing onto the same
subspace, which would make the fusion gate ineffective.
"""

import numpy as np

from aware_kernel.aware.types import Array


def ridge_prediction_loss(y: Array, phi: Array, w: Array) -> float:
    """Compute ridge prediction loss ``||y - Phi w||_2^2``.

    This is the unregularized prediction error on the training data.

    Args:
        y: Targets of shape ``(n,)``.
        phi: Fused features of shape ``(n, m)``.
        w: Coefficients of shape ``(m,)``.

    Returns:
        Sum of squared residuals.
    """
    residuals = y - phi @ w
    return float(np.sum(residuals**2))


def orthogonality_penalty(R: Array) -> float:
    """Compute orthogonality penalty ``||R^T R - I||_F^2``.

    Measures how far ``R`` is from being an orthogonal matrix.  The
    penalty is zero when ``R`` is exactly orthogonal and increases
    quadratically with deviation.

    Args:
        R: Projection matrix of shape ``(d, d)``.

    Returns:
        Orthogonality penalty (non-negative).
    """
    d = R.shape[0]
    return float(np.sum((R.T @ R - np.eye(d))**2))


def diversity_penalty(
    phi_g: Array,
    phi_l_perp: Array,
) -> float:
    """Compute diversity penalty ``D(Phi_g, Phi_l_perp)``.

    The diversity penalty measures the overlap between global and local
    feature subspaces:

        ``D = ||Phi_g^T Phi_l_perp||_F^2 / (tr(Phi_g^T Phi_g) * tr(Phi_l_perp^T Phi_l_perp))``

    This is normalized by the traces to be scale-invariant.  The penalty
    is zero when the subspaces are orthogonal and increases as they
    become more aligned.

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        phi_l_perp: Orthogonalized local features of shape ``(n, m_l)``.

    Returns:
        Diversity penalty (non-negative, scale-invariant).
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

    Combines all four terms of the outer objective into a single scalar
    that is minimized by the outer-loop optimizer.

    Args:
        y: Targets of shape ``(n,)``.
        phi: Fused features of shape ``(n, m)``.
        w: Ridge coefficients of shape ``(m,)``.
        R: Projection matrix of shape ``(d, d)``.
        phi_g: Global features of shape ``(n, r_g)``.
        phi_l_perp: Orthogonalized local features of shape ``(n, m_l)``.
        lambda_r: Weight for Frobenius norm of ``R``.
        lambda_orth: Weight for orthogonality penalty.
        gamma_div: Weight for diversity penalty.

    Returns:
        Scalar outer objective value.
    """
    loss = ridge_prediction_loss(y, phi, w)
    reg_r = lambda_r * float(np.sum(R**2))
    reg_orth = lambda_orth * orthogonality_penalty(R)
    div = gamma_div * diversity_penalty(phi_g, phi_l_perp)
    return loss + reg_r + reg_orth + div
