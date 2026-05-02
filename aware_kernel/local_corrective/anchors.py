"""Residual-aware anchor selection.

Implements Section 5:
    Global-only ridge: w_g = (Phi_g^T Phi_g + lambda I)^{-1} Phi_g^T y
    r = y - Phi_g w_g
    l_i = sum_j s_j(x_i)
    tilde_l_i = l_i / sum_q l_q
    tilde_r_i = r_i^2 / sum_q r_q^2
    p_i propto alpha_a tilde_l_i + (1 - alpha_a) tilde_r_i
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_residuals(
    phi_g: Array,
    y: Array,
    lambda_reg: float,
) -> Array:
    """Compute residuals from global-only ridge regression.

    Args:
        phi_g: Global features of shape (n, r_g).
        y: Targets of shape (n,).
        lambda_reg: Ridge regularization.

    Returns:
        Residual vector r of shape (n,).
    """
    # Solve (Phi_g^T Phi_g + lambda I) w_g = Phi_g^T y
    s_g = phi_g.T @ phi_g + lambda_reg * np.eye(phi_g.shape[1])
    b_g = phi_g.T @ y
    w_g = np.linalg.solve(s_g, b_g)
    return y - phi_g @ w_g


def compute_coverage_weights(s: Array) -> Array:
    """Compute normalized coverage weights tilde_l_i.

    Args:
        s: Sparse feature matrix of shape (n, m_l).

    Returns:
        Normalized coverage weights of shape (n,).
    """
    l_i = np.sum(s, axis=1)
    total = np.sum(l_i)
    if total == 0.0:
        return np.ones(s.shape[0]) / s.shape[0]
    return l_i / total


def compute_residual_weights(r: Array) -> Array:
    """Compute normalized residual weights tilde_r_i.

    Args:
        r: Residuals of shape (n,).

    Returns:
        Normalized residual weights of shape (n,).
    """
    r_sq = r**2
    total = np.sum(r_sq)
    if total == 0.0:
        return np.ones(r.shape[0]) / r.shape[0]
    return r_sq / total


def residual_aware_sample(
    embeddings: Array,
    s: Array,
    r: Array,
    alpha_a: float,
    m_l: int,
    rng: np.random.Generator,
) -> Array:
    """Select anchors via residual-aware sampling.

    Args:
        embeddings: Normalized embeddings of shape (n, d).
        s: Sparse feature matrix of shape (n, m_l_candidate).
        r: Residuals of shape (n,).
        alpha_a: Mix weight between coverage and residual (0=coverage, 1=residual).
        m_l: Number of anchors to select.
        rng: Random generator.

    Returns:
        Selected anchors of shape (m_l, d).
    """
    n = embeddings.shape[0]
    tilde_l = compute_coverage_weights(s)
    tilde_r = compute_residual_weights(r)
    p = alpha_a * tilde_l + (1.0 - alpha_a) * tilde_r

    # Ensure valid probability distribution
    p = np.maximum(p, 0.0)
    total_p = np.sum(p)
    if total_p <= 0.0:
        p = np.ones(n) / n
    else:
        p = p / total_p

    indices = rng.choice(n, size=m_l, replace=False, p=p)
    return embeddings[indices]
