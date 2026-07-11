"""Residual-aware anchor selection.

Implements Section 5 of the method blueprint: selecting local corrective
anchors ``A`` by blending coverage weights and residual weights.

The anchor selection procedure determines *where* in the input space the
local corrective basis should focus its capacity.  Two signals are
combined:

* **Coverage weights** (``tilde_l_i``): Proportional to the sum of sparse
  feature activations, favoring regions with many nearby data points.
* **Residual weights** (``tilde_r_i``): Proportional to squared
  residuals from a global-only ridge fit, favoring regions where the
  global basis performs poorly.

The mix is controlled by ``alpha_a``:

    ``p_i = alpha_a * tilde_l_i + (1 - alpha_a) * tilde_r_i``

When ``alpha_a = 0``, anchor selection is purely coverage-based (equivalent
to k-means++).  When ``alpha_a = 1``, it is purely residual-based.  The
default ``alpha_a = 0.5`` balances both objectives.

Algorithm
---------
1. Fit a global-only ridge: ``w_g = (Phi_g^T Phi_g + lambda I)^{-1} Phi_g^T y``.
2. Compute residuals: ``r = y - Phi_g w_g``.
3. Compute coverage weights from sparse feature activations.
4. Compute residual weights from squared residuals.
5. Blend and sample ``m_l`` anchors proportional to the blend.

Complexity
----------
O(n * r_g^2) for the global-only ridge solve, O(n * m_l) for the
coverage weight computation, O(n * m_l) for the anchor sampling.
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_residuals(
    phi_g: Array,
    y: Array,
    lambda_reg: float,
) -> Array:
    """Compute residuals from global-only ridge regression.

    Solves the ridge regression using only global features to obtain
    a baseline predictor, then computes ``r = y - Phi_g w_g``.  These
    residuals identify regions where the global basis is insufficient,
    guiding the local corrective anchor placement.

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        y: Targets of shape ``(n,)``.
        lambda_reg: Ridge regularization parameter.

    Returns:
        Residual vector ``r`` of shape ``(n,)``.
    """
    # Solve (Phi_g^T Phi_g + lambda I) w_g = Phi_g^T y
    s_g = phi_g.T @ phi_g + lambda_reg * np.eye(phi_g.shape[1])
    b_g = phi_g.T @ y
    w_g = np.linalg.solve(s_g, b_g)
    result: Array = y - phi_g @ w_g
    return result


def compute_coverage_weights(s: Array) -> Array:
    """Compute normalized coverage weights ``tilde_l_i``.

    Coverage weights are proportional to the sum of sparse feature
    activations for each data point.  Points with more nearby anchors
    receive higher weight, ensuring the local basis covers densely
    populated regions.

    Args:
        s: Sparse feature matrix of shape ``(n, m_l)``.

    Returns:
        Normalized coverage weights of shape ``(n,)`` summing to 1.
    """
    l_i = np.sum(s, axis=1)
    total = np.sum(l_i)
    if total == 0.0:
        # Uniform fallback when all activations are zero
        fallback: Array = np.ones(s.shape[0]) / s.shape[0]
        return fallback
    result: Array = l_i / total
    return result


def compute_residual_weights(r: Array) -> Array:
    """Compute normalized residual weights ``tilde_r_i``.

    Residual weights are proportional to squared residuals, directing
    the local basis toward regions where the global predictor performs
    poorly.

    Args:
        r: Residuals of shape ``(n,)``.

    Returns:
        Normalized residual weights of shape ``(n,)`` summing to 1.
    """
    r_sq = r**2
    total = np.sum(r_sq)
    if total == 0.0:
        # Uniform fallback when all residuals are zero (perfect fit)
        fallback: Array = np.ones(r.shape[0]) / r.shape[0]
        return fallback
    result: Array = r_sq / total
    return result


def residual_aware_sample(
    embeddings: Array,
    s: Array,
    r: Array,
    alpha_a: float,
    m_l: int,
    rng: np.random.Generator,
) -> Array:
    """Select anchors via residual-aware sampling.

    Blends coverage and residual weights to produce a sampling
    distribution over data points, then draws ``m_l`` anchors without
    replacement.  This ensures anchors are placed both in densely
    populated regions (coverage) and in regions with large prediction
    errors (residuals).

    Args:
        embeddings: Normalized embeddings of shape ``(n, d)``.
        s: Sparse feature matrix of shape ``(n, m_l_candidate)``.
        r: Residuals of shape ``(n,)``.
        alpha_a: Mix weight between coverage and residual.  ``0.0``
            means pure coverage, ``1.0`` means pure residual.
        m_l: Number of anchors to select.
        rng: Random generator for reproducibility.

    Returns:
        Selected anchors of shape ``(m_l, d)``.

    Raises:
        ValueError: If ``m_l > n``.
    """
    n = embeddings.shape[0]
    tilde_l = compute_coverage_weights(s)
    tilde_r = compute_residual_weights(r)
    p = alpha_a * tilde_l + (1.0 - alpha_a) * tilde_r

    # Ensure valid probability distribution (non-negative, sums to 1)
    p = np.maximum(p, 0.0)
    total_p = np.sum(p)
    p = np.ones(n) / n if total_p <= 0.0 else p / total_p

    indices = rng.choice(n, size=m_l, replace=False, p=p)
    result: Array = embeddings[indices]
    return result
