"""Inference utilities for mean and variance prediction.

Implements Section 12 of the method blueprint: computing predictive
means and variances from the learned feature representation.

The predictive equations are:

    ``mu_*(x_*) = phi(x_*)^T w*``
    ``sigma_*^2(x_*) = phi(x_*)^T phi(x_*) - phi(x_*)^T (Phi^T Phi + lambda I)^{-1} phi(x_*)``

The mean prediction is a simple inner product between the fused features
and the ridge coefficients.  The variance prediction additionally
requires the inverse normal matrix ``S^{-1}``, which captures the
epistemic uncertainty of the ridge estimate.

Design rationale
----------------
The variance prediction uses the standard Bayesian linear regression
posterior variance formula.  The clamping at zero ensures non-negative
variances despite numerical errors.

The predictor is split into stateless functions (``predict_mean``,
``predict_variance``) and a stateful wrapper (``Predictor``) for
convenience.  The stateless functions can be used independently in
evaluation pipelines.
"""

import numpy as np

from aware_kernel.aware.types import Array


def predict_mean(phi_query: Array, w: Array) -> float | Array:
    """Predict mean for query point(s).

    Computes ``mu_*(x_*) = phi(x_*)^T w*``.

    Args:
        phi_query: Fused features of shape ``(m,)`` for a single query
            or ``(n_query, m)`` for a batch.
        w: Ridge coefficients of shape ``(m,)``.

    Returns:
        Mean prediction(s): scalar for single query, array for batch.
    """
    if phi_query.ndim == 1:
        return float(phi_query @ w)
    return phi_query @ w


def predict_variance(
    phi_query: Array,
    s_inv: Array,
) -> float | Array:
    """Predict variance for query point(s).

    Computes the posterior variance from Bayesian linear regression:

        ``sigma^2(x_*) = phi(x_*)^T phi(x_*) - phi(x_*)^T S^{-1} phi(x_*)``

    where ``S = Phi^T Phi + lambda I`` is the regularized normal matrix.

    The variance is clamped at zero to handle numerical errors that
    could produce small negative values.

    Args:
        phi_query: Fused features of shape ``(m,)`` or ``(n_query, m)``.
        s_inv: Inverse of ``(Phi^T Phi + lambda I)`` of shape ``(m, m)``.

    Returns:
        Variance prediction(s), clamped at zero.
    """
    if phi_query.ndim == 1:
        self_term = float(phi_query @ phi_query)
        correction = float(phi_query @ s_inv @ phi_query)
        var = max(self_term - correction, 0.0)
        return var

    # Batch prediction using Einstein summation for efficiency.
    self_terms = np.sum(phi_query * phi_query, axis=1)
    corrections = np.einsum("ij,jk,ik->i", phi_query, s_inv, phi_query)
    variances = np.maximum(self_terms - corrections, 0.0)
    return variances


class Predictor:
    """Stateful predictor for mean and variance.

    Wraps the ridge coefficients ``w`` and optionally the inverse normal
    matrix ``S^{-1}`` for prediction.  The mean prediction is always
    available; the variance prediction requires ``S^{-1}``.

    Attributes:
        w: Ridge coefficients of shape ``(m,)``.
        s_inv: Inverse normal matrix of shape ``(m, m)`` or ``None``.
    """

    def __init__(self, w: Array, s_inv: Array | None = None) -> None:
        """Initialize predictor.

        Args:
            w: Ridge coefficients of shape ``(m,)``.
            s_inv: Optional inverse normal matrix of shape ``(m, m)``
                for variance prediction.  If ``None``, only mean
                prediction is available.
        """
        self.w = w
        self.s_inv = s_inv

    def predict(self, phi_query: Array) -> float | Array:
        """Predict mean for query point(s).

        Args:
            phi_query: Fused features of shape ``(m,)`` or
                ``(n_query, m)``.

        Returns:
            Mean prediction(s).
        """
        return predict_mean(phi_query, self.w)

    def predict_variance(self, phi_query: Array) -> float | Array:
        """Predict variance for query point(s).

        Args:
            phi_query: Fused features of shape ``(m,)`` or
                ``(n_query, m)``.

        Returns:
            Variance prediction(s).

        Raises:
            ValueError: If ``s_inv`` has not been set.
        """
        if self.s_inv is None:
            raise ValueError("s_inv must be set for variance prediction")
        return predict_variance(phi_query, self.s_inv)
