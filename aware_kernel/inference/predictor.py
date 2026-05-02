"""Inference utilities for mean and variance prediction.

Implements Section 12:
    mu_*(x_*) = phi(x_*)^T w*
    sigma_*^2(x_*) = phi(x_*)^T phi(x_*) - phi(x_*)^T (Phi^T Phi + lambda I)^{-1} phi(x_*)
"""

import numpy as np

from aware_kernel.aware.types import Array


def predict_mean(phi_query: Array, w: Array) -> float | Array:
    """Predict mean for query point(s).

    Args:
        phi_query: Fused features of shape (m,) or (n_query, m).
        w: Ridge coefficients of shape (m,).

    Returns:
        Mean prediction(s).
    """
    if phi_query.ndim == 1:
        return float(phi_query @ w)
    return phi_query @ w


def predict_variance(
    phi_query: Array,
    s_inv: Array,
) -> float | Array:
    """Predict variance for query point(s).

    Args:
        phi_query: Fused features of shape (m,) or (n_query, m).
        s_inv: Inverse of (Phi^T Phi + lambda I) of shape (m, m).

    Returns:
        Variance prediction(s), clamped at zero.
    """
    if phi_query.ndim == 1:
        self_term = float(phi_query @ phi_query)
        correction = float(phi_query @ s_inv @ phi_query)
        var = max(self_term - correction, 0.0)
        return var

    # Batch prediction
    self_terms = np.sum(phi_query * phi_query, axis=1)
    corrections = np.einsum("ij,jk,ik->i", phi_query, s_inv, phi_query)
    variances = np.maximum(self_terms - corrections, 0.0)
    return variances


class Predictor:
    """Stateful predictor for mean and variance."""

    def __init__(self, w: Array, s_inv: Array | None = None) -> None:
        """Initialize predictor.

        Args:
            w: Ridge coefficients of shape (m,).
            s_inv: Optional inverse normal matrix for variance.
        """
        self._w = w
        self._s_inv = s_inv

    @property
    def w(self) -> Array:
        """Ridge coefficients."""
        return self._w

    @w.setter
    def w(self, value: Array) -> None:
        """Update ridge coefficients."""
        self._w = value

    @property
    def s_inv(self) -> Array | None:
        """Inverse normal matrix."""
        return self._s_inv

    @s_inv.setter
    def s_inv(self, value: Array | None) -> None:
        """Update inverse normal matrix."""
        self._s_inv = value

    def predict(self, phi_query: Array) -> float | Array:
        """Predict mean for query point(s).

        Args:
            phi_query: Fused features.

        Returns:
            Mean prediction(s).
        """
        return predict_mean(phi_query, self._w)

    def predict_variance(self, phi_query: Array) -> float | Array:
        """Predict variance for query point(s).

        Args:
            phi_query: Fused features.

        Returns:
            Variance prediction(s).

        Raises:
            ValueError: If s_inv is not set.
        """
        if self._s_inv is None:
            raise ValueError("s_inv must be set for variance prediction")
        return predict_variance(phi_query, self._s_inv)
