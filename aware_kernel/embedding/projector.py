"""Projection module: normalization and R-projection.

Implements Section 3:
    e = f_theta(x)
    bar_e = e / (||e||_2 + delta)
    u = R * bar_e
"""

import numpy as np

from aware_kernel.aware.types import Array


def normalize_embeddings(
    embeddings: Array,
    delta: float = 1e-8,
    axis: int = -1,
) -> Array:
    """Normalize embeddings with a stabilization term delta.

    Args:
        embeddings: Array of shape (..., d).
        delta: Small constant to avoid division by zero.
        axis: Axis along which to compute the L2 norm.

    Returns:
        Normalized embeddings with same shape as input.
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    return embeddings / (norms + delta)


def project_embeddings(
    normalized_embeddings: Array,
    R: Array,
) -> Array:
    """Project normalized embeddings through matrix R.

    Args:
        normalized_embeddings: Array of shape (n, embedding_dim).
        R: Projection matrix of shape (embedding_dim, embedding_dim).

    Returns:
        Projected embeddings of shape (n, embedding_dim).
    """
    if normalized_embeddings.ndim == 1:
        return R @ normalized_embeddings
    return normalized_embeddings @ R.T


class Projector:
    """Stateful projector combining normalization and R-projection."""

    def __init__(self, R: Array, delta: float = 1e-8) -> None:
        """Initialize projector.

        Args:
            R: Projection matrix of shape (d, d).
            delta: Normalization stabilization constant.
        """
        self._R = R.copy()
        self._delta = delta

    @property
    def R(self) -> Array:
        """Current projection matrix (copy)."""
        return self._R.copy()

    @R.setter
    def R(self, value: Array) -> None:
        """Update projection matrix."""
        self._R = value.copy()

    def transform(self, embeddings: Array) -> Array:
        """Normalize and project embeddings.

        Args:
            embeddings: Array of shape (n, d) or (d,).

        Returns:
            Projected embeddings of same batch shape.
        """
        normalized = normalize_embeddings(embeddings, delta=self._delta)
        return project_embeddings(normalized, self._R)
