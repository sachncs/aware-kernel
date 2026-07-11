"""Projection module: normalization and R-projection.

Implements Stage 2 of the aware-kernel pipeline (Section 3):

    ``e = f_theta(x)``
    ``bar_e = e / (||e||_2 + delta)``
    ``u = R * bar_e``

The projection stage converts raw embeddings ``e`` into a normalized,
projected representation ``u`` that serves as input to the Nyström global
basis and local corrective stages.

The L2 normalization with stabilization constant ``delta`` ensures that
embeddings of different magnitudes are treated uniformly, preventing
outliers from dominating the kernel computations.  The learned projection
matrix ``R`` is optimized via the outer-loop objective to improve
downstream ridge regression performance.

Dependencies
------------
* ``aware_kernel.aware.types`` -- ``Array`` type alias.
"""

import numpy as np

from aware_kernel.aware.types import Array


def normalize_embeddings(
    embeddings: Array,
    delta: float = 1e-8,
    axis: int = -1,
) -> Array:
    """Normalize embeddings with a stabilization term delta.

    Computes ``bar_e = e / (||e||_2 + delta)`` along the specified axis.
    The stabilization constant ``delta`` prevents division by zero for
    zero-norm embeddings and adds numerical stability.

    The normalization is applied independently to each sample, so the
    output vectors lie on or near the unit hypersphere.

    Args:
        embeddings: Array of shape ``(..., d)``.
        delta: Small constant to avoid division by zero.  Default
            ``1e-8`` is sufficient for float64 inputs.
        axis: Axis along which to compute the L2 norm.  Default ``-1``
            (last axis).

    Returns:
        Normalized embeddings with same shape as input.
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    result: Array = embeddings / (norms + delta)
    return result


def project_embeddings(
    normalized_embeddings: Array,
    R: Array,
) -> Array:
    """Project normalized embeddings through matrix R.

    Computes ``u = bar_e @ R^T`` for batch inputs or ``u = R @ bar_e``
    for single vectors.  The projection matrix ``R`` is learned via the
    outer-loop optimizer and is encouraged (but not required) to be
    orthogonal.

    Args:
        normalized_embeddings: Array of shape ``(n, embedding_dim)`` or
            ``(embedding_dim,)`` for a single sample.
        R: Projection matrix of shape ``(embedding_dim, embedding_dim)``.

    Returns:
        Projected embeddings of shape ``(n, embedding_dim)`` or
        ``(embedding_dim,)``.
    """
    if normalized_embeddings.ndim == 1:
        return R @ normalized_embeddings
    return normalized_embeddings @ R.T


class Projector:
    """Stateful projector combining normalization and R-projection.

    Encapsulates the two-step projection pipeline:

    1. L2 normalization with stabilization.
    2. Linear projection through the learned matrix ``R``.

    The ``Projector`` is used by both the training loop (to project
    embeddings for basis construction) and the inference path (to project
    new inputs for prediction).  It stores a copy of ``R`` to avoid
    aliasing with the mutable state in ``ContinuousState``.

    Attributes:
        R: Internal projection matrix (stored as a copy).
        delta: Normalization stabilization constant.
    """

    def __init__(self, R: Array, delta: float = 1e-8) -> None:
        """Initialize projector.

        Args:
            R: Projection matrix of shape ``(d, d)``.  A copy is stored
                internally to avoid aliasing.
            delta: Normalization stabilization constant.  Default
                ``1e-8``.
        """
        self._R = R.copy()
        self.delta = delta

    @property
    def R(self) -> Array:
        """Current projection matrix (defensive copy)."""
        return self._R.copy()

    @R.setter
    def R(self, value: Array) -> None:
        """Update projection matrix.

        Args:
            value: New projection matrix.  A copy is stored internally.
        """
        self._R = value.copy()

    def transform(self, embeddings: Array) -> Array:
        """Normalize and project embeddings.

        Applies the full two-step pipeline: L2 normalization followed by
        linear projection through ``R``.

        Args:
            embeddings: Array of shape ``(n, d)`` or ``(d,)`` for a
                single sample.

        Returns:
            Projected embeddings of same batch shape.
        """
        normalized = normalize_embeddings(embeddings, delta=self.delta)
        return project_embeddings(normalized, self._R)
