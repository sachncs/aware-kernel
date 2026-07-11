"""Streamed normal-equations memory accumulator.

Accumulates ``S = Phi^T Phi`` and ``b = Phi^T y`` directly without
storing the full feature matrix.

Memory complexity: O(m^2) where ``m`` is the feature dimension,
independent of the number of samples ``n``.

The streamed mode is preferred when:
* The dataset is too large to fit ``Phi`` in memory.
* You only need the normal equations (not the full feature matrix).
* Memory is a binding constraint.

The normal equations are accumulated incrementally:

    ``S += phi_batch^T @ phi_batch``
    ``b += phi_batch^T @ y_batch``

This is O(batch_size * m^2) per batch, which is more efficient than
recomputing from the full matrix.

Trade-off
---------
The streamed mode discards individual samples, so it cannot reconstruct
the full feature matrix or recompute residuals.  If these capabilities
are needed (e.g., during the refresh pipeline), use the cached mode
instead.
"""

from typing import Any

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.memory.base import BaseMemoryAccumulator


class StreamedMemoryAccumulator(BaseMemoryAccumulator):
    """Accumulator that stores only normal equations.

    Maintains ``S = Phi^T Phi`` and ``b = Phi^T y`` by accumulating
    batch contributions.  The full feature matrix is never stored.

    Attributes:
        S: Accumulated normal matrix of shape ``(m, m)``.
        b: Accumulated normal vector of shape ``(m,)``.
        n: Count of accumulated samples.
    """

    def __init__(
        self,
        feature_dim: int,
        dtype: type[np.floating[Any]] = np.float64,
    ) -> None:
        """Initialize streamed accumulator.

        Args:
            feature_dim: Feature dimension ``m``.
            dtype: Numeric dtype for accumulations.
        """
        super().__init__(feature_dim, dtype)
        self.S = np.zeros((feature_dim, feature_dim), dtype=dtype)
        self.b = np.zeros(feature_dim, dtype=dtype)
        self.n = 0

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch into ``S`` and ``b``.

        Updates the normal equations incrementally without storing
        the batch features.

        Args:
            phi_batch: Feature batch of shape ``(batch_size, m)``.
            y_batch: Target batch of shape ``(batch_size,)``.

        Raises:
            ValueError: If the feature dimension does not match.
        """
        if phi_batch.shape[1] != self.feature_dim:
            raise ValueError(
                f"phi_batch feature dim {phi_batch.shape[1]} != "
                f"expected {self.feature_dim}"
            )

        self.S += phi_batch.T @ phi_batch
        self.b += phi_batch.T @ y_batch
        self.n += phi_batch.shape[0]

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return accumulated normal equations.

        Returns:
            Tuple of ``(S, b)`` where ``S`` has shape ``(m, m)`` and
            ``b`` has shape ``(m,)``.  Defensive copies are returned
            to prevent aliasing.
        """
        return self.S.copy(), self.b.copy()

    def reset(self, feature_dim: int) -> None:
        """Reset accumulator, zeroing all accumulated statistics.

        Args:
            feature_dim: New feature dimension.
        """
        self.feature_dim = feature_dim
        self.S = np.zeros((feature_dim, feature_dim), dtype=self.dtype)
        self.b = np.zeros(feature_dim, dtype=self.dtype)
        self.n = 0

    @property
    def n_accumulated(self) -> int:
        """Number of accumulated samples."""
        return self.n
