"""Streamed normal-equations memory accumulator.

Accumulates S = Phi^T Phi and b = Phi^T y directly (memory O(m^2)).
"""

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.memory.base import BaseMemoryAccumulator


class StreamedMemoryAccumulator(BaseMemoryAccumulator):
    """Accumulator that stores only normal equations.

    Attributes:
        S: Accumulated normal matrix of shape (m, m).
        b: Accumulated normal vector of shape (m,).
        n: Count of accumulated samples.
    """

    def __init__(self, feature_dim: int, dtype: np.dtype = np.float64) -> None:
        """Initialize streamed accumulator.

        Args:
            feature_dim: Feature dimension m.
            dtype: Numeric dtype.
        """
        super().__init__(feature_dim, dtype)
        self._S = np.zeros((feature_dim, feature_dim), dtype=dtype)
        self._b = np.zeros(feature_dim, dtype=dtype)
        self._n = 0

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch into S and b.

        Args:
            phi_batch: Feature batch of shape (batch_size, m).
            y_batch: Target batch of shape (batch_size,).
        """
        if phi_batch.shape[1] != self._feature_dim:
            raise ValueError(
                f"phi_batch feature dim {phi_batch.shape[1]} != "
                f"expected {self._feature_dim}"
            )

        self._S += phi_batch.T @ phi_batch
        self._b += phi_batch.T @ y_batch
        self._n += phi_batch.shape[0]

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return accumulated normal equations.

        Returns:
            Tuple of (S, b).
        """
        return self._S.copy(), self._b.copy()

    def reset(self, feature_dim: int) -> None:
        """Reset accumulator with new feature dimension.

        Args:
            feature_dim: New feature dimension.
        """
        self._feature_dim = feature_dim
        self._S = np.zeros((feature_dim, feature_dim), dtype=self._dtype)
        self._b = np.zeros(feature_dim, dtype=self._dtype)
        self._n = 0

    @property
    def n_accumulated(self) -> int:
        """Number of accumulated samples."""
        return self._n

    @property
    def S(self) -> Array:
        """Current accumulated normal matrix."""
        return self._S.copy()

    @property
    def b(self) -> Array:
        """Current accumulated normal vector."""
        return self._b.copy()
