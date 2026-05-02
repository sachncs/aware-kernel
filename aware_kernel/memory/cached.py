"""Cached-feature memory accumulator.

Stores the full feature matrix Phi explicitly (memory O(n*m)).
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.memory.base import BaseMemoryAccumulator


class CachedMemoryAccumulator(BaseMemoryAccumulator):
    """Accumulator that stores Phi explicitly.

    Attributes:
        Phi: Feature matrix of shape (n_accumulated, m).
        y: Target vector of shape (n_accumulated,).
    """

    def __init__(self, feature_dim: int, dtype: np.dtype = np.float64) -> None:
        """Initialize cached accumulator.

        Args:
            feature_dim: Feature dimension m.
            dtype: Numeric dtype.
        """
        super().__init__(feature_dim, dtype)
        self._Phi: Optional[Array] = None
        self._y: Optional[Array] = None

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch by appending to stored Phi.

        Args:
            phi_batch: Feature batch of shape (batch_size, m).
            y_batch: Target batch of shape (batch_size,).
        """
        if phi_batch.shape[1] != self._feature_dim:
            raise ValueError(
                f"phi_batch feature dim {phi_batch.shape[1]} != "
                f"expected {self._feature_dim}"
            )

        if self._Phi is None:
            self._Phi = phi_batch.astype(self._dtype, copy=False)
            self._y = y_batch.astype(self._dtype, copy=False)
        else:
            self._Phi = np.concatenate([self._Phi, phi_batch], axis=0)
            self._y = np.concatenate([self._y, y_batch], axis=0)

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return normal equations computed from stored Phi.

        Returns:
            Tuple of (S, b) where S = Phi^T Phi and b = Phi^T y.

        Raises:
            ValueError: If no data has been accumulated.
        """
        if self._Phi is None or self._y is None:
            raise ValueError("No data accumulated")
        s = self._Phi.T @ self._Phi
        b = self._Phi.T @ self._y
        return s, b

    def get_features(self) -> tuple[Array, Array]:
        """Return stored features and targets.

        Returns:
            Tuple of (Phi, y).
        """
        if self._Phi is None or self._y is None:
            raise ValueError("No data accumulated")
        return self._Phi, self._y

    def reset(self, feature_dim: int) -> None:
        """Reset accumulator with new feature dimension.

        Args:
            feature_dim: New feature dimension.
        """
        self._feature_dim = feature_dim
        self._Phi = None
        self._y = None

    @property
    def n_accumulated(self) -> int:
        """Number of accumulated samples."""
        if self._Phi is None:
            return 0
        return self._Phi.shape[0]
