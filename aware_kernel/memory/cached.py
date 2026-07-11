"""Cached-feature memory accumulator.

Stores the full feature matrix ``Phi`` explicitly, enabling direct
normal-equation construction and feature reconstruction.

Memory complexity: O(n * m) where ``n`` is the number of accumulated
samples and ``m`` is the feature dimension.

The cached mode is preferred when:
* The dataset fits in memory (n * m * 8 bytes is acceptable).
* You need access to the full feature matrix (e.g., for recomputing
  residuals during the refresh pipeline).
* You want the simplest possible normal-equation construction.

The normal equations are computed on demand via matrix multiplication:

    ``S = Phi^T Phi``
    ``b = Phi^T y``

This is O(n * m^2) per call, which is more expensive than the streamed
mode's O(m^2) but provides the full feature matrix for other uses.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.memory.base import BaseMemoryAccumulator


class CachedMemoryAccumulator(BaseMemoryAccumulator):
    """Accumulator that stores the full feature matrix ``Phi`` explicitly.

    This is the simplest accumulator implementation.  It appends each
    mini-batch to the stored matrix and computes normal equations on
    demand.

    Attributes:
        Phi: Feature matrix of shape ``(n_accumulated, m)``.  ``None``
            before the first accumulation.
        y: Target vector of shape ``(n_accumulated,)``.  ``None``
            before the first accumulation.
    """

    def __init__(self, feature_dim: int, dtype: np.dtype = np.float64) -> None:
        """Initialize cached accumulator.

        Args:
            feature_dim: Feature dimension ``m``.
            dtype: Numeric dtype for accumulations.
        """
        super().__init__(feature_dim, dtype)
        self.Phi: Optional[Array] = None
        self.y: Optional[Array] = None

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch by appending to stored ``Phi``.

        On the first call, the stored matrix is initialized from the
        batch.  On subsequent calls, the batch is concatenated along
        axis 0.

        Args:
            phi_batch: Feature batch of shape ``(batch_size, m)``.
            y_batch: Target batch of shape ``(batch_size,)``.

        Raises:
            ValueError: If the feature dimension of ``phi_batch`` does
                not match the expected dimension.
        """
        if phi_batch.shape[1] != self.feature_dim:
            raise ValueError(
                f"phi_batch feature dim {phi_batch.shape[1]} != "
                f"expected {self.feature_dim}"
            )

        if self.Phi is None:
            # First accumulation: initialize from batch.
            # astype with copy=False avoids an unnecessary copy when
            # the dtype already matches.
            self.Phi = phi_batch.astype(self.dtype, copy=False)
            self.y = y_batch.astype(self.dtype, copy=False)
        else:
            self.Phi = np.concatenate([self.Phi, phi_batch], axis=0)
            self.y = np.concatenate([self.y, y_batch], axis=0)

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return normal equations computed from stored ``Phi``.

        Returns:
            Tuple of ``(S, b)`` where ``S = Phi^T Phi`` has shape
            ``(m, m)`` and ``b = Phi^T y`` has shape ``(m,)``.

        Raises:
            ValueError: If no data has been accumulated yet.
        """
        if self.Phi is None or self.y is None:
            raise ValueError("No data accumulated")
        s = self.Phi.T @ self.Phi
        b = self.Phi.T @ self.y
        return s, b

    def get_features(self) -> tuple[Array, Array]:
        """Return stored features and targets.

        This is specific to the cached mode and provides access to the
        full feature matrix for tasks like residual computation.

        Returns:
            Tuple of ``(Phi, y)``.

        Raises:
            ValueError: If no data has been accumulated yet.
        """
        if self.Phi is None or self.y is None:
            raise ValueError("No data accumulated")
        return self.Phi, self.y

    def reset(self, feature_dim: int) -> None:
        """Reset accumulator, discarding all stored data.

        Called when the discrete basis is refreshed and the feature
        dimension changes.

        Args:
            feature_dim: New feature dimension.
        """
        self.feature_dim = feature_dim
        self.Phi = None
        self.y = None

    @property
    def n_accumulated(self) -> int:
        """Number of accumulated samples."""
        if self.Phi is None:
            return 0
        return self.Phi.shape[0]
