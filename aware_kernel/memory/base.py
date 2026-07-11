"""Base class for memory accumulators.

The ``MemoryAccumulator`` protocol is defined in ``aware_kernel.aware.types``.
This module provides the shared abstract base class that both the cached
and streamed accumulators inherit from.

Memory accumulators are responsible for maintaining the normal equations
``S = Phi^T Phi`` and ``b = Phi^T y`` as feature-target pairs arrive in
mini-batches.  The choice between cached and streamed modes is controlled
by ``TrainingConfig.memory_mode``:

* **Cached** (``CachedMemoryAccumulator``): Stores the full feature
  matrix ``Phi``.  Enables direct normal-equation construction and
  feature reconstruction.  O(n * m) memory.
* **Streamed** (``StreamedMemoryAccumulator``): Accumulates ``S`` and
  ``b`` directly.  O(m^2) memory, but discards individual samples.

Both modes produce identical coefficients when given the same data and
seed.
"""

from abc import ABC, abstractmethod

import numpy as np

from aware_kernel.aware.types import Array


class BaseMemoryAccumulator(ABC):
    """Abstract base class for memory accumulators with dimension tracking.

    Provides shared infrastructure for feature dimension management and
    dtype tracking.  Subclasses must implement ``accumulate``,
    ``get_normal_eq``, and ``reset``.

    Attributes:
        feature_dim: Current feature dimension ``m``.
        dtype: Numeric dtype for accumulations.
    """

    def __init__(self, feature_dim: int, dtype: np.dtype = np.float64) -> None:
        """Initialize accumulator.

        Args:
            feature_dim: Current feature dimension ``m``.  This may
                change when the discrete basis is refreshed.
            dtype: Numeric dtype for accumulations.  ``float64`` is
                recommended for numerical stability.
        """
        self.feature_dim = feature_dim
        self.dtype = dtype

    @abstractmethod
    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch of features and targets.

        Args:
            phi_batch: Feature batch of shape ``(batch_size, m)``.
            y_batch: Target batch of shape ``(batch_size,)``.
        """
        ...

    @abstractmethod
    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return accumulated normal equations ``(S, b)``.

        Returns:
            Tuple of ``(S, b)`` where ``S`` has shape ``(m, m)`` and
            ``b`` has shape ``(m,)``.
        """
        ...

    @abstractmethod
    def reset(self, feature_dim: int) -> None:
        """Reset accumulator with a new feature dimension.

        Called when the discrete basis is refreshed and the feature
        dimension changes.  All accumulated data is discarded.

        Args:
            feature_dim: New feature dimension.
        """
        ...
