"""Base class for memory accumulators.

The MemoryAccumulator protocol is defined in aware_kernel.aware.types.
This module provides shared utilities for dimension tracking.
"""

from abc import ABC, abstractmethod

import numpy as np

from aware_kernel.aware.types import Array


class BaseMemoryAccumulator(ABC):
    """Abstract base class for memory accumulators with dimension tracking."""

    def __init__(self, feature_dim: int, dtype: np.dtype = np.float64) -> None:
        """Initialize accumulator.

        Args:
            feature_dim: Current feature dimension m.
            dtype: Numeric dtype for accumulations.
        """
        self._feature_dim = feature_dim
        self._dtype = dtype

    @property
    def feature_dim(self) -> int:
        """Current feature dimension."""
        return self._feature_dim

    @abstractmethod
    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch."""
        ...

    @abstractmethod
    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return normal equations (S, b)."""
        ...

    @abstractmethod
    def reset(self, feature_dim: int) -> None:
        """Reset accumulator with new feature dimension."""
        ...
