"""Shared types and protocols."""

from typing import Protocol, Union

import numpy as np

Array = Union[np.ndarray]
FeatureMatrix = np.ndarray


class Embedder(Protocol):
    """Protocol for embedding functions."""

    def embed(self, x: Array) -> Array:
        """Embed inputs into continuous representation.

        Args:
            x: Input array of shape (n, ...).

        Returns:
            Embedded array of shape (n, embedding_dim).
        """
        ...


class FeatureBuilder(Protocol):
    """Protocol for feature builders."""

    def build(self, u: Array) -> FeatureMatrix:
        """Build feature matrix from projected embeddings.

        Args:
            u: Projected embeddings of shape (n, rank).

        Returns:
            Feature matrix of shape (n, feature_dim).
        """
        ...


class RidgeSolver(Protocol):
    """Protocol for ridge regression solvers."""

    def solve(self, s: Array, b: Array) -> Array:
        """Solve ridge normal equations.

        Args:
            s: Normal matrix S = Phi^T Phi of shape (m, m).
            b: Normal vector b = Phi^T y of shape (m,).

        Returns:
            Coefficient vector w of shape (m,).
        """
        ...


class MemoryAccumulator(Protocol):
    """Protocol for memory accumulators."""

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch of features and targets.

        Args:
            phi_batch: Feature batch of shape (batch_size, feature_dim).
            y_batch: Target batch of shape (batch_size,).
        """
        ...

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return accumulated normal equations.

        Returns:
            Tuple of (S, b) where S has shape (m, m) and b has shape (m,).
        """
        ...


class RefreshPolicy(Protocol):
    """Protocol for refresh policies."""

    def should_refresh(self, state: object, metrics: dict) -> bool:
        """Evaluate whether a refresh should trigger.

        Args:
            state: Current training state.
            metrics: Dictionary of current metrics.

        Returns:
            True if refresh should trigger.
        """
        ...
