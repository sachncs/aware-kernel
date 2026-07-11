"""Shared type aliases and structural protocols for aware-kernel.

This module defines the foundational type vocabulary used across the entire
codebase.  It contains:

* **Type aliases** -- ``Array`` and ``FeatureMatrix`` provide semantic meaning
  to raw ``numpy.ndarray`` parameters, making function signatures more
  readable and self-documenting.
* **Structural protocols** (``typing.Protocol``) -- These define the minimal
  interfaces that components must satisfy to be interchangeable.  Using
  protocols instead of base classes keeps modules decoupled and makes it
  straightforward to swap implementations (e.g., replacing the default
  Cholesky solver with a GPU-backed solver, or substituting a learned
  embedder for the linear one).

Design rationale
----------------
The aware-kernel pipeline is composed of clearly separated stages
(embedding, projection, basis construction, fusion, solving, etc.).
Protocols allow each stage to depend only on an interface, not on a
concrete class, so stages can be tested in isolation and composed freely.

Dependencies
------------
Only ``numpy`` and ``typing`` from the standard library are required.
"""

from typing import Protocol, Union

import numpy as np

Array = Union[np.ndarray]
FeatureMatrix = np.ndarray


class Embedder(Protocol):
    """Protocol for mapping raw inputs into a continuous embedding space.

    An ``Embedder`` implements the function ``f_theta(x)`` described in
    Section 3 of the method blueprint.  The default implementation is a
    simple linear layer (``DenseEmbedder``), but the protocol allows
    arbitrary neural networks, random Fourier feature maps, or any other
    embedding strategy.

    Implementations must be deterministic given the same input so that the
    downstream projection and basis stages are reproducible.
    """

    def embed(self, x: Array) -> Array:
        """Embed inputs into continuous representation.

        Args:
            x: Input array of shape ``(n, ...)`` where the trailing
                dimensions are determined by the embedding implementation.

        Returns:
            Embedded array of shape ``(n, embedding_dim)``.

        Raises:
            ValueError: If input shapes are incompatible with the
                embedding weights.
        """
        ...


class FeatureBuilder(Protocol):
    """Protocol for constructing feature matrices from projected embeddings.

    A ``FeatureBuilder`` converts projected embeddings ``u`` into the
    feature space used by the ridge regression solver.  Both the global
    Nyström builder and the local corrective builder implement this
    protocol, allowing the fusion stage to treat them uniformly.
    """

    def build(self, u: Array) -> FeatureMatrix:
        """Build feature matrix from projected embeddings.

        Args:
            u: Projected embeddings of shape ``(n, rank)`` where ``rank``
                is the effective dimensionality after projection.

        Returns:
            Feature matrix of shape ``(n, feature_dim)`` where
            ``feature_dim`` depends on the builder (``r_g`` for global,
            ``m_l`` for local).
        """
        ...


class RidgeSolver(Protocol):
    """Protocol for solving ridge regression normal equations.

    A ``RidgeSolver`` takes the pre-assembled normal equations
    ``S = Phi^T Phi + lambda * I`` and ``b = Phi^T y`` and returns the
    coefficient vector ``w`` that minimizes the Tikhonov-regularized
    least-squares objective.  Two implementations are provided:

    * ``DirectRidgeSolver`` -- Cholesky decomposition, O(m^3).
    * ``IterativeRidgeSolver`` -- Preconditioned conjugate gradients,
      O(k * m^2) where k is the number of iterations.
    """

    def solve(self, s: Array, b: Array) -> Array:
        """Solve ridge normal equations ``(S) w = b``.

        Args:
            s: Normal matrix ``S = Phi^T Phi + lambda * I`` of shape
                ``(m, m)``.  Must be symmetric positive-definite.
            b: Normal vector ``b = Phi^T y`` of shape ``(m,)``.

        Returns:
            Coefficient vector ``w`` of shape ``(m,)`` that solves the
            ridge regression problem.

        Raises:
            ConditioningError: If the matrix is too ill-conditioned.
        """
        ...


class MemoryAccumulator(Protocol):
    """Protocol for accumulating normal equations over mini-batches.

    Two implementations are provided:

    * ``CachedMemoryAccumulator`` -- Stores the full feature matrix
      ``Phi`` explicitly.  Memory O(n * m) but enables direct
      normal-equation construction.
    * ``StreamedMemoryAccumulator`` -- Accumulates ``S = Phi^T Phi``
      and ``b = Phi^T y`` incrementally.  Memory O(m^2) but discards
      individual samples.

    The choice between cached and streamed modes is controlled by
    ``TrainingConfig.memory_mode``.
    """

    def accumulate(self, phi_batch: Array, y_batch: Array) -> None:
        """Accumulate a mini-batch of features and targets.

        Args:
            phi_batch: Feature batch of shape ``(batch_size, feature_dim)``.
            y_batch: Target batch of shape ``(batch_size,)``.
        """
        ...

    def get_normal_eq(self) -> tuple[Array, Array]:
        """Return accumulated normal equations.

        Returns:
            Tuple of ``(S, b)`` where ``S`` has shape ``(m, m)`` and
            ``b`` has shape ``(m,)``.  ``S`` includes any ridge
            regularization added by the caller.
        """
        ...


class RefreshPolicy(Protocol):
    """Protocol for deciding when discrete basis parameters should refresh.

    A ``RefreshPolicy`` inspects the current training state and metrics
    to determine whether the cost of a discrete refresh (re-selecting
    landmarks, anchors, and whitening maps) is justified by the expected
    improvement in validation loss.

    The default implementation lives in ``aware_kernel.refresh.controller``
    and combines five conditions: drift threshold, cooldown, warmup,
    hysteresis, and budget-scaled validation gain.
    """

    def should_refresh(self, state: object, metrics: dict) -> bool:
        """Evaluate whether a refresh should trigger.

        Args:
            state: Current training state (typically ``FullState``).
            metrics: Dictionary of current metrics including at least
                ``drift`` and ``val_gain``.

        Returns:
            ``True`` if all conditions are met for a refresh.
        """
        ...
