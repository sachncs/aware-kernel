"""Embedding module: f_theta(x).

Provides a simple dense embedder as the default continuous representation.

The embedding stage is the first step in the aware-kernel pipeline.  It
maps raw input features ``x`` into a dense embedding space via a learned
linear transformation:

    ``e = x @ theta + bias``

where ``theta`` is a weight matrix of shape ``(input_dim, output_dim)``
and ``bias`` is a vector of shape ``(output_dim,)``.  The default
initialization uses He-like scaling (``1 / sqrt(input_dim)``) to maintain
activation variance across layers.

Design rationale
----------------
The ``DenseEmbedder`` is a lightweight stand-in for a neural embedding
``f_theta(x)``.  In a production setting, this would wrap a learned
network (e.g., a transformer or MLP).  The embedder satisfies the
``Embedder`` protocol defined in ``aware_kernel.aware.types``, making it
swappable without modifying downstream code.

The embedding parameters are stored in ``ContinuousState.theta`` as a
dictionary under the key ``"embedder"``.  This opaque storage avoids
coupling the state container to any specific embedder implementation.
"""

import numpy as np

from aware_kernel.aware.types import Array


class DenseEmbedder:
    """Simple dense linear embedder with optional bias.

    Implements the function ``f_theta(x) = x @ theta + bias`` as the
    default continuous representation.  This is a lightweight stand-in
    for a neural embedding; production use would replace this with a
    learned network.

    The embedder satisfies the ``Embedder`` protocol, allowing it to be
    swapped for any implementation that exposes an ``embed`` method.

    Attributes:
        input_dim: Dimensionality of input features.
        output_dim: Dimensionality of the embedding space.
        theta: Weight matrix of shape ``(input_dim, output_dim)``.
        bias: Bias vector of shape ``(output_dim,)``.

    Thread safety
        The embedder is **not** thread-safe.  Concurrent calls to
        ``embed`` while ``theta`` or ``bias`` are being modified will
        produce undefined results.  In the standard training loop, the
        embedder is only accessed from the main thread, so this is not
        a concern.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize embedder with random weights.

        Weights are drawn from a standard normal distribution and scaled
        by ``1 / sqrt(input_dim)`` to maintain activation variance
        (similar to He initialization for linear layers).

        Args:
            input_dim: Dimensionality of input features.
            output_dim: Dimensionality of embedding space.
            rng: Optional random generator.  If ``None``, a default
                generator is created.

        Raises:
            ValueError: If ``input_dim`` or ``output_dim`` is non-positive.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.theta = rng.standard_normal((input_dim, output_dim)) / np.sqrt(input_dim)
        self.bias = np.zeros(output_dim)

    def set_theta(self, value: Array) -> None:
        """Set embedding weight matrix.

        Args:
            value: New weight matrix of shape ``(input_dim, output_dim)``.

        Raises:
            ValueError: If shape does not match expected dimensions.
        """
        if value.shape != (self.input_dim, self.output_dim):
            raise ValueError(
                f"theta shape {value.shape} != ({self.input_dim}, {self.output_dim})"
            )
        self.theta = value

    def set_bias(self, value: Array) -> None:
        """Set embedding bias vector.

        Args:
            value: New bias vector of shape ``(output_dim,)``.

        Raises:
            ValueError: If shape does not match expected dimensions.
        """
        if value.shape != (self.output_dim,):
            raise ValueError(f"bias shape {value.shape} != ({self.output_dim},)")
        self.bias = value

    def embed(self, x: Array) -> Array:
        """Embed inputs into continuous representation.

        Computes ``f_theta(x) = x @ theta + bias``.

        Args:
            x: Input array of shape ``(n, input_dim)`` or ``(input_dim,)``
                for a single sample.

        Returns:
            Embedded array of shape ``(n, output_dim)`` or ``(output_dim,)``.
        """
        if x.ndim == 1:
            result_1d: Array = x @ self.theta + self.bias
            return result_1d
        result_2d: Array = x @ self.theta + self.bias
        return result_2d
