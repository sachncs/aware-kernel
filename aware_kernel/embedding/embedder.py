"""Embedding module: f_theta(x).

Provides a simple dense embedder as the default continuous representation.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.types import Array


class DenseEmbedder:
    """Simple dense linear embedder with optional bias.

    This is a lightweight stand-in for a neural embedding f_theta(x).
    In a production setting, this would wrap a learned network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialize embedder with random weights.

        Args:
            input_dim: Dimensionality of input features.
            output_dim: Dimensionality of embedding space.
            rng: Optional random generator.
        """
        if rng is None:
            rng = np.random.default_rng()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._theta = rng.standard_normal((input_dim, output_dim)) / np.sqrt(input_dim)
        self._bias = np.zeros(output_dim)

    @property
    def theta(self) -> Array:
        """Embedding weight matrix."""
        return self._theta

    @theta.setter
    def theta(self, value: Array) -> None:
        """Set embedding weight matrix."""
        if value.shape != (self._input_dim, self._output_dim):
            raise ValueError(
                f"theta shape {value.shape} != ({self._input_dim}, {self._output_dim})"
            )
        self._theta = value

    @property
    def bias(self) -> Array:
        """Embedding bias vector."""
        return self._bias

    @bias.setter
    def bias(self, value: Array) -> None:
        """Set embedding bias vector."""
        if value.shape != (self._output_dim,):
            raise ValueError(f"bias shape {value.shape} != ({self._output_dim},)")
        self._bias = value

    def embed(self, x: Array) -> Array:
        """Embed inputs into continuous representation.

        Args:
            x: Input array of shape (n, input_dim) or (input_dim,).

        Returns:
            Embedded array of shape (n, output_dim) or (output_dim,).
        """
        if x.ndim == 1:
            return x @ self._theta + self._bias
        return x @ self._theta + self._bias
