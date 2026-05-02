"""Unit tests for aware_kernel.embedding modules."""

import numpy as np
import pytest

from aware_kernel.embedding.embedder import DenseEmbedder
from aware_kernel.embedding.projector import (
    Projector,
    normalize_embeddings,
    project_embeddings,
)


class TestNormalizeEmbeddings:
    """Tests for normalize_embeddings."""

    def test_unit_norm(self, rng: np.random.Generator) -> None:
        """Normalized vectors should have unit norm (up to delta)."""
        e = rng.standard_normal((10, 5))
        normalized = normalize_embeddings(e, delta=0.0)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector(self) -> None:
        """Zero vector should remain zero (delta prevents division by zero)."""
        e = np.zeros((3, 4))
        normalized = normalize_embeddings(e, delta=1e-8)
        np.testing.assert_array_equal(normalized, np.zeros((3, 4)))

    def test_single_vector(self) -> None:
        """1-D input should work."""
        e = np.array([3.0, 4.0])
        normalized = normalize_embeddings(e, delta=0.0)
        np.testing.assert_allclose(np.linalg.norm(normalized), 1.0)


class TestProjectEmbeddings:
    """Tests for project_embeddings."""

    def test_batch(self, rng: np.random.Generator) -> None:
        """Batch projection should preserve shape."""
        e = rng.standard_normal((10, 5))
        R = np.eye(5)
        projected = project_embeddings(e, R)
        assert projected.shape == (10, 5)
        np.testing.assert_allclose(projected, e, atol=1e-10)

    def test_single_vector(self) -> None:
        """1-D input should produce 1-D output."""
        e = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)
        projected = project_embeddings(e, R)
        assert projected.ndim == 1
        np.testing.assert_allclose(projected, e)


class TestDenseEmbedder:
    """Tests for DenseEmbedder."""

    def test_output_shape_batch(self, rng: np.random.Generator) -> None:
        """Batch embedding should have shape (n, output_dim)."""
        embedder = DenseEmbedder(input_dim=4, output_dim=8, rng=rng)
        x = rng.standard_normal((20, 4))
        e = embedder.embed(x)
        assert e.shape == (20, 8)

    def test_output_shape_single(self, rng: np.random.Generator) -> None:
        """Single vector embedding should have shape (output_dim,)."""
        embedder = DenseEmbedder(input_dim=4, output_dim=8, rng=rng)
        x = rng.standard_normal(4)
        e = embedder.embed(x)
        assert e.shape == (8,)

    def test_theta_setter_validation(self, rng: np.random.Generator) -> None:
        """Setting theta with wrong shape should raise ValueError."""
        embedder = DenseEmbedder(input_dim=4, output_dim=8, rng=rng)
        with pytest.raises(ValueError, match="theta shape"):
            embedder.theta = np.ones((3, 8))


class TestProjector:
    """Tests for Projector."""

    def test_transform_shape(self, rng: np.random.Generator) -> None:
        """Transform should preserve batch shape."""
        R = rng.standard_normal((5, 5))
        projector = Projector(R, delta=1e-8)
        e = rng.standard_normal((10, 5))
        u = projector.transform(e)
        assert u.shape == (10, 5)

    def test_transform_single(self, rng: np.random.Generator) -> None:
        """Single vector transform should work."""
        R = np.eye(5)
        projector = Projector(R)
        e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        u = projector.transform(e)
        assert u.ndim == 1
        np.testing.assert_allclose(u, e / np.linalg.norm(e), atol=1e-8)

    def test_R_property(self, rng: np.random.Generator) -> None:
        """R property should return a copy."""
        R = rng.standard_normal((5, 5))
        projector = Projector(R)
        retrieved = projector.R
        retrieved[0, 0] = 999.0
        np.testing.assert_array_equal(projector.R, R)
