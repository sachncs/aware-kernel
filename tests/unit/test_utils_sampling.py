"""Unit tests for aware_kernel.utils.sampling."""

import numpy as np
import pytest

from aware_kernel.utils.sampling import farthest_point_sampling, kmeans_pp


class TestKmeansPP:
    """Tests for kmeans_pp."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output should have shape (k, d)."""
        x = rng.standard_normal((100, 5))
        landmarks = kmeans_pp(x, k=10, rng=rng)
        assert landmarks.shape == (10, 5)

    def test_k_exceeds_n_raises(self, rng: np.random.Generator) -> None:
        """k > n should raise ValueError."""
        x = rng.standard_normal((5, 3))
        with pytest.raises(ValueError, match="cannot exceed"):
            kmeans_pp(x, k=10, rng=rng)

    def test_determinism(self, rng: np.random.Generator) -> None:
        """Same seed should yield same landmarks."""
        x = rng.standard_normal((50, 4))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        landmarks1 = kmeans_pp(x, k=5, rng=rng1)
        landmarks2 = kmeans_pp(x, k=5, rng=rng2)
        np.testing.assert_array_equal(landmarks1, landmarks2)


class TestFarthestPointSampling:
    """Tests for farthest_point_sampling."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output should have shape (k, d)."""
        x = rng.standard_normal((100, 5))
        landmarks = farthest_point_sampling(x, k=10, rng=rng)
        assert landmarks.shape == (10, 5)

    def test_k_exceeds_n_raises(self, rng: np.random.Generator) -> None:
        """k > n should raise ValueError."""
        x = rng.standard_normal((5, 3))
        with pytest.raises(ValueError, match="cannot exceed"):
            farthest_point_sampling(x, k=10, rng=rng)

    def test_first_center_randomness(self, rng: np.random.Generator) -> None:
        """Different seeds should produce different first centers."""
        x = rng.standard_normal((50, 4))
        landmarks1 = farthest_point_sampling(x, k=3, rng=np.random.default_rng(1))
        landmarks2 = farthest_point_sampling(x, k=3, rng=np.random.default_rng(2))
        assert not np.allclose(landmarks1[0], landmarks2[0])
