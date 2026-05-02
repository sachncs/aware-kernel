"""Unit tests for aware_kernel.memory modules."""

import numpy as np
import pytest

from aware_kernel.memory.cached import CachedMemoryAccumulator
from aware_kernel.memory.streamed import StreamedMemoryAccumulator


class TestCachedMemoryAccumulator:
    """Tests for CachedMemoryAccumulator."""

    def test_accumulate_shape(self, rng: np.random.Generator) -> None:
        """Accumulated features should have correct shape."""
        acc = CachedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 5))
        y = rng.standard_normal(10)
        acc.accumulate(phi, y)
        phi_out, y_out = acc.get_features()
        assert phi_out.shape == (10, 5)
        assert y_out.shape == (10,)

    def test_multiple_batches(self, rng: np.random.Generator) -> None:
        """Multiple batches should concatenate."""
        acc = CachedMemoryAccumulator(feature_dim=5)
        for _ in range(3):
            phi = rng.standard_normal((5, 5))
            y = rng.standard_normal(5)
            acc.accumulate(phi, y)
        assert acc.n_accumulated == 15

    def test_normal_eq(self, rng: np.random.Generator) -> None:
        """Normal equations should match manual computation."""
        acc = CachedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        acc.accumulate(phi, y)
        s, b = acc.get_normal_eq()
        np.testing.assert_allclose(s, phi.T @ phi, atol=1e-10)
        np.testing.assert_allclose(b, phi.T @ y, atol=1e-10)

    def test_reset(self, rng: np.random.Generator) -> None:
        """Reset should clear accumulated data."""
        acc = CachedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 5))
        y = rng.standard_normal(10)
        acc.accumulate(phi, y)
        acc.reset(feature_dim=8)
        assert acc.feature_dim == 8
        assert acc.n_accumulated == 0
        with pytest.raises(ValueError, match="No data accumulated"):
            acc.get_features()

    def test_wrong_feature_dim_raises(self, rng: np.random.Generator) -> None:
        """Accumulating with wrong feature dim should raise ValueError."""
        acc = CachedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 3))
        y = rng.standard_normal(10)
        with pytest.raises(ValueError, match="feature dim"):
            acc.accumulate(phi, y)


class TestStreamedMemoryAccumulator:
    """Tests for StreamedMemoryAccumulator."""

    def test_accumulate_shape(self, rng: np.random.Generator) -> None:
        """Accumulator should track correct dimensions."""
        acc = StreamedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 5))
        y = rng.standard_normal(10)
        acc.accumulate(phi, y)
        assert acc.n_accumulated == 10

    def test_normal_eq(self, rng: np.random.Generator) -> None:
        """Normal equations should match manual computation."""
        acc = StreamedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        acc.accumulate(phi, y)
        s, b = acc.get_normal_eq()
        np.testing.assert_allclose(s, phi.T @ phi, atol=1e-10)
        np.testing.assert_allclose(b, phi.T @ y, atol=1e-10)

    def test_multiple_batches(self, rng: np.random.Generator) -> None:
        """Multiple batches should sum correctly."""
        acc = StreamedMemoryAccumulator(feature_dim=5)
        total_phi = []
        total_y = []
        for _ in range(3):
            phi = rng.standard_normal((5, 5))
            y = rng.standard_normal(5)
            total_phi.append(phi)
            total_y.append(y)
            acc.accumulate(phi, y)

        s, b = acc.get_normal_eq()
        phi_all = np.concatenate(total_phi, axis=0)
        y_all = np.concatenate(total_y, axis=0)
        np.testing.assert_allclose(s, phi_all.T @ phi_all, atol=1e-10)
        np.testing.assert_allclose(b, phi_all.T @ y_all, atol=1e-10)

    def test_reset(self, rng: np.random.Generator) -> None:
        """Reset should zero out accumulators."""
        acc = StreamedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 5))
        y = rng.standard_normal(10)
        acc.accumulate(phi, y)
        acc.reset(feature_dim=8)
        assert acc.feature_dim == 8
        assert acc.n_accumulated == 0
        s, b = acc.get_normal_eq()
        np.testing.assert_allclose(s, 0.0, atol=1e-12)
        np.testing.assert_allclose(b, 0.0, atol=1e-12)

    def test_wrong_feature_dim_raises(self, rng: np.random.Generator) -> None:
        """Accumulating with wrong feature dim should raise ValueError."""
        acc = StreamedMemoryAccumulator(feature_dim=5)
        phi = rng.standard_normal((10, 3))
        y = rng.standard_normal(10)
        with pytest.raises(ValueError, match="feature dim"):
            acc.accumulate(phi, y)


class TestCachedVsStreamed:
    """Tests comparing cached and streamed modes."""

    def test_normal_eq_parity(self, rng: np.random.Generator) -> None:
        """Cached and streamed should produce identical S and b."""
        phi = rng.standard_normal((50, 8))
        y = rng.standard_normal(50)

        cached = CachedMemoryAccumulator(feature_dim=8)
        cached.accumulate(phi, y)
        s_c, b_c = cached.get_normal_eq()

        streamed = StreamedMemoryAccumulator(feature_dim=8)
        streamed.accumulate(phi, y)
        s_s, b_s = streamed.get_normal_eq()

        np.testing.assert_allclose(s_c, s_s, atol=1e-10)
        np.testing.assert_allclose(b_c, b_s, atol=1e-10)

    def test_batch_parity(self, rng: np.random.Generator) -> None:
        """Cached and streamed should match with multiple batches."""
        cached = CachedMemoryAccumulator(feature_dim=6)
        streamed = StreamedMemoryAccumulator(feature_dim=6)

        for _ in range(4):
            phi = rng.standard_normal((10, 6))
            y = rng.standard_normal(10)
            cached.accumulate(phi, y)
            streamed.accumulate(phi, y)

        s_c, b_c = cached.get_normal_eq()
        s_s, b_s = streamed.get_normal_eq()

        np.testing.assert_allclose(s_c, s_s, atol=1e-10)
        np.testing.assert_allclose(b_c, b_s, atol=1e-10)
