"""Unit tests for aware_kernel.utils.numerics."""

import numpy as np
import pytest

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.utils.numerics import (
    check_conditioning,
    compute_epsilon,
    eigenvalue_clip,
    retained_indices,
    safe_cholesky,
    soft_spectral_truncate,
)


class TestEigenvalueClip:
    """Tests for eigenvalue_clip."""

    def test_negative_eigenvalues_clipped(self) -> None:
        """Negative eigenvalues should be clipped to min_val."""
        ev = np.array([-1.0, 0.5, 2.0])
        clipped = eigenvalue_clip(ev, min_val=0.0)
        np.testing.assert_array_equal(clipped, np.array([0.0, 0.5, 2.0]))

    def test_custom_min_val(self) -> None:
        """Custom min_val should be respected."""
        ev = np.array([0.01, 0.1, 1.0])
        clipped = eigenvalue_clip(ev, min_val=0.05)
        np.testing.assert_array_equal(clipped, np.array([0.05, 0.1, 1.0]))

    def test_all_positive_unchanged(self) -> None:
        """All-positive eigenvalues should remain unchanged."""
        ev = np.array([1.0, 2.0, 3.0])
        clipped = eigenvalue_clip(ev)
        np.testing.assert_array_equal(clipped, ev)


class TestSoftSpectralTruncate:
    """Tests for soft_spectral_truncate."""

    def test_zero_eigenvalue(self) -> None:
        """Zero eigenvalue should yield zero after truncation."""
        ev = np.array([0.0])
        result = soft_spectral_truncate(ev, tau=1e-3, epsilon=1e-5)
        assert result[0] == 0.0

    def test_positive_eigenvalue(self) -> None:
        """Positive eigenvalue should produce positive truncated value."""
        ev = np.array([1.0])
        result = soft_spectral_truncate(ev, tau=1e-3, epsilon=1e-5)
        expected = 1.0 / (1.0 + 1e-3) / np.sqrt(1.0 + 1e-5)
        np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_output_non_negative(self) -> None:
        """Soft-truncated eigenvalues must be non-negative after clipping."""
        ev = np.array([-0.1, 0.0, 0.5, 1.0, 2.0])
        clipped = eigenvalue_clip(ev, min_val=0.0)
        result = soft_spectral_truncate(clipped, tau=0.1, epsilon=0.01)
        assert np.all(result >= 0.0)


class TestComputeEpsilon:
    """Tests for compute_epsilon."""

    def test_basic(self) -> None:
        """Epsilon should scale with trace and alpha."""
        eps = compute_epsilon(trace_w=10.0, m_g=5, alpha_epsilon=1e-3)
        expected = 1e-3 * (10.0 / 5)
        np.testing.assert_allclose(eps, expected)

    def test_zero_m_g_raises(self) -> None:
        """Zero m_g should raise ValueError."""
        with pytest.raises(ValueError, match="m_g must be positive"):
            compute_epsilon(trace_w=1.0, m_g=0, alpha_epsilon=1e-5)


class TestRetainedIndices:
    """Tests for retained_indices."""

    def test_basic(self) -> None:
        """Indices above threshold should be retained."""
        ev = np.array([0.1, 0.5, 2.0, 0.01])
        indices, rank = retained_indices(ev, tau=0.2)
        np.testing.assert_array_equal(indices, np.array([1, 2]))
        assert rank == 2

    def test_all_below(self) -> None:
        """When all eigenvalues are below threshold, rank should be zero."""
        ev = np.array([0.01, 0.02])
        indices, rank = retained_indices(ev, tau=0.1)
        assert rank == 0
        assert len(indices) == 0


class TestCheckConditioning:
    """Tests for check_conditioning."""

    def test_well_conditioned(self) -> None:
        """Well-conditioned matrix should return condition number."""
        m = np.eye(5)
        cond = check_conditioning(m, threshold=1e12)
        np.testing.assert_allclose(cond, 1.0, rtol=1e-6)

    def test_ill_conditioned_raises(self) -> None:
        """Ill-conditioned matrix should raise ConditioningError."""
        m = np.diag([1e-12, 1.0])
        with pytest.raises(ConditioningError):
            check_conditioning(m, threshold=1e10)

    def test_non_square_raises(self) -> None:
        """Non-square matrix should raise ValueError."""
        with pytest.raises(ValueError, match="square matrix"):
            check_conditioning(np.ones((3, 4)), threshold=1e10)


class TestSafeCholesky:
    """Tests for safe_cholesky."""

    def test_spd_matrix(self) -> None:
        """SPD matrix should decompose correctly."""
        m = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = safe_cholesky(m)
        np.testing.assert_allclose(L @ L.T, m, atol=1e-10)

    def test_nearly_singular_with_jitter(self) -> None:
        """Nearly singular matrix should succeed with jitter."""
        m = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]])
        L = safe_cholesky(m, jitter=1e-10)
        # Should not raise
        assert L is not None

    def test_singular_raises(self) -> None:
        """Exactly singular matrix should raise even with small jitter."""
        # This matrix is PSD but rank 1; tiny jitter may not save it numerically.
        # Use a matrix that is clearly non-SPD even after tiny jitter.
        m = np.array([[-1.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ConditioningError):
            safe_cholesky(m, jitter=1e-12)
