"""Numerical correctness tests for whitening map properties.

These tests verify the algebraic guarantees from Section 4 of the method blueprint.
"""

import numpy as np
import pytest

from aware_kernel.utils.numerics import (
    compute_epsilon,
    eigenvalue_clip,
    retained_indices,
    soft_spectral_truncate,
)


class TestWhiteningMapProperties:
    """Tests for soft-truncated whitening correctness."""

    def test_soft_truncated_non_negative(self, rng: np.random.Generator) -> None:
        """Soft-truncated eigenvalues must be non-negative."""
        eigenvalues = rng.exponential(scale=1.0, size=20)
        eigenvalues[0] = -0.1  # inject a negative one
        clipped = eigenvalue_clip(eigenvalues, min_val=0.0)
        eps = compute_epsilon(trace_w=float(np.sum(clipped)), m_g=20, alpha_epsilon=1e-5)
        truncated = soft_spectral_truncate(clipped, tau=1e-3, epsilon=eps)
        assert np.all(truncated >= 0.0)

    def test_retained_rank_bound(self, rng: np.random.Generator) -> None:
        """Retained rank must be <= total number of eigenvalues."""
        eigenvalues = rng.exponential(scale=1.0, size=15)
        _, rank = retained_indices(eigenvalues, tau=1e-3)
        assert 0 <= rank <= 15

    def test_identity_covariance_on_landmarks(self, rng: np.random.Generator) -> None:
        """Whitened features on landmarks should have bounded covariance.

        For soft-truncated whitening, the covariance is not exactly identity,
        but it should be diagonal-dominant and finite.
        """
        m_g = 20
        Z = rng.standard_normal((m_g, 5))
        W = Z @ Z.T  # linear kernel for simplicity

        eigenvalues, U = np.linalg.eigh(W)
        clipped = eigenvalue_clip(eigenvalues, min_val=0.0)
        eps = compute_epsilon(trace_w=float(np.sum(clipped)), m_g=m_g, alpha_epsilon=1e-5)
        truncated = soft_spectral_truncate(clipped, tau=1e-3, epsilon=eps)
        indices, rank = retained_indices(clipped, tau=1e-3)

        if rank == 0:
            pytest.skip("No eigenvalues retained")

        M_g = U[:, indices] @ np.diag(truncated[indices])
        phi_g = W @ M_g
        cov = phi_g.T @ phi_g

        # Check covariance is finite and diagonal elements are positive
        assert np.all(np.isfinite(cov))
        diag = np.diag(cov)
        assert np.all(diag > 0)

        # Check off-diagonal is small relative to diagonal (near-orthogonal components)
        off_diag = cov - np.diag(diag)
        max_off = np.max(np.abs(off_diag))
        assert max_off < 1e-6
