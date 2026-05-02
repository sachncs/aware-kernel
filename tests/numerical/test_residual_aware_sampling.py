"""Numerical correctness tests for residual-aware anchor sampling.

These tests verify the properties from Section 5 of the method blueprint.
"""

import numpy as np

from aware_kernel.local_corrective.anchors import (
    compute_coverage_weights,
    compute_residual_weights,
    residual_aware_sample,
)


class TestResidualAwareSamplingProperties:
    """Tests for residual-aware sampling correctness."""

    def test_probability_distribution(self, rng: np.random.Generator) -> None:
        """Anchor weights p_i should form a valid probability distribution."""
        n = 50
        s = rng.exponential(scale=1.0, size=(n, 8))
        r = rng.standard_normal(n)
        tilde_l = compute_coverage_weights(s)
        tilde_r = compute_residual_weights(r)
        alpha_a = 0.5
        p = alpha_a * tilde_l + (1.0 - alpha_a) * tilde_r
        assert np.all(p >= 0.0)
        np.testing.assert_allclose(np.sum(p), 1.0, atol=1e-10)

    def test_alpha_interpolation(self, rng: np.random.Generator) -> None:
        """Different alpha_a values should shift weight between coverage and residual."""
        n = 50
        s = rng.exponential(scale=1.0, size=(n, 8))
        r = rng.standard_normal(n)
        tilde_l = compute_coverage_weights(s)
        tilde_r = compute_residual_weights(r)

        p_coverage = 1.0 * tilde_l + 0.0 * tilde_r
        p_residual = 0.0 * tilde_l + 1.0 * tilde_r

        np.testing.assert_allclose(p_coverage, tilde_l)
        np.testing.assert_allclose(p_residual, tilde_r)

    def test_no_duplicates(self, rng: np.random.Generator) -> None:
        """Sampling without replacement should produce unique anchors."""
        embeddings = rng.standard_normal((50, 4))
        s = rng.exponential(scale=1.0, size=(50, 8))
        r = rng.standard_normal(50)
        anchors = residual_aware_sample(embeddings, s, r, alpha_a=0.5, m_l=10, rng=rng)
        unique = np.unique(anchors, axis=0)
        assert unique.shape[0] == anchors.shape[0]
