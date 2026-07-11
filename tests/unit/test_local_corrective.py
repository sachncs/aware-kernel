"""Unit tests for aware_kernel.local_corrective modules."""

import numpy as np
import pytest

from aware_kernel.local_corrective.anchors import (
    compute_coverage_weights,
    compute_residual_weights,
    compute_residuals,
    residual_aware_sample,
)
from aware_kernel.local_corrective.orthogonalizer import (
    check_orthogonality,
    compute_orthogonalization_matrix,
    orthogonalize_local_features,
)
from aware_kernel.local_corrective.sparse_features import (
    build_local_features,
    compute_local_normalizers,
    compute_sparse_features,
)


class TestComputeSparseFeatures:
    """Tests for compute_sparse_features."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Output should have shape (n, m_l)."""
        embeddings = rng.standard_normal((50, 4))
        anchors = rng.standard_normal((10, 4))
        s = compute_sparse_features(embeddings, anchors, tau=0.5, k=3)
        assert s.shape == (50, 10)

    def test_sparsity(self, rng: np.random.Generator) -> None:
        """Each row should have exactly k non-zero entries."""
        embeddings = rng.standard_normal((50, 4))
        anchors = rng.standard_normal((10, 4))
        s = compute_sparse_features(embeddings, anchors, tau=0.5, k=3)
        nonzeros_per_row = np.count_nonzero(s, axis=1)
        np.testing.assert_array_equal(nonzeros_per_row, 3)

    def test_k_exceeds_m_l_raises(self, rng: np.random.Generator) -> None:
        """k > m_l should raise ValueError."""
        embeddings = rng.standard_normal((10, 4))
        anchors = rng.standard_normal((5, 4))
        with pytest.raises(ValueError, match="cannot exceed"):
            compute_sparse_features(embeddings, anchors, tau=0.5, k=10)

    def test_non_negative(self, rng: np.random.Generator) -> None:
        """All entries should be non-negative."""
        embeddings = rng.standard_normal((20, 4))
        anchors = rng.standard_normal((8, 4))
        s = compute_sparse_features(embeddings, anchors, tau=0.5, k=3)
        assert np.all(s >= 0.0)


class TestComputeLocalNormalizers:
    """Tests for compute_local_normalizers."""

    def test_positive(self, rng: np.random.Generator) -> None:
        """Normalizers should be positive."""
        s = rng.standard_normal((20, 5))
        d = compute_local_normalizers(s, eta=1e-8)
        assert np.all(d > 0.0)

    def test_empty_features(self) -> None:
        """All-zero features should still have positive normalizers due to eta."""
        s = np.zeros((10, 5))
        d = compute_local_normalizers(s, eta=1e-8)
        np.testing.assert_allclose(d, 1e-8)


class TestBuildLocalFeatures:
    """Tests for build_local_features."""

    def test_output_shapes(self, rng: np.random.Generator) -> None:
        """Should return phi_l and d with correct shapes."""
        embeddings = rng.standard_normal((30, 4))
        anchors = rng.standard_normal((8, 4))
        phi_l, d = build_local_features(embeddings, anchors, tau=0.5, k=3)
        assert phi_l.shape == (30, 8)
        assert d.shape == (8,)


class TestComputeResiduals:
    """Tests for compute_residuals."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """Residuals should have shape (n,)."""
        phi_g = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        r = compute_residuals(phi_g, y, lambda_reg=1e-2)
        assert r.shape == (20,)

    def test_perfect_fit(self, rng: np.random.Generator) -> None:
        """If y is in the column space, residual norm should be small."""
        phi_g = rng.standard_normal((20, 5))
        w_true = rng.standard_normal(5)
        y = phi_g @ w_true
        r = compute_residuals(phi_g, y, lambda_reg=1e-6)
        assert np.linalg.norm(r) < 1e-3


class TestComputeCoverageWeights:
    """Tests for compute_coverage_weights."""

    def test_sum_to_one(self, rng: np.random.Generator) -> None:
        """Coverage weights should sum to 1."""
        s = rng.exponential(scale=1.0, size=(20, 5))
        weights = compute_coverage_weights(s)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-10)

    def test_all_zero(self) -> None:
        """All-zero features should yield uniform weights."""
        s = np.zeros((10, 5))
        weights = compute_coverage_weights(s)
        np.testing.assert_allclose(weights, 0.1)


class TestComputeResidualWeights:
    """Tests for compute_residual_weights."""

    def test_sum_to_one(self, rng: np.random.Generator) -> None:
        """Residual weights should sum to 1."""
        r = rng.standard_normal(20)
        weights = compute_residual_weights(r)
        np.testing.assert_allclose(np.sum(weights), 1.0, atol=1e-10)

    def test_all_zero(self) -> None:
        """All-zero residuals should yield uniform weights."""
        r = np.zeros(10)
        weights = compute_residual_weights(r)
        np.testing.assert_allclose(weights, 0.1)


class TestResidualAwareSample:
    """Tests for residual_aware_sample."""

    def test_output_shape(self, rng: np.random.Generator) -> None:
        """Should select exactly m_l anchors."""
        embeddings = rng.standard_normal((50, 4))
        s = rng.exponential(scale=1.0, size=(50, 8))
        r = rng.standard_normal(50)
        anchors = residual_aware_sample(embeddings, s, r, alpha_a=0.5, m_l=8, rng=rng)
        assert anchors.shape == (8, 4)

    def test_valid_probability(self, rng: np.random.Generator) -> None:
        """Sampled anchors should be from the embedding set."""
        embeddings = rng.standard_normal((50, 4))
        s = rng.exponential(scale=1.0, size=(50, 8))
        r = rng.standard_normal(50)
        anchors = residual_aware_sample(embeddings, s, r, alpha_a=0.5, m_l=8, rng=rng)
        # Each anchor should match some embedding
        for a in anchors:
            matches = np.all(np.isclose(embeddings, a, atol=1e-10), axis=1)
            assert np.any(matches)


class TestComputeOrthogonalizationMatrix:
    """Tests for compute_orthogonalization_matrix."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """P_g should have shape (n, n)."""
        phi_g = rng.standard_normal((20, 5))
        P_g = compute_orthogonalization_matrix(phi_g, eta_o=1e-4)
        assert P_g.shape == (20, 20)

    def test_idempotent(self, rng: np.random.Generator) -> None:
        """P_g should be approximately idempotent: P_g @ P_g ~= P_g."""
        phi_g = rng.standard_normal((20, 5))
        P_g = compute_orthogonalization_matrix(phi_g, eta_o=1e-4)
        P_g_sq = P_g @ P_g
        np.testing.assert_allclose(P_g_sq, P_g, atol=1e-5)


class TestOrthogonalizeLocalFeatures:
    """Tests for orthogonalize_local_features."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """Phi_l_perp should have same shape as Phi_l."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        assert phi_l_perp.shape == phi_l.shape

    def test_orthogonal(self, rng: np.random.Generator) -> None:
        """Phi_g^T Phi_l_perp should be near zero."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        cross = phi_g.T @ phi_l_perp
        # Relative tolerance scaled by feature norms
        denom = float(
            np.linalg.norm(phi_g.T @ phi_g, "fro")
            * np.linalg.norm(phi_l_perp.T @ phi_l_perp, "fro")
        )
        if denom > 0:
            rel_norm = float(np.linalg.norm(cross, "fro") ** 2 / denom)
            assert rel_norm < 1e-5


class TestCheckOrthogonality:
    """Tests for check_orthogonality."""

    def test_true_for_orthogonalized(self, rng: np.random.Generator) -> None:
        """Should return True for orthogonalized features."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        assert check_orthogonality(phi_g, phi_l_perp, tol=1e-3)

    def test_false_for_random(self, rng: np.random.Generator) -> None:
        """Should return False for random non-orthogonal features."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        assert not check_orthogonality(phi_g, phi_l, tol=1e-6)
