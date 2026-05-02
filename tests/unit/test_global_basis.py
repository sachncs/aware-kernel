"""Unit tests for aware_kernel.global_basis modules."""

import numpy as np
import pytest

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.global_basis.whitening import build_whitening_map, compute_kernel_on_landmarks


class TestComputeKernelOnLandmarks:
    """Tests for compute_kernel_on_landmarks."""

    def test_psd(self, rng: np.random.Generator) -> None:
        """Kernel matrix should be symmetric PSD."""
        Z = rng.standard_normal((10, 4))
        W = compute_kernel_on_landmarks(Z)
        assert W.shape == (10, 10)
        np.testing.assert_allclose(W, W.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(W)
        assert np.all(eigenvalues >= -1e-10)

    def test_diagonal_is_one(self, rng: np.random.Generator) -> None:
        """Diagonal entries of RBF kernel should be 1.0."""
        Z = rng.standard_normal((5, 3))
        W = compute_kernel_on_landmarks(Z)
        np.testing.assert_allclose(np.diag(W), 1.0, atol=1e-12)

    def test_values_between_zero_and_one(self, rng: np.random.Generator) -> None:
        """RBF kernel values should be in [0, 1]."""
        Z = rng.standard_normal((5, 3))
        W = compute_kernel_on_landmarks(Z)
        assert np.all(W >= 0.0)
        assert np.all(W <= 1.0 + 1e-12)


class TestBuildWhiteningMap:
    """Tests for build_whitening_map."""

    def test_output_shapes(self, rng: np.random.Generator) -> None:
        """Output shapes should be consistent."""
        m_g = 10
        Z = rng.standard_normal((m_g, 4))
        W = compute_kernel_on_landmarks(Z)
        config = NumericsConfig(tau_eig=1e-3)
        U, eigenvalues, M_g, r_g = build_whitening_map(W, config)

        assert U.shape == (m_g, m_g)
        assert eigenvalues.shape == (m_g,)
        assert M_g.shape[0] == m_g
        assert M_g.shape[1] == r_g or (r_g == 0 and M_g.shape[1] == 1)

    def test_m_g_degenerate(self) -> None:
        """Degenerate W should produce r_g = 0."""
        W = np.zeros((5, 5))
        config = NumericsConfig(tau_eig=1e-3)
        _, _, M_g, r_g = build_whitening_map(W, config)
        assert r_g == 0
        assert M_g.shape == (5, 1)

    def test_non_square_raises(self) -> None:
        """Non-square W should raise ValueError."""
        with pytest.raises(ValueError, match="square"):
            build_whitening_map(np.ones((3, 4)), NumericsConfig())


class TestNystromGlobalBasis:
    """Tests for NystromGlobalBasis."""

    def test_from_landmarks(self, rng: np.random.Generator) -> None:
        """Should build from landmarks without error."""
        Z = rng.standard_normal((10, 4))
        config = NumericsConfig(tau_eig=1e-3)
        basis = NystromGlobalBasis.from_landmarks(Z, config)
        assert basis.Z is Z
        assert basis.r_g >= 0

    def test_from_data(self, rng: np.random.Generator) -> None:
        """Should select landmarks from data."""
        U_data = rng.standard_normal((100, 4))
        config = NumericsConfig(tau_eig=1e-3)
        basis = NystromGlobalBasis.from_data(U_data, m_g=10, config=config, rng=rng)
        assert basis.Z.shape == (10, 4)

    def test_build_features_batch_shape(self, rng: np.random.Generator) -> None:
        """Batch features should have shape (n, r_g)."""
        Z = rng.standard_normal((10, 4))
        config = NumericsConfig(tau_eig=1e-3)
        basis = NystromGlobalBasis.from_landmarks(Z, config)
        u = rng.standard_normal((20, 4))
        phi_g = basis.build_features(u)
        assert phi_g.shape == (20, basis.r_g)

    def test_build_features_single_shape(self, rng: np.random.Generator) -> None:
        """Single features should have shape (r_g,)."""
        Z = rng.standard_normal((10, 4))
        config = NumericsConfig(tau_eig=1e-3)
        basis = NystromGlobalBasis.from_landmarks(Z, config)
        u = rng.standard_normal(4)
        phi_g = basis.build_features(u)
        assert phi_g.shape == (basis.r_g,)

    def test_compute_kernel_vector_batch(self, rng: np.random.Generator) -> None:
        """Batch kernel vector should have shape (n, m_g)."""
        Z = rng.standard_normal((10, 4))
        config = NumericsConfig(tau_eig=1e-3)
        basis = NystromGlobalBasis.from_landmarks(Z, config)
        u = rng.standard_normal((20, 4))
        k_vec = basis.compute_kernel_vector(u)
        assert k_vec.shape == (20, 10)
        assert np.all(k_vec >= 0.0)
        assert np.all(k_vec <= 1.0 + 1e-12)
