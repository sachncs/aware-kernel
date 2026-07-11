"""Numerical correctness tests for orthogonalization.

These tests verify the algebraic guarantees from Section 6 of the method blueprint.
"""

import numpy as np

from aware_kernel.local_corrective.orthogonalizer import (
    compute_orthogonalization_matrix,
    orthogonalize_local_features,
)


class TestOrthogonalizationProperties:
    """Tests for orthogonalization correctness."""

    def test_projection_is_symmetric(self, rng: np.random.Generator) -> None:
        """P_g should be symmetric."""
        phi_g = rng.standard_normal((20, 5))
        P_g = compute_orthogonalization_matrix(phi_g, eta_o=1e-4)
        np.testing.assert_allclose(P_g, P_g.T, atol=1e-10)

    def test_phi_l_perp_in_nullspace(self, rng: np.random.Generator) -> None:
        """P_g @ Phi_l_perp should be approximately zero."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        P_g = compute_orthogonalization_matrix(phi_g, eta_o=1e-4)
        projected = P_g @ phi_l_perp
        np.testing.assert_allclose(projected, 0.0, atol=1e-4)

    def test_reconstruction(self, rng: np.random.Generator) -> None:
        """Phi_l = P_g Phi_l + Phi_l_perp should hold."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        P_g = compute_orthogonalization_matrix(phi_g, eta_o=1e-4)
        reconstructed = P_g @ phi_l + phi_l_perp
        np.testing.assert_allclose(reconstructed, phi_l, atol=1e-6)

    def test_frobenius_ratio(self, rng: np.random.Generator) -> None:
        """Cross-term Frobenius norm should be negligible relative to traces."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 8))
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        cross = phi_g.T @ phi_l_perp
        denom = np.trace(phi_g.T @ phi_g) * np.trace(phi_l_perp.T @ phi_l_perp)
        ratio = float(np.linalg.norm(cross, "fro") ** 2 / denom)
        assert ratio < 1e-8
