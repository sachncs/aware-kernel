"""Numerical correctness tests for calibration stability.

These tests verify the guarantees from Section 7 of the method blueprint.
"""

import numpy as np
import pytest

from aware_kernel.fusion.calibration import (
    compute_global_calibration,
    compute_local_calibration,
)


class TestCalibrationStability:
    """Tests for calibration stability under edge cases."""

    def test_global_calibration_prevents_collapse(self, rng: np.random.Generator) -> None:
        """c_g should remain positive even when all features are zero."""
        phi_g = np.zeros((10, 5))
        c_g = compute_global_calibration(phi_g, epsilon_c=1e-8)
        assert c_g > 0.0
        assert c_g == pytest.approx(np.sqrt(1e-8), abs=1e-12)

    def test_local_calibration_prevents_collapse(self, rng: np.random.Generator) -> None:
        """c_l should remain positive even when all features are zero."""
        phi_l = np.zeros((10, 5))
        c_l = compute_local_calibration(phi_l, epsilon_c=1e-8)
        assert c_l > 0.0
        assert c_l == pytest.approx(np.sqrt(1e-8), abs=1e-12)

    def test_expected_norm_after_calibration(self, rng: np.random.Generator) -> None:
        """Calibrated features should have unit expected squared norm."""
        phi_g = rng.standard_normal((100, 5))
        c_g = compute_global_calibration(phi_g, epsilon_c=1e-8)
        bar_phi_g = phi_g / c_g
        expected_norm = np.mean(np.sum(bar_phi_g**2, axis=1))
        # For standard normal features, c_g^2 ≈ d = 5, so E[||bar_phi||^2] ≈ 1
        np.testing.assert_allclose(expected_norm, 1.0, rtol=0.5)
