"""Unit tests for aware_kernel.fusion modules."""

import numpy as np
import pytest

from aware_kernel.fusion.calibration import (
    calibrate_global_features,
    calibrate_local_features,
    compute_global_calibration,
    compute_local_calibration,
)
from aware_kernel.fusion.gate import compute_gate, fuse_features, sigmoid, split_fused_features
from aware_kernel.fusion.builder import FusedFeatureBuilder


class TestSigmoid:
    """Tests for sigmoid."""

    def test_zero(self) -> None:
        """sigmoid(0) should be 0.5."""
        assert sigmoid(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_large_positive(self) -> None:
        """Large positive input should approach 1."""
        assert sigmoid(10.0) > 0.99

    def test_large_negative(self) -> None:
        """Large negative input should approach 0."""
        assert sigmoid(-10.0) < 0.01

    def test_between_zero_and_one(self) -> None:
        """Output should always be in (0, 1)."""
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            result = sigmoid(x)
            assert 0.0 < result < 1.0


class TestComputeGate:
    """Tests for compute_gate."""

    def test_default(self) -> None:
        """compute_gate(0.0) should be 0.5."""
        assert compute_gate(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_positive(self) -> None:
        """Positive a should give rho > 0.5."""
        assert compute_gate(1.0) > 0.5

    def test_negative(self) -> None:
        """Negative a should give rho < 0.5."""
        assert compute_gate(-1.0) < 0.5


class TestFuseFeatures:
    """Tests for fuse_features."""

    def test_batch_shape(self, rng: np.random.Generator) -> None:
        """Fused batch features should have shape (n, r_g + m_l)."""
        phi_g = rng.standard_normal((10, 4))
        phi_l = rng.standard_normal((10, 3))
        fused = fuse_features(phi_g, phi_l, rho=0.5)
        assert fused.shape == (10, 7)

    def test_single_shape(self) -> None:
        """Fused single features should have shape (r_g + m_l,)."""
        phi_g = np.array([1.0, 2.0])
        phi_l = np.array([3.0, 4.0, 5.0])
        fused = fuse_features(phi_g, phi_l, rho=0.25)
        assert fused.shape == (5,)

    def test_rho_zero(self, rng: np.random.Generator) -> None:
        """rho=0 should give only local features."""
        phi_g = rng.standard_normal((5, 4))
        phi_l = rng.standard_normal((5, 3))
        fused = fuse_features(phi_g, phi_l, rho=0.0)
        g_part, l_part = split_fused_features(fused, r_g=4)
        np.testing.assert_allclose(g_part, 0.0, atol=1e-12)
        np.testing.assert_allclose(l_part, phi_l, atol=1e-12)

    def test_rho_one(self, rng: np.random.Generator) -> None:
        """rho=1 should give only global features."""
        phi_g = rng.standard_normal((5, 4))
        phi_l = rng.standard_normal((5, 3))
        fused = fuse_features(phi_g, phi_l, rho=1.0)
        g_part, l_part = split_fused_features(fused, r_g=4)
        np.testing.assert_allclose(g_part, phi_g, atol=1e-12)
        np.testing.assert_allclose(l_part, 0.0, atol=1e-12)

    def test_invalid_rho_raises(self) -> None:
        """rho outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="rho must be"):
            fuse_features(np.zeros(2), np.zeros(3), rho=-0.1)
        with pytest.raises(ValueError, match="rho must be"):
            fuse_features(np.zeros(2), np.zeros(3), rho=1.1)


class TestSplitFusedFeatures:
    """Tests for split_fused_features."""

    def test_batch(self, rng: np.random.Generator) -> None:
        """Should correctly split batch features."""
        fused = rng.standard_normal((10, 7))
        g, l = split_fused_features(fused, r_g=4)
        assert g.shape == (10, 4)
        assert l.shape == (10, 3)

    def test_single(self) -> None:
        """Should correctly split single features."""
        fused = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g, l = split_fused_features(fused, r_g=2)
        np.testing.assert_array_equal(g, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(l, np.array([3.0, 4.0, 5.0]))


class TestComputeGlobalCalibration:
    """Tests for compute_global_calibration."""

    def test_positive(self, rng: np.random.Generator) -> None:
        """c_g should be positive."""
        phi_g = rng.standard_normal((20, 5))
        c_g = compute_global_calibration(phi_g, epsilon_c=1e-8)
        assert c_g > 0.0

    def test_zero_features(self) -> None:
        """Zero features should yield c_g = sqrt(epsilon_c)."""
        phi_g = np.zeros((10, 5))
        c_g = compute_global_calibration(phi_g, epsilon_c=1e-4)
        np.testing.assert_allclose(c_g, np.sqrt(1e-4), atol=1e-12)


class TestComputeLocalCalibration:
    """Tests for compute_local_calibration."""

    def test_positive(self, rng: np.random.Generator) -> None:
        """c_l should be positive."""
        phi_l = rng.standard_normal((20, 5))
        c_l = compute_local_calibration(phi_l, epsilon_c=1e-8)
        assert c_l > 0.0

    def test_zero_features(self) -> None:
        """Zero features should yield c_l = sqrt(epsilon_c)."""
        phi_l = np.zeros((10, 5))
        c_l = compute_local_calibration(phi_l, epsilon_c=1e-4)
        np.testing.assert_allclose(c_l, np.sqrt(1e-4), atol=1e-12)


class TestCalibrateGlobalFeatures:
    """Tests for calibrate_global_features."""

    def test_scale(self, rng: np.random.Generator) -> None:
        """Calibration should scale features by 1/c_g."""
        phi_g = rng.standard_normal((10, 5))
        c_g = 2.0
        calibrated = calibrate_global_features(phi_g, c_g)
        np.testing.assert_allclose(calibrated, phi_g / 2.0)


class TestCalibrateLocalFeatures:
    """Tests for calibrate_local_features."""

    def test_scale(self, rng: np.random.Generator) -> None:
        """Calibration should scale features by 1/c_l."""
        phi_l = rng.standard_normal((10, 5))
        c_l = 3.0
        calibrated = calibrate_local_features(phi_l, c_l)
        np.testing.assert_allclose(calibrated, phi_l / 3.0)


class TestFusedFeatureBuilder:
    """Tests for FusedFeatureBuilder."""

    def test_from_features(self, rng: np.random.Generator) -> None:
        """Should build from feature matrices."""
        phi_g = rng.standard_normal((20, 4))
        phi_l = rng.standard_normal((20, 3))
        builder = FusedFeatureBuilder.from_features(phi_g, phi_l, a=0.0, epsilon_c=1e-8)
        assert builder.c_g > 0.0
        assert builder.c_l > 0.0
        assert builder.rho == pytest.approx(0.5, abs=1e-10)

    def test_build_batch(self, rng: np.random.Generator) -> None:
        """Build should produce correct shape."""
        phi_g = rng.standard_normal((20, 4))
        phi_l = rng.standard_normal((20, 3))
        builder = FusedFeatureBuilder.from_features(phi_g, phi_l, a=0.0, epsilon_c=1e-8)
        fused = builder.build(phi_g[:5], phi_l[:5])
        assert fused.shape == (5, 7)

    def test_build_single(self, rng: np.random.Generator) -> None:
        """Build should handle single sample."""
        phi_g = rng.standard_normal(4)
        phi_l = rng.standard_normal(3)
        builder = FusedFeatureBuilder(c_g=1.0, c_l=1.0, rho=0.5)
        fused = builder.build(phi_g, phi_l)
        assert fused.shape == (7,)
