"""Unit tests for aware_kernel.inference modules."""

import numpy as np
import pytest

from aware_kernel.inference.predictor import Predictor, predict_mean, predict_variance


class TestPredictMean:
    """Tests for predict_mean."""

    def test_single(self) -> None:
        """Single query should return float."""
        phi = np.array([1.0, 2.0, 3.0])
        w = np.array([0.5, 1.0, 1.5])
        result = predict_mean(phi, w)
        assert isinstance(result, float)
        assert result == pytest.approx(7.0, abs=1e-10)

    def test_batch(self, rng: np.random.Generator) -> None:
        """Batch query should return array."""
        phi = rng.standard_normal((10, 5))
        w = rng.standard_normal(5)
        result = predict_mean(phi, w)
        assert result.shape == (10,)


class TestPredictVariance:
    """Tests for predict_variance."""

    def test_single_non_negative(self, rng: np.random.Generator) -> None:
        """Variance should be non-negative."""
        phi = rng.standard_normal(5)
        s_inv = np.eye(5) * 0.1
        result = predict_variance(phi, s_inv)
        assert result >= 0.0

    def test_batch_non_negative(self, rng: np.random.Generator) -> None:
        """Batch variance should be non-negative."""
        phi = rng.standard_normal((10, 5))
        s_inv = np.eye(5) * 0.1
        result = predict_variance(phi, s_inv)
        assert np.all(result >= 0.0)

    def test_clamping(self, rng: np.random.Generator) -> None:
        """Negative variances should be clamped to zero."""
        # Construct a case where self_term < correction
        phi = np.array([0.1, 0.1])
        s_inv = np.eye(2) * 10.0  # large inverse = large correction
        result = predict_variance(phi, s_inv)
        assert result == 0.0


class TestPredictor:
    """Tests for Predictor."""

    def test_mean_prediction(self, rng: np.random.Generator) -> None:
        """Predictor should produce mean predictions."""
        w = rng.standard_normal(5)
        predictor = Predictor(w=w)
        phi = rng.standard_normal((10, 5))
        result = predictor.predict(phi)
        assert result.shape == (10,)

    def test_variance_without_s_inv_raises(self, rng: np.random.Generator) -> None:
        """Variance prediction without s_inv should raise."""
        w = rng.standard_normal(5)
        predictor = Predictor(w=w)
        phi = rng.standard_normal(5)
        with pytest.raises(ValueError, match="s_inv must be set"):
            predictor.predict_variance(phi)

    def test_variance_with_s_inv(self, rng: np.random.Generator) -> None:
        """Variance prediction with s_inv should succeed."""
        w = rng.standard_normal(5)
        s_inv = np.eye(5) * 0.1
        predictor = Predictor(w=w, s_inv=s_inv)
        phi = rng.standard_normal((10, 5))
        result = predictor.predict_variance(phi)
        assert np.all(result >= 0.0)
