"""Unit tests for the public sklearn-compatible API."""

import numpy as np
import pytest

from aware_kernel import AwareKernelEstimator


class TestAwareKernelEstimator:
    """Tests for AwareKernelEstimator."""

    def _make_data(self, rng: np.random.Generator, n: int = 100, d: int = 4) -> tuple:
        """Helper to create synthetic data."""
        X = rng.standard_normal((n, d))
        true_w = rng.standard_normal(d)
        y = X @ true_w + 0.1 * rng.standard_normal(n)
        return X, y

    def test_fit(self, rng: np.random.Generator) -> None:
        """Should fit without error."""
        X, y = self._make_data(rng)
        model = AwareKernelEstimator(
            embedding_dim=4,
            m_g=16,
            m_l=4,
            lambda_reg=1e-2,
            max_steps=5,
            eval_freq=5,
            k_local=3,
            seed=42,
        )
        model.fit(X, y)
        assert hasattr(model, "state_")
        assert model.state_ is not None
        assert model.state_.w is not None

    def test_predict(self, rng: np.random.Generator) -> None:
        """Should predict after fitting."""
        X, y = self._make_data(rng)
        model = AwareKernelEstimator(
            embedding_dim=4,
            m_g=16,
            m_l=4,
            lambda_reg=1e-2,
            max_steps=5,
            eval_freq=5,
            k_local=3,
            seed=42,
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_score(self, rng: np.random.Generator) -> None:
        """Should return a finite R^2 score."""
        X, y = self._make_data(rng)
        model = AwareKernelEstimator(
            embedding_dim=4,
            m_g=16,
            m_l=4,
            lambda_reg=1e-2,
            max_steps=5,
            eval_freq=5,
            k_local=3,
            seed=42,
        )
        model.fit(X, y)
        r2 = model.score(X, y)
        assert np.isfinite(r2)
        assert r2 <= 1.0

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit should raise."""
        model = AwareKernelEstimator(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42
        )
        with pytest.raises((RuntimeError, ValueError)):
            model.predict(np.zeros((5, 3)))

    def test_get_params(self) -> None:
        """Sklearn get_params should return hyperparameters."""
        model = AwareKernelEstimator(embedding_dim=8, m_g=32, seed=42)
        params = model.get_params()
        assert params["embedding_dim"] == 8
        assert params["m_g"] == 32
        assert params["seed"] == 42

    def test_set_params(self) -> None:
        """Sklearn set_params should update hyperparameters."""
        model = AwareKernelEstimator()
        model.set_params(embedding_dim=16, lambda_reg=1e-1)
        assert model.embedding_dim == 16
        assert model.lambda_reg == 1e-1

    def test_streamed_mode(self, rng: np.random.Generator) -> None:
        """Should work with streamed memory mode."""
        X, y = self._make_data(rng)
        model = AwareKernelEstimator(
            embedding_dim=4,
            m_g=16,
            m_l=4,
            lambda_reg=1e-2,
            memory_mode="streamed",
            max_steps=5,
            eval_freq=5,
            k_local=3,
            seed=42,
        )
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)
