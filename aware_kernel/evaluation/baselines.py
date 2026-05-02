"""Baseline models for comparison against aware-kernel.

Each baseline exposes a simple fit/predict interface so the experiment
runner can treat them uniformly.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.solver.ridge import DirectRidgeSolver


class RidgeBaseline:
    """Standard ridge regression in the original feature space."""

    def __init__(self, lambda_reg: float = 1e-3) -> None:
        """Initialize baseline.

        Args:
            lambda_reg: Ridge regularization parameter.
        """
        self._lambda_reg = lambda_reg
        self._solver = DirectRidgeSolver(lambda_reg=lambda_reg)
        self._w: Optional[Array] = None

    def fit(self, X: Array, y: Array) -> None:
        """Fit ridge regression.

        Args:
            X: Training inputs of shape (n, d).
            y: Training targets of shape (n,).
        """
        self._w = self._solver.solve(X, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape (n, d).

        Returns:
            Predictions of shape (n,).

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self._w is None:
            raise RuntimeError("RidgeBaseline must be fitted before predict")
        return X @ self._w


class NystromRidgeBaseline:
    """Standard Nystr\"om ridge regression without local corrective or refresh."""

    def __init__(
        self,
        m_g: int = 512,
        lambda_reg: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize baseline.

        Args:
            m_g: Number of landmarks.
            lambda_reg: Ridge regularization parameter.
            seed: Random seed for landmark selection.
        """
        self._m_g = m_g
        self._lambda_reg = lambda_reg
        self._seed = seed
        self._basis: Optional[NystromGlobalBasis] = None
        self._w: Optional[Array] = None
        self._solver = DirectRidgeSolver(lambda_reg=lambda_reg)

    def fit(self, X: Array, y: Array) -> None:
        """Fit Nystr\"om ridge regression.

        Args:
            X: Training inputs of shape (n, d).
            y: Training targets of shape (n,).
        """
        rng = np.random.default_rng(self._seed)
        from aware_kernel.aware.config import NumericsConfig

        config = NumericsConfig()
        self._basis = NystromGlobalBasis.from_data(
            U_data=X,
            m_g=min(self._m_g, X.shape[0]),
            config=config,
            rng=rng,
        )
        phi_g = self._basis.build_features(X)
        self._w = self._solver.solve(phi_g, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape (n, d).

        Returns:
            Predictions of shape (n,).

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self._basis is None or self._w is None:
            raise RuntimeError("NystromRidgeBaseline must be fitted before predict")
        phi_g = self._basis.build_features(X)
        return phi_g @ self._w


class RandomFeatureBaseline:
    """Random Fourier Features baseline for kernel ridge regression."""

    def __init__(
        self,
        n_features: int = 1000,
        gamma: float = 1.0,
        lambda_reg: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize baseline.

        Args:
            n_features: Number of random Fourier features.
            gamma: RBF kernel bandwidth parameter (scale = 1 / sqrt(2*gamma)).
            lambda_reg: Ridge regularization parameter.
            seed: Random seed.
        """
        self._n_features = n_features
        self._gamma = gamma
        self._lambda_reg = lambda_reg
        self._seed = seed
        self._omega: Optional[Array] = None
        self._b: Optional[Array] = None
        self._w: Optional[Array] = None
        self._solver = DirectRidgeSolver(lambda_reg=lambda_reg)

    def _build_features(self, X: Array) -> Array:
        """Build random Fourier features.

        Args:
            X: Inputs of shape (n, d).

        Returns:
            Random Fourier features of shape (n, n_features).
        """
        if self._omega is None or self._b is None:
            raise RuntimeError("RandomFeatureBaseline parameters not initialized")
        z = X @ self._omega + self._b
        return np.cos(z) * np.sqrt(2.0 / self._n_features)

    def fit(self, X: Array, y: Array) -> None:
        """Fit random feature ridge regression.

        Args:
            X: Training inputs of shape (n, d).
            y: Training targets of shape (n,).
        """
        rng = np.random.default_rng(self._seed)
        d = X.shape[1]
        scale = np.sqrt(2.0 * self._gamma)
        self._omega = rng.standard_normal((d, self._n_features)) / scale
        self._b = rng.uniform(0.0, 2.0 * np.pi, size=self._n_features)
        phi = self._build_features(X)
        self._w = self._solver.solve(phi, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape (n, d).

        Returns:
            Predictions of shape (n,).

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self._w is None:
            raise RuntimeError("RandomFeatureBaseline must be fitted before predict")
        phi = self._build_features(X)
        return phi @ self._w
