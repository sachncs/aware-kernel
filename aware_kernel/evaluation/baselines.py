"""Baseline models for comparison against aware-kernel.

Each baseline exposes a simple ``fit(X, y)`` / ``predict(X)`` interface so the
experiment runner can treat them uniformly.  All baselines share the same
ridge solver infrastructure as aware-kernel, isolating the contribution of
the hybrid feature pipeline.

Three baselines are provided, ordered by complexity:

1. **RidgeBaseline**: Ridge regression in the original input space.
   No kernel approximation — serves as a lower bound.
2. **NystromRidgeBaseline**: Nystr\"om kernel ridge with a fixed global
   basis (no local corrective, no refresh).  Isolates the contribution
   of continuous + discrete adaptation.
3. **RandomFeatureBaseline**: Random Fourier features (Rahimi & Recht,
   2007) with i.i.d. Gaussian frequencies.  A standard kernel
   approximation baseline.

All baselines use ``DirectRidgeSolver`` (Cholesky) for consistency with
the aware-kernel solver path.
"""

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.solver.ridge import DirectRidgeSolver


class RidgeBaseline:
    """Standard ridge regression in the original feature space.

    Solves ``w = argmin ||Xw - y||^2 + lambda * ||w||^2`` directly in
    the input space without any kernel feature mapping.  This serves as
    a lower bound baseline — any kernel method should outperform this
    when the true function has non-trivial kernel structure.

    Uses ``DirectRidgeSolver`` (Cholesky decomposition) for the solve.
    """

    def __init__(self, lambda_reg: float = 1e-3) -> None:
        """Initialize baseline.

        Args:
            lambda_reg: Ridge regularization parameter.
        """
        self.lambda_reg = lambda_reg
        self.solver = DirectRidgeSolver(lambda_reg=lambda_reg)
        self.w: Array | None = None

    def fit(self, X: Array, y: Array) -> None:
        """Fit ridge regression.

        Args:
            X: Training inputs of shape ``(n, d)``.
            y: Training targets of shape ``(n,)``.
        """
        self.w = self.solver.solve(X, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape ``(n, d)``.

        Returns:
            Predictions of shape ``(n,)``.

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self.w is None:
            raise RuntimeError("RidgeBaseline must be fitted before predict")
        return X @ self.w


class NystromRidgeBaseline:
    """Standard Nystr\"om ridge regression without local corrective or refresh.

    Constructs a fixed Nystr\"om global basis from ``m_g`` landmarks
    selected via k-means++, applies spectral whitening, and solves ridge
    regression in the resulting feature space.  Unlike aware-kernel, this
    baseline has **no local corrective features**, **no refresh mechanism**,
    and **no learnable projection ``R``**.

    This baseline isolates the contribution of the global Nystr\"om basis
    alone.  The gap between this baseline and aware-kernel quantifies
    the value of the hybrid continuous-discrete pipeline.
    """

    def __init__(
        self,
        m_g: int = 512,
        lambda_reg: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        """Initialize baseline.

        Args:
            m_g: Number of landmarks.
            lambda_reg: Ridge regularization parameter.
            seed: Random seed for landmark selection.
        """
        self.m_g = m_g
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.basis: NystromGlobalBasis | None = None
        self.w: Array | None = None
        self.solver = DirectRidgeSolver(lambda_reg=lambda_reg)

    def fit(self, X: Array, y: Array) -> None:
        """Fit Nystr\"om ridge regression.

        Args:
            X: Training inputs of shape ``(n, d)``.
            y: Training targets of shape ``(n,)``.
        """
        rng = np.random.default_rng(self.seed)
        from aware_kernel.aware.config import NumericsConfig

        config = NumericsConfig()
        self.basis = NystromGlobalBasis.from_data(
            U_data=X,
            m_g=min(self.m_g, X.shape[0]),
            config=config,
            rng=rng,
        )
        phi_g = self.basis.build_features(X)
        self.w = self.solver.solve(phi_g, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape ``(n, d)``.

        Returns:
            Predictions of shape ``(n,)``.

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self.basis is None or self.w is None:
            raise RuntimeError("NystromRidgeBaseline must be fitted before predict")
        phi_g = self.basis.build_features(X)
        return phi_g @ self.w


class RandomFeatureBaseline:
    """Random Fourier Features baseline for kernel ridge regression.

    Approximates an RBF kernel using random Fourier features (Rahimi &
    Recht, 2007):

        ``phi(x) = sqrt(2/D) * cos(X @ omega + b)``

    where ``omega ~ N(0, gamma * I)`` and ``b ~ Uniform(0, 2*pi)``.
    The number of features ``D`` controls the approximation quality.

    This baseline uses a fixed random projection (no learnable ``R``),
    no refresh, and no local features.  It represents a standard kernel
    approximation approach that aware-kernel aims to improve upon.
    """

    def __init__(
        self,
        n_features: int = 1000,
        gamma: float = 1.0,
        lambda_reg: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        """Initialize baseline.

        Args:
            n_features: Number of random Fourier features.
            gamma: RBF kernel bandwidth parameter (scale = 1 / sqrt(2*gamma)).
            lambda_reg: Ridge regularization parameter.
            seed: Random seed.
        """
        self.n_features = n_features
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.seed = seed
        self.omega: Array | None = None
        self.b: Array | None = None
        self.w: Array | None = None
        self.solver = DirectRidgeSolver(lambda_reg=lambda_reg)

    def _build_features(self, X: Array) -> Array:
        """Build random Fourier features.

        Applies the random Fourier feature map ``phi(x) = sqrt(2/D) * cos(X @ omega + b)``
        where ``omega`` and ``b`` are sampled once during ``fit`` and reused here.

        Args:
            X: Inputs of shape ``(n, d)``.

        Returns:
            Random Fourier features of shape ``(n, n_features)``.
        """
        if self.omega is None or self.b is None:
            raise RuntimeError("RandomFeatureBaseline parameters not initialized")
        z = X @ self.omega + self.b
        result: Array = np.cos(z) * np.sqrt(2.0 / self.n_features)
        return result

    def fit(self, X: Array, y: Array) -> None:
        """Fit random feature ridge regression.

        Args:
            X: Training inputs of shape ``(n, d)``.
            y: Training targets of shape ``(n,)``.
        """
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        scale = np.sqrt(2.0 * self.gamma)
        self.omega = rng.standard_normal((d, self.n_features)) / scale
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        phi = self._build_features(X)
        self.w = self.solver.solve(phi, y)

    def predict(self, X: Array) -> Array:
        """Predict on new data.

        Args:
            X: Inputs of shape ``(n, d)``.

        Returns:
            Predictions of shape ``(n,)``.

        Raises:
            RuntimeError: If fit has not been called.
        """
        if self.w is None:
            raise RuntimeError("RandomFeatureBaseline must be fitted before predict")
        phi = self._build_features(X)
        return phi @ self.w
