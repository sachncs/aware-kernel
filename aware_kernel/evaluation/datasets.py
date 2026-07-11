"""Synthetic regression datasets for evaluation.

Provides reproducible dataset generators for benchmarking aware-kernel
against baselines.  Each generator returns data as NumPy arrays with
optional access to ground-truth weights (enabling noise-free evaluation
of model quality).

Four dataset types are provided, covering increasing difficulty:

1. **Linear**: ``y = Xw + noise`` — the simplest case where ridge
   regression is optimal; any kernel method should match this.
2. **Polynomial**: Univariate polynomial with additive noise — tests
   ability to capture nonlinear structure.
3. **High-dimensional**: Sparse true weights in a high-dimensional
   space — tests behavior when ``d >> n``.
4. **Heteroscedastic**: Input-dependent noise — tests robustness to
   non-stationary noise.

All generators accept an ``np.random.Generator`` for reproducibility.
"""

import numpy as np

from aware_kernel.aware.types import Array


def make_linear_regression(
    rng: np.random.Generator,
    n_samples: int = 500,
    n_features: int = 10,
    noise: float = 0.1,
) -> tuple[Array, Array, Array]:
    """Generate a synthetic linear regression dataset.

    Produces ``y = X @ true_w + noise * epsilon`` where ``X`` and
    ``true_w`` are drawn from ``N(0, 1)`` and ``epsilon ~ N(0, 1)``.
    This is the easiest benchmark — a well-tuned ridge regression should
    recover ``true_w`` nearly exactly for large ``n``.

    Args:
        rng: NumPy random generator for reproducibility.
        n_samples: Number of samples.
        n_features: Number of input features.
        noise: Standard deviation of additive Gaussian noise.

    Returns:
        Tuple of ``(X, y, true_weights)`` where ``X`` has shape
        ``(n_samples, n_features)``, ``y`` has shape ``(n_samples,)``,
        and ``true_weights`` has shape ``(n_features,)``.
    """
    X = rng.standard_normal((n_samples, n_features))
    true_w = rng.standard_normal(n_features)
    y = X @ true_w + noise * rng.standard_normal(n_samples)
    return X, y, true_w


def make_polynomial_regression(
    rng: np.random.Generator,
    n_samples: int = 500,
    degree: int = 3,
    noise: float = 0.1,
) -> tuple[Array, Array]:
    """Generate a synthetic univariate polynomial regression dataset.

    Produces ``y = sum_p coeffs[p] * x^p + noise`` where ``x`` is
    drawn from ``N(0, 1)`` and ``coeffs`` are i.i.d. ``N(0, 1)``.
    The polynomial structure requires nonlinear function approximation,
    making this a basic test of kernel methods.

    Args:
        rng: NumPy random generator for reproducibility.
        n_samples: Number of samples.
        degree: Polynomial degree (highest power).
        noise: Standard deviation of additive Gaussian noise.

    Returns:
        Tuple of ``(X, y)`` where ``X`` has shape ``(n_samples, 1)``
        and ``y`` has shape ``(n_samples,)``.
    """
    X = rng.standard_normal((n_samples, 1))
    coeffs = rng.standard_normal(degree + 1)
    y = np.zeros(n_samples)
    for p in range(degree + 1):
        y += coeffs[p] * (X[:, 0] ** p)
    y += noise * rng.standard_normal(n_samples)
    return X, y


def make_high_dim_regression(
    rng: np.random.Generator,
    n_samples: int = 500,
    n_features: int = 100,
    n_informative: int = 10,
    noise: float = 0.1,
) -> tuple[Array, Array, Array]:
    """Generate a high-dimensional regression with sparse true weights.

    Produces ``y = X @ true_w + noise * epsilon`` where ``true_w`` has
    exactly ``n_informative`` nonzero entries chosen uniformly at random.
    This tests behavior in the ``d >> n`` regime and verifies that the
    ridge penalty effectively regularizes irrelevant dimensions.

    Args:
        rng: NumPy random generator for reproducibility.
        n_samples: Number of samples.
        n_features: Total number of input features.
        n_informative: Number of nonzero entries in ``true_w``.
        noise: Standard deviation of additive Gaussian noise.

    Returns:
        Tuple of ``(X, y, true_weights)``.
    """
    X = rng.standard_normal((n_samples, n_features))
    true_w = np.zeros(n_features)
    informative_indices = rng.choice(n_features, size=n_informative, replace=False)
    true_w[informative_indices] = rng.standard_normal(n_informative)
    y = X @ true_w + noise * rng.standard_normal(n_samples)
    return X, y, true_w


def make_heteroscedastic_regression(
    rng: np.random.Generator,
    n_samples: int = 500,
    noise_base: float = 0.05,
) -> tuple[Array, Array]:
    """Generate a regression dataset with input-dependent noise.

    Produces ``y = sin(2*pi*x) + noise_base * (1 + |x|) * epsilon``.
    The noise standard deviation grows with ``|x|``, creating
    heteroscedasticity that penalizes models which assume uniform noise.
    This tests robustness to non-stationary noise distributions.

    Args:
        rng: NumPy random generator for reproducibility.
        n_samples: Number of samples.
        noise_base: Base noise scale (multiplied by ``1 + |x|``).

    Returns:
        Tuple of ``(X, y)`` where ``X`` has shape ``(n_samples, 1)``
        and ``y`` has shape ``(n_samples,)``.
    """
    X = rng.standard_normal((n_samples, 1))
    signal = np.sin(2.0 * np.pi * X[:, 0])
    noise_std = noise_base * (1.0 + np.abs(X[:, 0]))
    y = signal + noise_std * rng.standard_normal(n_samples)
    return X, y


def split_train_test(
    X: Array,
    y: Array,
    test_size: float = 0.2,
    rng: np.random.Generator | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Split data into train and test sets.

    Uses a simple index-based split (no stratification) appropriate for
    regression.  The split is deterministic when ``rng`` is provided.

    Args:
        X: Input matrix of shape ``(n, d)``.
        y: Target vector of shape ``(n,)``.
        test_size: Fraction of data to hold out for testing (0 to 1).
        rng: Optional random generator for shuffling before splitting.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)``.
    """
    n = X.shape[0]
    n_test = int(n * test_size)
    perm = rng.permutation(n) if rng is not None else np.arange(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
