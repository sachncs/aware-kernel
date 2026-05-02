"""Synthetic regression datasets for evaluation.

Implements standard synthetic benchmarks used to validate the
aware-kernel method against baselines.
"""

from typing import Optional, Tuple

import numpy as np

from aware_kernel.aware.types import Array


def make_linear_regression(
    rng: np.random.Generator,
    n_samples: int = 500,
    n_features: int = 10,
    noise: float = 0.1,
) -> Tuple[Array, Array, Array]:
    """Generate a synthetic linear regression dataset.

    Args:
        rng: NumPy random generator.
        n_samples: Number of samples.
        n_features: Number of input features.
        noise: Standard deviation of Gaussian noise.

    Returns:
        Tuple of (X, y, true_weights) where X has shape (n_samples, n_features),
        y has shape (n_samples,), and true_weights has shape (n_features,).
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
) -> Tuple[Array, Array]:
    """Generate a synthetic univariate polynomial regression dataset.

    Args:
        rng: NumPy random generator.
        n_samples: Number of samples.
        degree: Polynomial degree.
        noise: Standard deviation of Gaussian noise.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, 1) and y has shape
        (n_samples,).
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
) -> Tuple[Array, Array, Array]:
    """Generate a high-dimensional regression with sparse true weights.

    Args:
        rng: NumPy random generator.
        n_samples: Number of samples.
        n_features: Number of input features.
        n_informative: Number of nonzero true weights.
        noise: Standard deviation of Gaussian noise.

    Returns:
        Tuple of (X, y, true_weights).
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
) -> Tuple[Array, Array]:
    """Generate a regression dataset with input-dependent noise.

    Args:
        rng: NumPy random generator.
        n_samples: Number of samples.
        noise_base: Base noise scale.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, 1).
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
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array, Array, Array]:
    """Split data into train and test sets.

    Args:
        X: Input matrix of shape (n, d).
        y: Target vector of shape (n,).
        test_size: Fraction of data to use for testing.
        rng: Optional random generator for shuffling.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    n = X.shape[0]
    n_test = int(n * test_size)
    if rng is not None:
        perm = rng.permutation(n)
    else:
        perm = np.arange(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
