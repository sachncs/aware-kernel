"""Shared fixtures and hypothesis strategies for aware-kernel tests."""

import numpy as np
import pytest

from aware_kernel.aware.config import (
    MemoryMode,
    NumericsConfig,
    RefreshConfig,
    TrainingConfig,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def numerics_config() -> NumericsConfig:
    """Default numerics configuration for tests."""
    return NumericsConfig(
        tau_eig=1e-6,
        alpha_epsilon=1e-5,
        epsilon_c=1e-8,
        lambda_min=1e-6,
        eta_o=1e-4,
        beta=1e-3,
        kappa_threshold=1e12,
        precision="float64",
    )


@pytest.fixture
def refresh_config() -> RefreshConfig:
    """Default refresh configuration for tests."""
    return RefreshConfig(
        delta_hi=0.1,
        t_cool=50,
        t_warmup=10,
        gamma_cost=0.01,
        alpha_a=0.5,
        tau_local=0.1,
        k_local=5,
    )


@pytest.fixture
def training_config(
    numerics_config: NumericsConfig, refresh_config: RefreshConfig
) -> TrainingConfig:
    """Default training configuration for tests."""
    return TrainingConfig(
        embedding_dim=8,
        m_g=64,
        m_l=16,
        lambda_reg=1e-2,
        memory_mode=MemoryMode.CACHED,
        numerics=numerics_config,
        refresh=refresh_config,
        max_steps=100,
        eval_freq=5,
        seed=42,
    )


@pytest.fixture
def synthetic_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic regression dataset for quick tests.

    Returns:
        Tuple of (X, y) with X shape (200, 4) and y shape (200,).
    """
    n, d = 200, 4
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.1 * rng.standard_normal(n)
    return X, y
