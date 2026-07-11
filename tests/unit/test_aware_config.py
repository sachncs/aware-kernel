"""Unit tests for aware_kernel.aware.config."""

import pytest

from aware_kernel.aware.config import (
    MemoryMode,
    NumericsConfig,
    RefreshConfig,
    TrainingConfig,
)


class TestNumericsConfig:
    """Tests for NumericsConfig validation."""

    def test_default_creation(self) -> None:
        """Default NumericsConfig should instantiate without error."""
        cfg = NumericsConfig()
        assert cfg.tau_eig == 1e-6
        assert cfg.alpha_epsilon == 1e-5
        assert cfg.epsilon_c == 1e-8
        assert cfg.lambda_min == 1e-6
        assert cfg.eta_o == 1e-4
        assert cfg.beta == 1e-3
        assert cfg.kappa_threshold == 1e12
        assert cfg.precision == "float64"

    def test_precision_validation(self) -> None:
        """Invalid precision should raise ValueError."""
        with pytest.raises(ValueError, match="precision must be"):
            NumericsConfig(precision="float16")

    def test_custom_values(self) -> None:
        """Custom values should be accepted."""
        cfg = NumericsConfig(tau_eig=1e-4, epsilon_c=1e-6)
        assert cfg.tau_eig == 1e-4
        assert cfg.epsilon_c == 1e-6


class TestRefreshConfig:
    """Tests for RefreshConfig."""

    def test_default_creation(self) -> None:
        """Default RefreshConfig should instantiate."""
        cfg = RefreshConfig()
        assert cfg.delta_hi == 0.1
        assert cfg.t_cool == 50
        assert cfg.t_warmup == 10
        assert cfg.gamma_cost == 0.01
        assert cfg.alpha_a == 0.5
        assert cfg.tau_local == 0.1
        assert cfg.k_local == 5


class TestTrainingConfig:
    """Tests for TrainingConfig validation."""

    def test_default_creation(self) -> None:
        """Default TrainingConfig should instantiate."""
        cfg = TrainingConfig()
        assert cfg.embedding_dim == 64
        assert cfg.m_g == 512
        assert cfg.m_l == 128
        assert cfg.lambda_reg == 1e-3
        assert cfg.memory_mode == MemoryMode.CACHED

    def test_m_l_constraint(self) -> None:
        """m_l must be <= 0.25 * m_g."""
        with pytest.raises(ValueError, match="must be <= 0.25"):
            TrainingConfig(m_g=100, m_l=30)

    def test_lambda_reg_constraint(self) -> None:
        """lambda_reg must be >= lambda_min from numerics."""
        numerics = NumericsConfig(lambda_min=1e-2)
        with pytest.raises(ValueError, match="lambda_reg"):
            TrainingConfig(numerics=numerics, lambda_reg=1e-4)

    def test_valid_local_rank(self) -> None:
        """Valid m_l should pass."""
        cfg = TrainingConfig(m_g=100, m_l=25)
        assert cfg.m_l == 25


class TestMemoryMode:
    """Tests for MemoryMode enum."""

    def test_members(self) -> None:
        """Enum members should be accessible."""
        assert MemoryMode.CACHED.value == "cached"
        assert MemoryMode.STREAMED.value == "streamed"
