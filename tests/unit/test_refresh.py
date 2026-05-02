"""Unit tests for aware_kernel.refresh modules."""

import numpy as np
import pytest

from aware_kernel.aware.config import (
    AblationConfig,
    NumericsConfig,
    RefreshConfig,
    TrainingConfig,
)
from aware_kernel.aware.exceptions import BudgetExceededError
from aware_kernel.aware.state import ContinuousState, DiscreteState, FullState
from aware_kernel.refresh.budget import BudgetAccountant
from aware_kernel.refresh.controller import should_refresh, transition_state
from aware_kernel.refresh.drift import compute_drift
from aware_kernel.refresh.pipeline import run_refresh_pipeline


class TestComputeDrift:
    """Tests for compute_drift."""

    def test_identical(self) -> None:
        """Identical matrices should have zero drift."""
        R = np.eye(5)
        assert compute_drift(R, R) == 0.0

    def test_zero_reference(self) -> None:
        """Zero reference should return zero drift."""
        current = np.eye(5)
        reference = np.zeros((5, 5))
        assert compute_drift(current, reference) == 0.0

    def test_positive_drift(self, rng: np.random.Generator) -> None:
        """Different matrices should have positive drift."""
        current = rng.standard_normal((5, 5))
        reference = rng.standard_normal((5, 5))
        drift = compute_drift(current, reference)
        assert drift >= 0.0


class TestShouldRefresh:
    """Tests for should_refresh."""

    def _make_state(self, step: int = 100, t_r: int = 0) -> FullState:
        """Helper to create a test state."""
        discrete = DiscreteState(t_r=t_r, b_t=1)
        return FullState(step=step, discrete=discrete)

    def test_drift_below_threshold(self) -> None:
        """Drift below delta_hi should not trigger."""
        state = self._make_state()
        config = RefreshConfig(delta_hi=0.1)
        assert not should_refresh(state, drift=0.05, val_gain=1.0, refresh_cost=1.0, config=config)

    def test_cooldown_not_elapsed(self) -> None:
        """Cooldown not elapsed should not trigger."""
        state = self._make_state(step=10, t_r=0)
        config = RefreshConfig(delta_hi=0.1, t_cool=50)
        assert not should_refresh(state, drift=0.2, val_gain=1.0, refresh_cost=1.0, config=config)

    def test_warmup_not_passed(self) -> None:
        """Warmup not passed should not trigger."""
        state = self._make_state(step=5)
        config = RefreshConfig(delta_hi=0.1, t_warmup=10)
        assert not should_refresh(state, drift=0.2, val_gain=1.0, refresh_cost=1.0, config=config)

    def test_hysteresis_inactive(self) -> None:
        """Inactive hysteresis should not trigger."""
        state = self._make_state()
        state = state.copy_with(discrete=state.discrete.copy_with(b_t=0))
        config = RefreshConfig(delta_hi=0.1)
        assert not should_refresh(state, drift=0.2, val_gain=1.0, refresh_cost=1.0, config=config)

    def test_val_gain_below_cost(self) -> None:
        """Validation gain below cost threshold should not trigger."""
        state = self._make_state()
        config = RefreshConfig(delta_hi=0.1, gamma_cost=1.0)
        assert not should_refresh(state, drift=0.2, val_gain=0.5, refresh_cost=1.0, config=config)

    def test_all_conditions_met(self) -> None:
        """All conditions met should trigger."""
        state = self._make_state(step=100, t_r=0)
        config = RefreshConfig(delta_hi=0.1, t_cool=50, t_warmup=10, gamma_cost=0.01)
        assert should_refresh(state, drift=0.2, val_gain=1.0, refresh_cost=1.0, config=config)


class TestTransitionState:
    """Tests for transition_state."""

    def test_refresh_updates_t_r(self) -> None:
        """Refreshing should update t_r to current step."""
        state = FullState(step=100, discrete=DiscreteState(t_r=0))
        new_state = transition_state(state, refreshed=True)
        assert new_state.discrete.t_r == 100

    def test_no_refresh_preserves_t_r(self) -> None:
        """Not refreshing should preserve t_r."""
        state = FullState(step=100, discrete=DiscreteState(t_r=50))
        new_state = transition_state(state, refreshed=False)
        assert new_state.discrete.t_r == 50


class TestBudgetAccountant:
    """Tests for BudgetAccountant."""

    def test_remaining(self) -> None:
        """Remaining should decrease after recording refresh."""
        accountant = BudgetAccountant(total_budget=100.0)
        accountant.record_refresh(cost=30.0)
        assert accountant.remaining == 70.0

    def test_exceeding_budget_raises(self) -> None:
        """Exceeding total budget should raise BudgetExceededError."""
        accountant = BudgetAccountant(total_budget=10.0)
        with pytest.raises(BudgetExceededError, match="budget exceeded"):
            accountant.record_refresh(cost=20.0)

    def test_spent_property(self) -> None:
        """Spent should track cumulative cost."""
        accountant = BudgetAccountant(total_budget=100.0)
        accountant.record_refresh(cost=10.0)
        accountant.record_refresh(cost=20.0)
        assert accountant.spent == 30.0


class TestRunRefreshPipeline:
    """Tests for run_refresh_pipeline."""

    def test_output_is_discrete_state(self, rng: np.random.Generator) -> None:
        """Pipeline should return a DiscreteState."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        state = FullState(step=10)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert isinstance(new_discrete, DiscreteState)
        assert new_discrete.Z is not None
        assert new_discrete.A is not None
        assert new_discrete.M_g is not None
        assert new_discrete.c_g > 0.0
        assert new_discrete.c_l > 0.0

    def test_landmark_shape(self, rng: np.random.Generator) -> None:
        """Landmarks should have shape (m_g, d)."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        state = FullState(step=10)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert new_discrete.Z.shape == (16, d)

    def test_anchor_shape(self, rng: np.random.Generator) -> None:
        """Anchors should have shape (m_l, d)."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        state = FullState(step=10)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert new_discrete.A.shape == (4, d)

    def test_ablation_disable_residual_aware_anchors(self, rng: np.random.Generator) -> None:
        """Disable residual-aware anchors should still produce valid anchors."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        state = FullState(step=10)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
            ablation=AblationConfig(disable_residual_aware_anchors=True),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert new_discrete.A is not None
        assert new_discrete.A.shape == (4, d)

    def test_ablation_disable_orthogonalization(self, rng: np.random.Generator) -> None:
        """Disable orthogonalization should skip the orthogonalization step."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        state = FullState(step=10)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
            ablation=AblationConfig(disable_orthogonalization=True),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert new_discrete.A is not None
        assert new_discrete.c_g > 0.0

    def test_ablation_static_scaling(self, rng: np.random.Generator) -> None:
        """Static scaling should reuse existing calibration scalars."""
        n, d = 100, 4
        U_data = rng.standard_normal((n, d))
        y_data = rng.standard_normal(n)
        initial_discrete = DiscreteState(c_g=2.0, c_l=3.0, rho=0.7)
        state = FullState(step=10, discrete=initial_discrete)
        config = TrainingConfig(
            embedding_dim=d, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
            ablation=AblationConfig(static_scaling=True),
        )
        new_discrete = run_refresh_pipeline(state, U_data, y_data, config, rng=rng)
        assert new_discrete.c_g == pytest.approx(2.0)
        assert new_discrete.c_l == pytest.approx(3.0)


class TestConfigCopyWith:
    """Tests for config copy_with methods."""

    def test_numerics_config_copy_with(self) -> None:
        """copy_with should create a modified NumericsConfig."""
        config = NumericsConfig(tau_eig=1e-6)
        new_config = config.copy_with(tau_eig=1e-4)
        assert new_config.tau_eig == pytest.approx(1e-4)
        assert config.tau_eig == pytest.approx(1e-6)

    def test_refresh_config_copy_with(self) -> None:
        """copy_with should create a modified RefreshConfig."""
        config = RefreshConfig(t_cool=50)
        new_config = config.copy_with(t_cool=10)
        assert new_config.t_cool == 10
        assert config.t_cool == 50
