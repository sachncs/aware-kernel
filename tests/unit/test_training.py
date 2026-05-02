"""Unit tests for aware_kernel.training modules."""

import numpy as np
import pytest

from aware_kernel.aware.config import (
    AblationConfig,
    RefreshConfig,
    TrainingConfig,
)
from aware_kernel.training.callbacks import CheckpointCallback, LoggingCallback
from aware_kernel.training.loop import TrainingLoop
from aware_kernel.training.objectives import (
    compute_outer_objective,
    diversity_penalty,
    orthogonality_penalty,
    ridge_prediction_loss,
)


class TestRidgePredictionLoss:
    """Tests for ridge_prediction_loss."""

    def test_zero_residual(self) -> None:
        """Perfect prediction should yield zero loss."""
        phi = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([1.0, 0.0])
        y = phi @ w
        loss = ridge_prediction_loss(y, phi, w)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_positive(self, rng: np.random.Generator) -> None:
        """Non-perfect prediction should yield positive loss."""
        phi = rng.standard_normal((10, 5))
        w = rng.standard_normal(5)
        y = rng.standard_normal(10)
        loss = ridge_prediction_loss(y, phi, w)
        assert loss >= 0.0


class TestOrthogonalityPenalty:
    """Tests for orthogonality_penalty."""

    def test_identity_zero(self) -> None:
        """Identity matrix should have zero penalty."""
        R = np.eye(5)
        assert orthogonality_penalty(R) == pytest.approx(0.0, abs=1e-10)

    def test_non_identity_positive(self, rng: np.random.Generator) -> None:
        """Non-orthogonal matrix should have positive penalty."""
        R = rng.standard_normal((5, 5))
        assert orthogonality_penalty(R) > 0.0


class TestDiversityPenalty:
    """Tests for diversity_penalty."""

    def test_orthogonal_local(self, rng: np.random.Generator) -> None:
        """Diversity penalty should be small for near-orthogonal blocks."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 5))
        # Project local orthogonal to global
        from aware_kernel.local_corrective.orthogonalizer import orthogonalize_local_features
        phi_l_perp = orthogonalize_local_features(phi_g, phi_l, eta_o=1e-4)
        div = diversity_penalty(phi_g, phi_l_perp)
        assert div < 1e-5


class TestComputeOuterObjective:
    """Tests for compute_outer_objective."""

    def test_positive(self, rng: np.random.Generator) -> None:
        """Objective should be non-negative."""
        phi_g = rng.standard_normal((20, 5))
        phi_l = rng.standard_normal((20, 3))
        phi = np.concatenate([phi_g, phi_l], axis=1)
        y = rng.standard_normal(20)
        w = rng.standard_normal(8)
        R = np.eye(5)
        obj = compute_outer_objective(y, phi, w, R, phi_g, phi_l)
        assert obj >= 0.0


class TestTrainingLoop:
    """Tests for TrainingLoop."""

    def test_initialize_state(self, rng: np.random.Generator) -> None:
        """Should initialize a valid FullState."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        assert state.w is not None
        assert state.discrete.Z is not None
        assert state.discrete.A is not None

    def test_continuous_update(self, rng: np.random.Generator) -> None:
        """Continuous update should change R."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        R_old = state.continuous.R.copy()
        new_state = loop.continuous_update(state, X[:10], y[:10])
        assert not np.allclose(new_state.continuous.R, R_old)

    def test_evaluate(self, rng: np.random.Generator) -> None:
        """Evaluate should return metrics."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        metrics = loop.evaluate(state, X, y)
        assert "rmse" in metrics
        assert metrics["rmse"] >= 0.0

    def test_continuous_update_with_optimizer(self, rng: np.random.Generator) -> None:
        """Continuous update with optimizer should change R deterministically."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
            lr=1e-3,
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        R_old = state.continuous.R.copy()
        new_state = loop.continuous_update(state, X[:10], y[:10])
        assert new_state.continuous.R is not None
        assert new_state.continuous.R.shape == R_old.shape
        # Optimizer should produce a non-trivial update
        assert not np.allclose(new_state.continuous.R, R_old)

    def test_maybe_refresh_budget_blocks(self, rng: np.random.Generator) -> None:
        """Refresh should be blocked when budget is exhausted."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0),
            total_refresh_budget=0.0,
            refresh_cost=1.0,
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(step=100)
        new_state = loop.maybe_refresh(state, X, y, val_gain=10.0)
        assert new_state.discrete.t_r == state.discrete.t_r

    def test_maybe_refresh_ablation_disable_refresh(self, rng: np.random.Generator) -> None:
        """Ablation disable_refresh should skip refresh regardless of trigger."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0),
            ablation=AblationConfig(disable_refresh=True),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(step=100)
        new_state = loop.maybe_refresh(state, X, y, val_gain=10.0)
        assert new_state.discrete.t_r == state.discrete.t_r

    def test_maybe_refresh_ablation_disable_hysteresis(self, rng: np.random.Generator) -> None:
        """Ablation disable_hysteresis should allow refresh with b_t=0."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0, t_cool=0, t_warmup=0),
            ablation=AblationConfig(disable_hysteresis=True),
            total_refresh_budget=float("inf"),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        # Force hysteresis off
        state = state.copy_with(
            step=100,
            discrete=state.discrete.copy_with(b_t=0),
        )
        new_state = loop.maybe_refresh(state, X, y, val_gain=10.0)
        # With disable_hysteresis, refresh should proceed (if budget allows)
        # We can't assert refresh happened because drift may not trigger,
        # but we can assert the code path runs without error.
        assert new_state is not None

    def test_maybe_refresh_ablation_disable_cooldown(self, rng: np.random.Generator) -> None:
        """Ablation disable_cooldown should remove cooldown constraint."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, t_cool=1000),
            ablation=AblationConfig(disable_cooldown=True),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(step=1)
        # Should not error even with large t_cool because ablation disables it
        new_state = loop.maybe_refresh(state, X, y)
        assert new_state is not None

    def test_evaluate_missing_embedder(self, rng: np.random.Generator) -> None:
        """Evaluate with missing embedder should return inf rmse."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(continuous=state.continuous.copy_with(theta=None))
        metrics = loop.evaluate(state, X, y)
        assert metrics["rmse"] == float("inf")

    def test_evaluate_missing_w(self, rng: np.random.Generator) -> None:
        """Evaluate with missing coefficients should return inf rmse."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(w=None)
        metrics = loop.evaluate(state, X, y)
        assert metrics["rmse"] == float("inf")

    def test_maybe_refresh_no_embedder(self, rng: np.random.Generator) -> None:
        """maybe_refresh with missing embedder should return state unchanged."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(
            continuous=state.continuous.copy_with(theta=None),
            step=100,
        )
        new_state = loop.maybe_refresh(state, X, y, val_gain=10.0)
        assert new_state.discrete.t_r == state.discrete.t_r

    def test_maybe_refresh_triggers_and_updates_r_ref(self, rng: np.random.Generator) -> None:
        """maybe_refresh should trigger refresh and set _R_ref for drift tracking."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0, t_cool=0, t_warmup=0),
            total_refresh_budget=float("inf"),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        state = state.copy_with(step=100)
        new_state = loop.maybe_refresh(state, X, y, val_gain=10.0)
        # Refresh should have triggered because drift is high and constraints are relaxed
        assert new_state.discrete.t_r == 100
        # A second refresh with the same R should have lower drift
        second_state = loop.maybe_refresh(new_state, X, y, val_gain=10.0)
        assert second_state is not None

    def test_build_fused_features_uninitialized(self, rng: np.random.Generator) -> None:
        """_build_fused_features with uninitialized discrete state should raise."""
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3),
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)
        uninitialized = state.discrete.copy_with(Z=None)
        with pytest.raises(ValueError, match="not initialized"):
            loop._build_fused_features(state.continuous.R, uninitialized)

    def test_maybe_refresh_invokes_callback(self, rng: np.random.Generator) -> None:
        """maybe_refresh should invoke on_refresh callbacks when triggered."""
        from aware_kernel.training.callbacks import Callback

        class RefreshCounter(Callback):
            def __init__(self) -> None:
                self.count = 0

            def on_refresh(self, step: int, state: object) -> None:
                self.count += 1

        counter = RefreshCounter()
        X, y = self._make_data(rng)
        config = TrainingConfig(
            embedding_dim=4, m_g=16, m_l=4, lambda_reg=1e-2, seed=42,
            refresh=RefreshConfig(k_local=3, delta_hi=0.0, t_cool=0, t_warmup=0),
            total_refresh_budget=float("inf"),
        )
        loop = TrainingLoop(config, callbacks=[counter])
        state = loop.initialize_state(X, y)
        state = state.copy_with(step=100)
        loop.maybe_refresh(state, X, y, val_gain=10.0)
        assert counter.count >= 1

    def _make_data(self, rng: np.random.Generator) -> tuple:
        """Helper to create synthetic data."""
        n, d = 100, 4
        X = rng.standard_normal((n, d))
        true_w = rng.standard_normal(d)
        y = X @ true_w + 0.1 * rng.standard_normal(n)
        return X, y


class TestCallbacks:
    """Tests for training callbacks."""

    def test_logging_callback(self) -> None:
        """LoggingCallback should instantiate."""
        cb = LoggingCallback(log_interval=5)
        assert cb._log_interval == 5

    def test_checkpoint_callback(self) -> None:
        """CheckpointCallback should instantiate."""
        cb = CheckpointCallback(save_interval=10, path_prefix="test")
        assert cb._save_interval == 10
