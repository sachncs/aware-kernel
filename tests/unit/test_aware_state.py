"""Unit tests for aware_kernel.aware.state."""

import numpy as np

from aware_kernel.aware.state import ContinuousState, DiscreteState, FullState


class TestContinuousState:
    """Tests for ContinuousState."""

    def test_default_creation(self) -> None:
        """Default ContinuousState should have None fields."""
        state = ContinuousState()
        assert state.theta is None
        assert state.R is None

    def test_copy_with(self) -> None:
        """copy_with should create an updated immutable state."""
        R = np.eye(3)
        state = ContinuousState().copy_with(R=R)
        assert state.R is not None
        np.testing.assert_array_equal(state.R, R)


class TestDiscreteState:
    """Tests for DiscreteState."""

    def test_default_creation(self) -> None:
        """Default DiscreteState should have sensible defaults."""
        state = DiscreteState()
        assert state.c_g == 1.0
        assert state.c_l == 1.0
        assert state.t_r == 0
        assert state.b_t == 1
        assert state.rho == 0.5

    def test_copy_with(self) -> None:
        """copy_with should update fields immutably."""
        state = DiscreteState(c_g=1.0)
        new_state = state.copy_with(c_g=2.0, t_r=10)
        assert new_state.c_g == 2.0
        assert new_state.t_r == 10
        assert state.c_g == 1.0  # original unchanged


class TestFullState:
    """Tests for FullState."""

    def test_default_creation(self) -> None:
        """Default FullState should contain default sub-states."""
        state = FullState()
        assert state.step == 0
        assert state.w is None
        assert isinstance(state.continuous, ContinuousState)
        assert isinstance(state.discrete, DiscreteState)

    def test_copy_with(self) -> None:
        """copy_with should update step and preserve sub-states."""
        state = FullState()
        new_state = state.copy_with(step=5)
        assert new_state.step == 5
        assert state.step == 0
