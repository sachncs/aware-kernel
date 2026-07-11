"""Integration tests for refresh trigger behavior."""

from aware_kernel.aware.config import RefreshConfig
from aware_kernel.aware.state import DiscreteState, FullState
from aware_kernel.refresh.controller import should_refresh


class TestRefreshTrigger:
    """Integration tests for refresh controller."""

    def test_refresh_rate_boundedness(self) -> None:
        """Refresh rate should respect cooldown: N_refresh(T) <= floor(T / T_cool) + 1."""
        config = RefreshConfig(delta_hi=0.05, t_cool=10, t_warmup=5)
        t_r = 0
        refresh_count = 0

        for step in range(1, 101):
            state = FullState(step=step, discrete=DiscreteState(t_r=t_r, b_t=1))
            # Simulate high drift after warmup
            drift = 0.1 if step > 5 else 0.01
            val_gain = 1.0
            refresh_cost = 0.1

            if should_refresh(state, drift, val_gain, refresh_cost, config):
                refresh_count += 1
                t_r = step

        max_allowed = (100 // 10) + 1
        assert refresh_count <= max_allowed

    def test_never_refreshes_before_warmup(self) -> None:
        """No refresh should trigger before warmup."""
        config = RefreshConfig(delta_hi=0.05, t_cool=1, t_warmup=10)
        for step in range(1, 10):
            state = FullState(step=step, discrete=DiscreteState(t_r=0, b_t=1))
            assert not should_refresh(
                state, drift=1.0, val_gain=1.0, refresh_cost=0.1, config=config
            )
