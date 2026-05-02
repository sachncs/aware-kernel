"""Refresh controller with hysteresis, cooldown, and warmup.

Implements Section 11:
    trig = 1[ Delta > delta_hi AND (t+1 - t_r) >= T_cool AND t+1 >= T_warmup
             AND b_t = 1 AND Delta L_val > gamma_cost * C_refresh ]
"""

from aware_kernel.aware.config import RefreshConfig
from aware_kernel.aware.state import FullState


def should_refresh(
    state: FullState,
    drift: float,
    val_gain: float,
    refresh_cost: float,
    config: RefreshConfig,
) -> bool:
    """Evaluate whether a refresh should trigger.

    Args:
        state: Current training state.
        drift: Current drift metric Delta.
        val_gain: Validation gain estimate Delta L_val.
        refresh_cost: Estimated cost C_refresh.
        config: Refresh configuration.

    Returns:
        True if all conditions hold for refresh.
    """
    # Condition 1: drift exceeds high threshold
    if drift <= config.delta_hi:
        return False

    # Condition 2: cooldown elapsed
    steps_since_refresh = state.step - state.discrete.t_r
    if steps_since_refresh < config.t_cool:
        return False

    # Condition 3: warmup passed
    if state.step < config.t_warmup:
        return False

    # Condition 4: hysteresis flag active
    if state.discrete.b_t != 1:
        return False

    # Condition 5: validation gain exceeds budget-scaled cost
    if val_gain <= config.gamma_cost * refresh_cost:
        return False

    return True


def transition_state(state: FullState, refreshed: bool) -> FullState:
    """Update controller state after refresh decision.

    Args:
        state: Current state.
        refreshed: Whether refresh was performed.

    Returns:
        Updated state.
    """
    if refreshed:
        new_discrete = state.discrete.copy_with(t_r=state.step)
        return state.copy_with(discrete=new_discrete)
    return state
