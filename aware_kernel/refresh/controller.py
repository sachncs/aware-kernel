"""Refresh controller with hysteresis, cooldown, and warmup.

Implements Section 11 of the method blueprint: the decision logic that
determines when discrete basis parameters should be refreshed.

A refresh triggers only when **all five** conditions are satisfied:

1. **Drift threshold**: ``Delta > delta_hi``
   The relative Frobenius-norm drift of the projection matrix ``R``
   exceeds the high threshold.

2. **Cooldown**: ``(t+1 - t_r) >= T_cool``
   Sufficient steps have elapsed since the last refresh.

3. **Warmup**: ``t+1 >= T_warmup``
   The training step exceeds the minimum warmup period.

4. **Hysteresis**: ``b_t = 1``
   The hysteresis flag is active (prevents refresh churn).

5. **Budget-scaled validation gain**: ``Delta L_val > gamma_cost * C_refresh``
   The estimated validation improvement justifies the refresh cost.

The combination of these conditions prevents both excessive refreshes
(churn) and missed opportunities (stale basis).

Design rationale
----------------
The five-condition design is intentionally conservative.  Each condition
addresses a specific failure mode:

* Drift threshold prevents refreshes when the basis is still adequate.
* Cooldown prevents rapid oscillation between basis states.
* Warmup allows the initial basis to stabilize before the first refresh.
* Hysteresis provides an additional dampening mechanism.
* Budget scaling ensures refreshes are economically justified.
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

    Checks all five conditions for refresh activation.  All must be
    satisfied for the function to return ``True``.

    Args:
        state: Current training state.  Provides ``step`` and discrete
            metadata (``t_r``, ``b_t``).
        drift: Current drift metric ``Delta = ||R_t - R_{t_r}||_F / ||R_{t_r}||_F``.
        val_gain: Validation gain estimate ``Delta L_val``.
        refresh_cost: Estimated cost ``C_refresh``.
        config: Refresh configuration containing thresholds.

    Returns:
        ``True`` if all five conditions are met.
    """
    # Condition 1: Drift exceeds high threshold.
    # This ensures the basis is sufficiently outdated before triggering.
    if drift <= config.delta_hi:
        return False

    # Condition 2: Cooldown elapsed since last refresh.
    # Prevents rapid oscillation between basis states.
    steps_since_refresh = state.step - state.discrete.t_r
    if steps_since_refresh < config.t_cool:
        return False

    # Condition 3: Warmup period passed.
    # Allows the initial basis to stabilize before the first refresh.
    if state.step < config.t_warmup:
        return False

    # Condition 4: Hysteresis flag active.
    # Additional dampening to prevent refresh churn.
    if state.discrete.b_t != 1:
        return False

    # Condition 5: Validation gain exceeds budget-scaled cost.
    # Ensures the refresh is economically justified.
    return not val_gain <= config.gamma_cost * refresh_cost


def transition_state(state: FullState, refreshed: bool) -> FullState:
    """Update controller state after a refresh decision.

    If a refresh was performed, updates the last-refresh timestamp
    (``t_r``) to the current step.  This resets the cooldown timer.

    Args:
        state: Current state.
        refreshed: Whether a refresh was performed.

    Returns:
        Updated state with refreshed timestamp (if applicable).
    """
    if refreshed:
        new_discrete = state.discrete.copy_with(t_r=state.step)
        return state.copy_with(discrete=new_discrete)
    return state
