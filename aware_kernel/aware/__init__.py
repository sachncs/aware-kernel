"""Core types, configuration, state, and exceptions for aware-kernel."""

from aware_kernel.aware.config import NumericsConfig, RefreshConfig, TrainingConfig
from aware_kernel.aware.exceptions import (
    BudgetExceededError,
    ConditioningError,
    RefreshChurnError,
)
from aware_kernel.aware.state import ContinuousState, DiscreteState, FullState
from aware_kernel.aware.types import Array, FeatureMatrix

__all__ = [
    "Array",
    "FeatureMatrix",
    "TrainingConfig",
    "NumericsConfig",
    "RefreshConfig",
    "ContinuousState",
    "DiscreteState",
    "FullState",
    "ConditioningError",
    "BudgetExceededError",
    "RefreshChurnError",
]
