"""Core types, configuration, state, and exceptions for aware-kernel.

This package defines the foundational abstractions used throughout the
library:

* ``types``: Type aliases (``Array``, ``FeatureMatrix``) and structural
  protocols (``Embedder``, ``RidgeSolver``, ``MemoryAccumulator``, etc.).
* ``config``: Immutable configuration dataclasses (``TrainingConfig``,
  ``NumericsConfig``, ``RefreshConfig``, ``AblationConfig``).
* ``state``: Frozen state containers (``ContinuousState``,
  ``DiscreteState``, ``FullState``).
* ``exceptions``: Domain-specific exceptions for numerical and budget
  failures.

All configuration and state objects are frozen (immutable) to prevent
accidental mutation during training.
"""

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
