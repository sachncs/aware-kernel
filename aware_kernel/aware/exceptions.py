"""Domain-specific exceptions for aware-kernel."""


class AwareKernelError(Exception):
    """Base exception for aware-kernel errors."""

    pass


class ConditioningError(AwareKernelError):
    """Raised when numerical conditioning falls below acceptable thresholds.

    This typically indicates that the normal equations matrix is too
    ill-conditioned for the current precision or regularization level.
    """

    pass


class BudgetExceededError(AwareKernelError):
    """Raised when refresh cost exceeds the allocated amortized budget."""

    pass


class RefreshChurnError(AwareKernelError):
    """Raised when refreshes happen too frequently, indicating basis churn.

    Consider increasing cooldown or drift thresholds.
    """

    pass


class ShapeError(AwareKernelError):
    """Raised when tensor shapes are inconsistent with the model configuration."""

    pass
