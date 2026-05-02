"""Unit tests for aware_kernel.aware.exceptions."""

import pytest

from aware_kernel.aware.exceptions import (
    AwareKernelError,
    BudgetExceededError,
    ConditioningError,
    RefreshChurnError,
    ShapeError,
)


class TestExceptions:
    """Tests for custom exceptions."""

    def test_base_exception(self) -> None:
        """AwareKernelError should be catchable as base."""
        with pytest.raises(AwareKernelError):
            raise ConditioningError("bad condition")

    def test_conditioning_error(self) -> None:
        """ConditioningError should carry message."""
        exc = ConditioningError("matrix is ill-conditioned")
        assert "ill-conditioned" in str(exc)

    def test_budget_exceeded_error(self) -> None:
        """BudgetExceededError should be a subtype."""
        with pytest.raises(AwareKernelError):
            raise BudgetExceededError("over budget")

    def test_refresh_churn_error(self) -> None:
        """RefreshChurnError should be a subtype."""
        with pytest.raises(AwareKernelError):
            raise RefreshChurnError("too frequent")

    def test_shape_error(self) -> None:
        """ShapeError should be a subtype."""
        with pytest.raises(AwareKernelError):
            raise ShapeError("mismatch")
