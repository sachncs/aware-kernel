"""Budget accountant for refresh cost tracking.

Tracks amortized refresh cost and enforces budget constraints.
"""

from aware_kernel.aware.exceptions import BudgetExceededError


class BudgetAccountant:
    """Tracks cumulative refresh cost against an amortized budget."""

    def __init__(self, total_budget: float) -> None:
        """Initialize budget accountant.

        Args:
            total_budget: Total allowed refresh cost over training horizon.
        """
        self._total_budget = total_budget
        self._spent = 0.0

    def record_refresh(self, cost: float) -> None:
        """Record a refresh expenditure.

        Args:
            cost: Cost of the refresh.

        Raises:
            BudgetExceededError: If cumulative cost exceeds total budget.
        """
        self._spent += cost
        if self._spent > self._total_budget:
            raise BudgetExceededError(
                f"Refresh budget exceeded: {self._spent:.2f} > {self._total_budget:.2f}"
            )

    @property
    def remaining(self) -> float:
        """Remaining budget."""
        return max(self._total_budget - self._spent, 0.0)

    @property
    def spent(self) -> float:
        """Total spent so far."""
        return self._spent
