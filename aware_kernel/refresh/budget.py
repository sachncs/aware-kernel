"""Budget accountant for refresh cost tracking.

Tracks amortized refresh cost and enforces budget constraints.

The budget accountant provides a safety mechanism that prevents the adaptive
refresh mechanism from consuming unlimited compute in pathological cases.
Each refresh incurs a fixed cost (``refresh_cost`` in ``TrainingConfig``),
and the cumulative cost is tracked against an amortized budget
(``total_refresh_budget``).

When the budget is exhausted, ``BudgetExceededError`` is raised, signaling
that the training loop should either increase the budget or adjust the
refresh controller parameters to reduce refresh frequency.

Design rationale
----------------
The budget is amortized over the entire training horizon.  A user who
sets ``total_refresh_budget = 100`` and ``refresh_cost = 1.0`` allows
at most 100 refreshes.  This provides a hard upper bound on the total
compute spent on discrete refreshes, which is important for cost
predictability in production deployments.
"""

from aware_kernel.aware.exceptions import BudgetExceededError


class BudgetAccountant:
    """Tracks cumulative refresh cost against an amortized budget.

    The accountant maintains a running total of refresh expenditure and
    raises ``BudgetExceededError`` when the total exceeds the budget.

    Attributes:
        total_budget: Total allowed refresh cost over training.
        spent: Cumulative cost spent so far.
    """

    def __init__(self, total_budget: float) -> None:
        """Initialize budget accountant.

        Args:
            total_budget: Total allowed refresh cost over the training
                horizon.  ``float("inf")`` disables budgeting entirely.
        """
        self.total_budget = total_budget
        self.spent = 0.0

    def record_refresh(self, cost: float) -> None:
        """Record a refresh expenditure.

        Adds the cost to the cumulative total and checks against the
        budget.  If the budget is exceeded, ``BudgetExceededError`` is
        raised immediately.

        Args:
            cost: Cost of the refresh (typically ``refresh_cost`` from
                ``TrainingConfig``).

        Raises:
            BudgetExceededError: If cumulative cost exceeds total budget.
        """
        self.spent += cost
        if self.spent > self.total_budget:
            raise BudgetExceededError(
                f"Refresh budget exceeded: {self.spent:.2f} > {self.total_budget:.2f}"
            )

    @property
    def remaining(self) -> float:
        """Remaining budget.  Clamped at zero."""
        return max(self.total_budget - self.spent, 0.0)
