"""Domain-specific exceptions for aware-kernel.

The exception hierarchy mirrors the main failure modes of the kernel
learning pipeline:

* ``AwareKernelError`` -- Base for all library exceptions.
* ``ConditioningError`` -- Numerical ill-conditioning in the normal
  equations or Cholesky decomposition.
* ``BudgetExceededError`` -- Refresh cost exceeded the amortized budget.
* ``RefreshChurnError`` -- Refreshes occurring too frequently, indicating
  basis instability.
* ``ShapeError`` -- Tensor shape mismatches between data and model
  configuration.

All exceptions inherit from ``AwareKernelError``, allowing callers to
catch the entire library's error surface with a single ``except`` clause.
"""


class AwareKernelError(Exception):
    """Base exception for all aware-kernel errors.

    Catch this to handle any error raised by the library without
    importing every specific exception type.
    """

    pass


class ConditioningError(AwareKernelError):
    """Raised when numerical conditioning falls below acceptable thresholds.

    This typically indicates that the normal equations matrix is too
    ill-conditioned for the current precision or regularization level.
    Common causes include:

    * Insufficient ridge regularization (``lambda_reg`` too small).
    * Degenerate feature matrices (e.g., constant features).
    * Numerical overflow in accumulated matrices.

    Recovery strategies
        * Increase ``lambda_reg`` or ``lambda_min``.
        * Switch to float64 precision.
        * Reduce the number of features (``m_g``, ``m_l``).
        * Enable jitter fallback in the Cholesky solver.
    """

    pass


class BudgetExceededError(AwareKernelError):
    """Raised when refresh cost exceeds the allocated amortized budget.

    The ``BudgetAccountant`` tracks cumulative refresh expenditure and
    raises this when the total exceeds ``total_refresh_budget``.  This
    prevents the adaptive refresh mechanism from consuming unlimited
    compute in pathological cases.

    Recovery strategies
        * Increase ``total_refresh_budget``.
        * Increase ``refresh_cost`` to reduce refresh frequency.
        * Increase ``t_cool`` or ``delta_hi`` to make refreshes less
          frequent.
    """

    pass


class RefreshChurnError(AwareKernelError):
    """Raised when refreshes happen too frequently, indicating basis churn.

    Basis churn occurs when the discrete parameters oscillate between
    states without converging, wasting compute and potentially degrading
    generalization.  This exception is a diagnostic signal that the
    refresh controller parameters need adjustment.

    Recovery strategies
        * Increase ``t_cool`` (cooldown period).
        * Increase ``delta_hi`` (drift threshold).
        * Increase ``gamma_cost`` (validation gain threshold).
    """

    pass


class ShapeError(AwareKernelError):
    """Raised when tensor shapes are inconsistent with the model configuration.

    This covers mismatches such as:

    * Input feature dimension not matching ``input_dim``.
    * Batch sizes inconsistent between features and targets.
    * Anchor count exceeding landmark count.
    * Projection matrix dimension mismatch.
    """

    pass
