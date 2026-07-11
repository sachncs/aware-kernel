"""Ridge regression solvers.

Implements Section 9 of the method blueprint: solving the Tikhonov-
regularized least-squares problem:

    ``w* = argmin_w ||y - Phi w||_2^2 + lambda ||w||_2^2``
    ``(Phi^T Phi + lambda I) w = Phi^T y``

Two solver strategies are provided:

* **Direct** (``DirectRidgeSolver``): Cholesky decomposition.  Exact,
  O(m^3), preferred when m is moderate (< ~2000).
* **Iterative** (``IterativeRidgeSolver``): Preconditioned conjugate
  gradients (PCG).  Approximate, O(k * m^2), preferred when m is large.

Both solvers include conditioning checks and will raise
``ConditioningError`` if the normal equations matrix is too
ill-conditioned.

Design rationale
----------------
The direct solver uses Cholesky because the normal equations matrix
``S = Phi^T Phi + lambda I`` is guaranteed symmetric positive-definite
(by construction, since ``lambda > 0``).  Cholesky is both faster and
more numerically stable than general-purpose LU decomposition for SPD
matrices.

The iterative solver is provided as an alternative for large feature
dimensions where the O(m^3) cost of Cholesky becomes prohibitive.  The
diagonal preconditioner accelerates convergence by scaling the eigenvalues
of the system matrix toward unity.
"""

from typing import Literal

import numpy as np

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.aware.types import Array
from aware_kernel.solver.normal_eq import build_normal_equations
from aware_kernel.solver.preconditioner import diagonal_preconditioner
from aware_kernel.utils.linalg import pcg_solve
from aware_kernel.utils.numerics import check_conditioning, safe_cholesky


class DirectRidgeSolver:
    """Direct ridge solver using Cholesky decomposition.

    Solves ``(S + lambda I) w = b`` via Cholesky decomposition with
    optional precision escalation.  The algorithm:

    1. Build normal equations ``S = Phi^T Phi + lambda I``, ``b = Phi^T y``.
    2. Check conditioning of ``S``.
    3. Compute Cholesky factor ``L`` such that ``S = L L^T``.
    4. Forward substitution: ``L v = b``.
    5. Backward substitution: ``L^T w = v``.

    Complexity: O(m^3) for the Cholesky decomposition, O(m^2) for the
    forward/backward substitutions.

    Attributes:
        lambda_reg: Ridge regularization parameter.
        kappa_threshold: Maximum acceptable condition number.
        precision: Solve precision.
    """

    def __init__(
        self,
        lambda_reg: float,
        kappa_threshold: float = 1e12,
        precision: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Initialize direct solver.

        Args:
            lambda_reg: Ridge regularization parameter.  Must be
                positive.
            kappa_threshold: Maximum acceptable condition number.
                Exceeding this raises ``ConditioningError``.
            precision: Solve precision.  ``"float64"`` is recommended
                for most use cases.
        """
        self.lambda_reg = lambda_reg
        self.kappa_threshold = kappa_threshold
        self.precision = precision

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ridge regression via Cholesky decomposition.

        Args:
            phi: Feature matrix of shape ``(n, m)``.
            y: Target vector of shape ``(n,)``.

        Returns:
            Coefficient vector ``w`` of shape ``(m,)``.

        Raises:
            ConditioningError: If the condition number of ``S + lambda I``
                exceeds ``kappa_threshold``.
            np.linalg.LinAlgError: If the Cholesky decomposition fails
                even with jitter fallback.
        """
        s, b = build_normal_equations(phi, y, self.lambda_reg)

        # Verify numerical conditioning before attempting Cholesky.
        # A poorly conditioned matrix can cause catastrophic
        # cancellation in the solve, producing garbage coefficients.
        check_conditioning(s, threshold=self.kappa_threshold, name="S + lambda I")

        # Cholesky decomposition: S = L L^T.
        # safe_cholesky adds jitter if the initial decomposition fails,
        # which handles near-singular matrices gracefully.
        L = safe_cholesky(s)
        # Forward substitution: L v = b
        v = np.linalg.solve(L, b)
        # Backward substitution: L^T w = v
        w = np.linalg.solve(L.T, v)
        return w


class IterativeRidgeSolver:
    """Iterative ridge solver using preconditioned conjugate gradients.

    Suitable for large feature dimensions where direct Cholesky is
    expensive.  Uses a diagonal preconditioner to accelerate convergence.

    The PCG algorithm converges to the exact solution in at most ``m``
    iterations (where ``m`` is the feature dimension), but typically
    converges much faster when the preconditioner is effective.

    Attributes:
        lambda_reg: Ridge regularization parameter.
        max_iter: Maximum CG iterations.
        tol: Convergence tolerance.
        kappa_threshold: Maximum acceptable condition number.
    """

    def __init__(
        self,
        lambda_reg: float,
        max_iter: int = 1000,
        tol: float = 1e-6,
        kappa_threshold: float = 1e12,
    ) -> None:
        """Initialize iterative solver.

        Args:
            lambda_reg: Ridge regularization parameter.
            max_iter: Maximum CG iterations.  Convergence is declared
                when the residual norm falls below ``tol``.
            tol: Convergence tolerance relative to the initial residual.
            kappa_threshold: Maximum acceptable condition number.
        """
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.kappa_threshold = kappa_threshold

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ridge regression via preconditioned conjugate gradients.

        Args:
            phi: Feature matrix of shape ``(n, m)``.
            y: Target vector of shape ``(n,)``.

        Returns:
            Coefficient vector ``w`` of shape ``(m,)``.

        Raises:
            ConditioningError: If the condition number exceeds threshold.
            ConditioningError: If CG fails to converge.
        """
        s, b = build_normal_equations(phi, y, self.lambda_reg)

        # Check conditioning before attempting iterative solve.
        check_conditioning(s, threshold=self.kappa_threshold, name="S + lambda I")

        # Build diagonal preconditioner: P = diag(1 / sqrt(diag(S))).
        # This scales the eigenvalues toward unity, accelerating CG
        # convergence for ill-conditioned systems.
        pre = diagonal_preconditioner(s)

        return pcg_solve(s, b, max_iter=self.max_iter, tol=self.tol, preconditioner=pre)
