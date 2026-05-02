"""Ridge regression solvers.

Implements Section 9:
    w* = argmin_w ||y - Phi w||_2^2 + lambda ||w||_2^2
    (Phi^T Phi + lambda I) w = Phi^T y
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

    Solves (S + lambda I) w = b via Cholesky with optional precision escalation.
    """

    def __init__(
        self,
        lambda_reg: float,
        kappa_threshold: float = 1e12,
        precision: Literal["float32", "float64"] = "float64",
    ) -> None:
        """Initialize direct solver.

        Args:
            lambda_reg: Ridge regularization parameter.
            kappa_threshold: Maximum acceptable condition number.
            precision: Solve precision.
        """
        self._lambda_reg = lambda_reg
        self._kappa_threshold = kappa_threshold
        self._precision = precision

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ridge regression via Cholesky.

        Args:
            phi: Feature matrix of shape (n, m).
            y: Target vector of shape (n,).

        Returns:
            Coefficient vector w of shape (m,).

        Raises:
            ConditioningError: If conditioning exceeds threshold.
        """
        s, b = build_normal_equations(phi, y, self._lambda_reg)

        # Check conditioning
        check_conditioning(s, threshold=self._kappa_threshold, name="S + lambda I")

        # Solve via Cholesky
        L = safe_cholesky(s)
        # Forward substitution: L v = b
        v = np.linalg.solve(L, b)
        # Backward substitution: L^T w = v
        w = np.linalg.solve(L.T, v)
        return w


class IterativeRidgeSolver:
    """Iterative ridge solver using preconditioned conjugate gradients.

    Suitable for large feature dimensions where direct Cholesky is expensive.
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
            max_iter: Maximum CG iterations.
            tol: Convergence tolerance.
            kappa_threshold: Maximum acceptable condition number.
        """
        self._lambda_reg = lambda_reg
        self._max_iter = max_iter
        self._tol = tol
        self._kappa_threshold = kappa_threshold

    def solve(self, phi: Array, y: Array) -> Array:
        """Solve ridge regression via PCG.

        Args:
            phi: Feature matrix of shape (n, m).
            y: Target vector of shape (n,).

        Returns:
            Coefficient vector w of shape (m,).

        Raises:
            ConditioningError: If conditioning exceeds threshold.
        """
        s, b = build_normal_equations(phi, y, self._lambda_reg)

        # Check conditioning
        check_conditioning(s, threshold=self._kappa_threshold, name="S + lambda I")

        # Build diagonal preconditioner
        pre = diagonal_preconditioner(s)

        return pcg_solve(s, b, max_iter=self._max_iter, tol=self._tol, preconditioner=pre)
