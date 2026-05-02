"""Numerical correctness tests for SPD solve.

These tests verify the guarantees from Section 9 and Section 13 of the method blueprint.
"""

import numpy as np

from aware_kernel.solver.normal_eq import build_normal_equations
from aware_kernel.solver.ridge import DirectRidgeSolver


class TestSPDSolve:
    """Tests for SPD normal equations solve."""

    def test_solve_matrix_is_spd(self, rng: np.random.Generator) -> None:
        """Phi^T Phi + lambda I must be SPD for lambda > 0."""
        phi = rng.standard_normal((50, 10))
        lambda_reg = 1e-2
        s, _ = build_normal_equations(phi, np.zeros(50), lambda_reg)
        eigenvalues = np.linalg.eigvalsh(s)
        assert np.all(eigenvalues > 0.0)

    def test_cholesky_decomposable(self, rng: np.random.Generator) -> None:
        """SPD matrix should be Cholesky-decomposable."""
        phi = rng.standard_normal((50, 10))
        lambda_reg = 1e-2
        s, _ = build_normal_equations(phi, np.zeros(50), lambda_reg)
        L = np.linalg.cholesky(s)
        np.testing.assert_allclose(L @ L.T, s, atol=1e-10)

    def test_solution_reduces_objective(self, rng: np.random.Generator) -> None:
        """Ridge solution should have lower objective than zero vector."""
        phi = rng.standard_normal((50, 5))
        w_true = rng.standard_normal(5)
        y = phi @ w_true + 0.1 * rng.standard_normal(50)
        lambda_reg = 1e-2
        solver = DirectRidgeSolver(lambda_reg=lambda_reg)
        w = solver.solve(phi, y)

        def objective(w_vec: np.ndarray) -> float:
            residual = y - phi @ w_vec
            return float(np.sum(residual**2) + lambda_reg * np.sum(w_vec**2))

        assert objective(w) < objective(np.zeros(5))
