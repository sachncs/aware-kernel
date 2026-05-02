"""Unit tests for aware_kernel.solver modules."""

import numpy as np
import pytest

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.solver.normal_eq import (
    accumulate_normal_matrix,
    accumulate_normal_vector,
    build_normal_equations,
)
from aware_kernel.solver.preconditioner import diagonal_preconditioner
from aware_kernel.solver.ridge import DirectRidgeSolver, IterativeRidgeSolver


class TestAccumulateNormalMatrix:
    """Tests for accumulate_normal_matrix."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """Output should have shape (m, m)."""
        phi = rng.standard_normal((10, 5))
        s = accumulate_normal_matrix(phi)
        assert s.shape == (5, 5)

    def test_symmetric(self, rng: np.random.Generator) -> None:
        """Output should be symmetric."""
        phi = rng.standard_normal((10, 5))
        s = accumulate_normal_matrix(phi)
        np.testing.assert_allclose(s, s.T, atol=1e-12)

    def test_psd(self, rng: np.random.Generator) -> None:
        """Output should be positive semi-definite."""
        phi = rng.standard_normal((10, 5))
        s = accumulate_normal_matrix(phi)
        eigenvalues = np.linalg.eigvalsh(s)
        assert np.all(eigenvalues >= -1e-10)


class TestAccumulateNormalVector:
    """Tests for accumulate_normal_vector."""

    def test_shape(self, rng: np.random.Generator) -> None:
        """Output should have shape (m,)."""
        phi = rng.standard_normal((10, 5))
        y = rng.standard_normal(10)
        b = accumulate_normal_vector(phi, y)
        assert b.shape == (5,)


class TestBuildNormalEquations:
    """Tests for build_normal_equations."""

    def test_shapes(self, rng: np.random.Generator) -> None:
        """S and b should have correct shapes."""
        phi = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        s, b = build_normal_equations(phi, y, lambda_reg=1e-2)
        assert s.shape == (5, 5)
        assert b.shape == (5,)

    def test_spd(self, rng: np.random.Generator) -> None:
        """S + lambda I should be SPD."""
        phi = rng.standard_normal((20, 5))
        y = rng.standard_normal(20)
        s, _ = build_normal_equations(phi, y, lambda_reg=1e-2)
        eigenvalues = np.linalg.eigvalsh(s)
        assert np.all(eigenvalues > 0.0)


class TestDiagonalPreconditioner:
    """Tests for diagonal_preconditioner."""

    def test_positive_diagonal(self) -> None:
        """Positive diagonal should yield standard preconditioner."""
        s = np.diag([4.0, 9.0, 16.0])
        pre = diagonal_preconditioner(s)
        expected = np.array([0.5, 1.0 / 3.0, 0.25])
        np.testing.assert_allclose(pre, expected)


class TestDirectRidgeSolver:
    """Tests for DirectRidgeSolver."""

    def test_basic(self, rng: np.random.Generator) -> None:
        """Should solve a simple ridge problem."""
        phi = rng.standard_normal((50, 5))
        w_true = rng.standard_normal(5)
        y = phi @ w_true + 0.1 * rng.standard_normal(50)
        solver = DirectRidgeSolver(lambda_reg=1e-2)
        w = solver.solve(phi, y)
        assert w.shape == (5,)

    def test_agrees_with_lstsq(self, rng: np.random.Generator) -> None:
        """Solution should match numpy lstsq with ridge."""
        phi = rng.standard_normal((50, 5))
        w_true = rng.standard_normal(5)
        y = phi @ w_true + 0.1 * rng.standard_normal(50)
        lambda_reg = 1e-2
        solver = DirectRidgeSolver(lambda_reg=lambda_reg)
        w = solver.solve(phi, y)
        # Reference: solve via direct normal equations
        s_ref = phi.T @ phi + lambda_reg * np.eye(5)
        b_ref = phi.T @ y
        w_ref = np.linalg.solve(s_ref, b_ref)
        np.testing.assert_allclose(w, w_ref, atol=1e-5)

    def test_ill_conditioned_raises(self) -> None:
        """Ill-conditioned problem should raise ConditioningError."""
        phi = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
        y = np.array([1.0, 2.0])
        solver = DirectRidgeSolver(lambda_reg=1e-12, kappa_threshold=1e10)
        with pytest.raises(ConditioningError):
            solver.solve(phi, y)


class TestIterativeRidgeSolver:
    """Tests for IterativeRidgeSolver."""

    def test_basic(self, rng: np.random.Generator) -> None:
        """Should solve a simple ridge problem."""
        phi = rng.standard_normal((50, 5))
        w_true = rng.standard_normal(5)
        y = phi @ w_true + 0.1 * rng.standard_normal(50)
        solver = IterativeRidgeSolver(lambda_reg=1e-2)
        w = solver.solve(phi, y)
        assert w.shape == (5,)

    def test_agrees_with_direct(self, rng: np.random.Generator) -> None:
        """Iterative and direct solvers should agree on well-conditioned data."""
        phi = rng.standard_normal((50, 5))
        w_true = rng.standard_normal(5)
        y = phi @ w_true + 0.1 * rng.standard_normal(50)
        lambda_reg = 1e-2
        direct = DirectRidgeSolver(lambda_reg=lambda_reg)
        iterative = IterativeRidgeSolver(lambda_reg=lambda_reg, tol=1e-8)
        w_direct = direct.solve(phi, y)
        w_iter = iterative.solve(phi, y)
        np.testing.assert_allclose(w_iter, w_direct, atol=1e-4)
