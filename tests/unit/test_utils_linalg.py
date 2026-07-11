"""Unit tests for aware_kernel.utils.linalg."""

import numpy as np

from aware_kernel.utils.linalg import (
    diagonal_preconditioner,
    frobenius_norm,
    pcg_solve,
    relative_frobenius_drift,
)


class TestFrobeniusNorm:
    """Tests for frobenius_norm."""

    def test_identity(self) -> None:
        """Frobenius norm of identity should be sqrt(n)."""
        m = np.eye(3)
        expected = np.sqrt(3)
        np.testing.assert_allclose(frobenius_norm(m), expected)

    def test_zero(self) -> None:
        """Frobenius norm of zero matrix should be zero."""
        assert frobenius_norm(np.zeros((4, 5))) == 0.0


class TestRelativeFrobeniusDrift:
    """Tests for relative_frobenius_drift."""

    def test_identical(self) -> None:
        """Drift of identical matrices should be zero."""
        m = np.ones((3, 3))
        assert relative_frobenius_drift(m, m) == 0.0

    def test_zero_reference(self) -> None:
        """Zero reference should return zero drift to avoid division by zero."""
        current = np.ones((3, 3))
        assert relative_frobenius_drift(current, np.zeros((3, 3))) == 0.0

    def test_nonzero_drift(self) -> None:
        """Non-zero drift should be positive."""
        ref = np.eye(3)
        current = ref + 0.1 * np.ones((3, 3))
        drift = relative_frobenius_drift(current, ref)
        assert drift > 0.0


class TestDiagonalPreconditioner:
    """Tests for diagonal_preconditioner."""

    def test_positive_diagonal(self) -> None:
        """Positive diagonal should yield standard preconditioner."""
        s = np.diag([4.0, 9.0, 16.0])
        pre = diagonal_preconditioner(s)
        expected = np.array([0.5, 1.0 / 3.0, 0.25])
        np.testing.assert_allclose(pre, expected)

    def test_non_positive_diagonal(self) -> None:
        """Non-positive diagonal entries should be treated as 1.0."""
        s = np.diag([0.0, -1.0, 4.0])
        pre = diagonal_preconditioner(s)
        expected = np.array([1.0, 1.0, 0.5])
        np.testing.assert_allclose(pre, expected)


class TestPcgSolve:
    """Tests for pcg_solve."""

    def test_simple_system(self) -> None:
        """CG should solve a simple SPD system."""
        s = np.array([[2.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, 2.0])
        w = pcg_solve(s, b)
        expected = np.linalg.solve(s, b)
        np.testing.assert_allclose(w, expected, atol=1e-5)

    def test_with_preconditioner(self) -> None:
        """CG with preconditioner should solve correctly."""
        s = np.array([[3.0, 0.5], [0.5, 2.0]])
        b = np.array([1.0, 1.0])
        pre = diagonal_preconditioner(s)
        w = pcg_solve(s, b, preconditioner=pre)
        expected = np.linalg.solve(s, b)
        np.testing.assert_allclose(w, expected, atol=1e-5)

    def test_non_convergence(self) -> None:
        """Ill-conditioned system should be solvable but may need tolerance."""
        s = np.array([[1.0, 0.0], [0.0, 1e-12]])
        b = np.array([1.0, 1.0])
        # With loose tolerance, CG should return something
        w = pcg_solve(s, b, tol=1e-3, max_iter=100)
        assert w.shape == (2,)
