"""Numerical correctness tests for conditioning bounds.

These tests verify that the solver respects conditioning thresholds.
"""

import numpy as np
import pytest

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.solver.ridge import DirectRidgeSolver


class TestConditioningBounds:
    """Tests for conditioning threshold enforcement."""

    def test_well_conditioned_passes(self, rng: np.random.Generator) -> None:
        """Well-conditioned system should solve without error."""
        phi = rng.standard_normal((50, 5))
        y = rng.standard_normal(50)
        solver = DirectRidgeSolver(lambda_reg=1e-2, kappa_threshold=1e12)
        w = solver.solve(phi, y)
        assert w.shape == (5,)

    def test_ill_conditioned_raises(self) -> None:
        """Ill-conditioned system with strict threshold should raise."""
        phi = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-15]])
        y = np.array([1.0, 2.0])
        solver = DirectRidgeSolver(lambda_reg=1e-12, kappa_threshold=1e8)
        with pytest.raises(ConditioningError):
            solver.solve(phi, y)
