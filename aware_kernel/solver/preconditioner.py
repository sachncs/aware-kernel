"""Preconditioners for iterative ridge solvers."""

import numpy as np

from aware_kernel.aware.types import Array


def diagonal_preconditioner(s: Array) -> Array:
    """Build a diagonal preconditioner P = diag(1 / sqrt(diag(S))).

    Args:
        s: Normal equations matrix S of shape (m, m).

    Returns:
        Preconditioner vector of shape (m,).
    """
    diag = np.diagonal(s)
    diag = np.where(diag <= 0, 1.0, diag)
    return 1.0 / np.sqrt(diag)
