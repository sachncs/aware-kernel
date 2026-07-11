"""Preconditioners for iterative ridge solvers.

Provides a diagonal Jacobi preconditioner for the preconditioned conjugate
gradient (PCG) solver.  The preconditioner scales the normal equations
matrix ``S`` so that its eigenvalues are closer to unity, which
accelerates PCG convergence.

The diagonal preconditioner is:

    ``P = diag(1 / sqrt(diag(S)))``

This is the simplest and cheapest preconditioner, requiring O(m) work to
construct and O(m) work per application.  More sophisticated
preconditioners (e.g., incomplete Cholesky, multigrid) could be added
by implementing a function with the same signature.
"""

import numpy as np

from aware_kernel.aware.types import Array


def diagonal_preconditioner(s: Array) -> Array:
    """Build a diagonal preconditioner ``P = diag(1 / sqrt(diag(S)))``.

    The preconditioner scales each dimension by the inverse square root
    of the corresponding diagonal entry of ``S``.  This normalizes the
    diagonal to unity, which is the optimal diagonal preconditioner in
    the sense of minimizing the condition number of ``P S P``.

    Non-positive diagonal entries are treated as ``1.0`` to avoid
    division by zero or negative values (which would indicate a
    non-PSD matrix).

    Args:
        s: Normal equations matrix ``S`` of shape ``(m, m)``.

    Returns:
        Preconditioner vector of shape ``(m,)``.
    """
    diag = np.diagonal(s)
    diag = np.where(diag <= 0, 1.0, diag)
    return 1.0 / np.sqrt(diag)
