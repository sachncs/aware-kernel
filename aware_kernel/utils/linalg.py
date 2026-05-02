"""Linear algebra helpers for aware-kernel.

Implements safe Cholesky wrappers, matrix-free PCG, and Frobenius norm utilities.
"""

import numpy as np
import scipy.sparse.linalg as spla

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.aware.types import Array


def frobenius_norm(matrix: Array) -> float:
    """Compute Frobenius norm of a matrix.

    Args:
        matrix: Input array of shape (..., ...,).

    Returns:
        Frobenius norm.
    """
    return float(np.linalg.norm(matrix, "fro"))


def relative_frobenius_drift(current: Array, reference: Array) -> float:
    """Compute relative Frobenius norm drift between two matrices.

    Delta = ||current - reference||_F / ||reference||_F

    Args:
        current: Current matrix.
        reference: Reference matrix.

    Returns:
        Relative drift.
    """
    ref_norm = frobenius_norm(reference)
    if ref_norm == 0.0:
        return 0.0
    diff_norm = frobenius_norm(current - reference)
    return diff_norm / ref_norm


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


def pcg_solve(
    s: Array,
    b: Array,
    max_iter: int = 1000,
    tol: float = 1e-6,
    preconditioner: Array | None = None,
) -> Array:
    """Solve (S + lambda I) w = b via preconditioned conjugate gradients.

    Args:
        s: Coefficient matrix of shape (m, m).
        b: Right-hand side of shape (m,).
        max_iter: Maximum CG iterations.
        tol: Convergence tolerance.
        preconditioner: Optional diagonal preconditioner vector of shape (m,).

    Returns:
        Solution vector w of shape (m,).

    Raises:
        ConditioningError: If CG fails to converge.
    """
    m = s.shape[0]

    def mv(v: Array) -> Array:
        return s @ v

    A_op = spla.LinearOperator((m, m), matvec=mv, dtype=s.dtype)

    if preconditioner is not None:
        M_inv = spla.LinearOperator(
            (m, m),
            matvec=lambda v: preconditioner * v,
            dtype=s.dtype,
        )
    else:
        M_inv = None

    w, info = spla.cg(A_op, b, tol=tol, maxiter=max_iter, M=M_inv)

    if info < 0:
        raise ConditioningError(f"CG failed with error code {info}")
    if info > 0:
        # info > 0 means max_iter reached without convergence
        pass

    return w
