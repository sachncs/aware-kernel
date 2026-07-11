"""Linear algebra helpers for aware-kernel.

Provides safe Cholesky wrappers, matrix-free preconditioned conjugate
gradients (PCG), and Frobenius norm utilities.

These utilities are used across the solver, refresh, and training
modules to perform robust linear algebra operations with proper error
handling and numerical safeguards.

Design rationale
----------------
The PCG solver is provided as an alternative to direct Cholesky for
large feature dimensions.  It uses ``scipy.sparse.linalg.cg`` under the
 hood, wrapped with a ``LinearOperator`` to avoid materializing the full
matrix when possible.

The Frobenius norm utilities are used by the drift computation and
orthogonality penalty in the outer objective.
"""

import numpy as np
import scipy.sparse.linalg as spla

from aware_kernel.aware.exceptions import ConditioningError
from aware_kernel.aware.types import Array


def frobenius_norm(matrix: Array) -> float:
    """Compute Frobenius norm of a matrix.

    The Frobenius norm is ``||A||_F = sqrt(sum_ij |a_ij|^2)``, which
    is the Euclidean norm of the matrix flattened to a vector.

    Args:
        matrix: Input array of shape ``(i, j)``.

    Returns:
        Frobenius norm (non-negative scalar).
    """
    return float(np.linalg.norm(matrix, "fro"))


def relative_frobenius_drift(current: Array, reference: Array) -> float:
    """Compute relative Frobenius-norm drift between two matrices.

    The drift metric is:

        ``Delta = ||current - reference||_F / ||reference||_F``

    This measures the fractional change in the matrix, which is
    scale-invariant.  Returns ``0.0`` when the reference matrix is zero
    to avoid division by zero.

    Args:
        current: Current matrix (e.g., ``R_t``).
        reference: Reference matrix (e.g., ``R_{t_r}`` at last refresh).

    Returns:
        Relative drift ``Delta >= 0``.
    """
    ref_norm = frobenius_norm(reference)
    if ref_norm == 0.0:
        return 0.0
    diff_norm = frobenius_norm(current - reference)
    return diff_norm / ref_norm


def diagonal_preconditioner(s: Array) -> Array:
    """Build a diagonal preconditioner ``P = diag(1 / sqrt(diag(S)))``.

    This is the Jacobi (diagonal) preconditioner for PCG.  It scales
    each dimension by the inverse square root of the corresponding
    diagonal entry, normalizing the diagonal to unity.

    Non-positive diagonal entries are treated as ``1.0`` to avoid
    numerical issues.

    Args:
        s: Normal equations matrix ``S`` of shape ``(m, m)``.

    Returns:
        Preconditioner vector of shape ``(m,)``.
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
    """Solve ``(S + lambda I) w = b`` via preconditioned conjugate gradients.

    PCG is an iterative method for solving symmetric positive-definite
    linear systems.  It converges to the exact solution in at most ``m``
    iterations (where ``m`` is the system dimension), but typically
    converges much faster with a good preconditioner.

    The implementation wraps ``scipy.sparse.linalg.cg`` with a
    ``LinearOperator`` to avoid materializing the full matrix when
    possible (e.g., in matrix-free settings).

    Args:
        s: Coefficient matrix of shape ``(m, m)``.  Must be SPD.
        b: Right-hand side of shape ``(m,)``.
        max_iter: Maximum CG iterations.
        tol: Convergence tolerance relative to the initial residual
            norm.
        preconditioner: Optional diagonal preconditioner vector of
            shape ``(m,)``.  If ``None``, no preconditioning is applied.

    Returns:
        Solution vector ``w`` of shape ``(m,)``.

    Raises:
        ConditioningError: If CG fails with a negative error code
            (indicating a breakdown).
    """
    m = s.shape[0]

    # Wrap the matrix-vector product as a LinearOperator to enable
    # matrix-free iterations when the full matrix is not needed.
    def mv(v: Array) -> Array:
        return s @ v

    A_op = spla.LinearOperator((m, m), matvec=mv, dtype=s.dtype)

    # Build the inverse preconditioner operator if provided.
    M_inv = None
    if preconditioner is not None:
        M_inv = spla.LinearOperator(
            (m, m),
            matvec=lambda v: preconditioner * v,
            dtype=s.dtype,
        )

    w, info = spla.cg(A_op, b, tol=tol, maxiter=max_iter, M=M_inv)

    if info < 0:
        # Negative info indicates a breakdown (e.g., indefinite matrix).
        raise ConditioningError(f"CG failed with error code {info}")
    if info > 0:
        # info > 0 means max_iter reached without convergence.
        # This is not an error per se, but the solution may be
        # inaccurate.  We return the best approximation found.
        pass

    return w
