"""Drift computation for refresh controller.

Implements Section 11:
    Delta_{t+1} = ||R_{t+1} - R_{t_r}||_F / ||R_{t_r}||_F
"""

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.utils.linalg import relative_frobenius_drift


def compute_drift(current_R: Array, reference_R: Array) -> float:
    """Compute relative Frobenius norm drift.

    Args:
        current_R: Current projection matrix.
        reference_R: Reference projection matrix at last refresh.

    Returns:
        Relative drift Delta.
    """
    return relative_frobenius_drift(current_R, reference_R)
