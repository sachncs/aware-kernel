"""Drift computation for refresh controller.

Implements Section 11 of the method blueprint: computing the relative
Frobenius-norm drift of the projection matrix ``R``.

The drift metric is:

    ``Delta_{t+1} = ||R_{t+1} - R_{t_r}||_F / ||R_{t_r}||_F``

where ``R_{t_r}`` is the projection matrix at the last refresh.  This
measures how much the learned projection has drifted from its state at
the last discrete refresh.

The drift is the primary signal for the refresh controller.  When it
exceeds ``delta_hi``, the discrete basis may be outdated and a refresh
should be considered (subject to the other controller conditions).

Design rationale
----------------
The relative Frobenius norm is preferred over the absolute norm because
it is scale-invariant.  A projection matrix with large entries would
have a large absolute drift even for small relative changes, while a
normalized metric correctly captures the fractional change.
"""

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.utils.linalg import relative_frobenius_drift


def compute_drift(current_R: Array, reference_R: Array) -> float:
    """Compute relative Frobenius-norm drift.

    Measures how much the current projection matrix ``R`` has changed
    since the last refresh:

        ``Delta = ||R_current - R_ref||_F / ||R_ref||_F``

    Args:
        current_R: Current projection matrix.
        reference_R: Reference projection matrix at the last refresh.

    Returns:
        Relative drift ``Delta >= 0``.  Returns ``0.0`` when the
        reference matrix is zero.
    """
    return relative_frobenius_drift(current_R, reference_R)
