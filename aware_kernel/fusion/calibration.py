"""Feature calibration and stability.

Implements Section 7 of the method blueprint: trace-based calibration
of global and local feature blocks.

Calibration normalizes the feature blocks so that their contributions to
the ridge regression are scale-balanced.  Without calibration, a feature
block with large trace would dominate the normal equations, regardless of
its actual predictive power.

The calibration scalars are:

    ``c_g = sqrt(1/n * tr(Phi_g^T Phi_g) + epsilon_c)``
    ``c_l = sqrt(1/n * tr(Phi_l_perp^T Phi_l_perp) + epsilon_c)``

and the calibrated features are:

    ``bar_phi_g(x) = phi_g(x) / c_g``
    ``bar_phi_l_perp(x) = phi_l_perp(x) / c_l``

The ``epsilon_c`` term prevents collapse to zero when feature traces are
near zero (e.g., early in training or with degenerate features).

Design rationale
----------------
The calibration step is critical for the fusion gate ``rho`` to work
correctly.  If one feature block has much larger magnitude than the
other, the gate would need to be extreme to balance them, reducing the
effectiveness of the learned gating.
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_global_calibration(
    phi_g: Array,
    epsilon_c: float,
) -> float:
    """Compute global calibration scalar ``c_g``.

    The calibration scalar is ``sqrt(tr(Phi_g^T Phi_g) / n + epsilon_c)``,
    which normalizes the average squared feature magnitude to approximately
    1.

    Args:
        phi_g: Global features of shape ``(n, r_g)``.
        epsilon_c: Minimum calibration scaling.  Prevents collapse when
            the feature trace is near zero.

    Returns:
        Calibration scalar ``c_g > 0``.
    """
    n = phi_g.shape[0]
    trace = float(np.trace(phi_g.T @ phi_g))
    return np.sqrt(trace / n + epsilon_c)


def compute_local_calibration(
    phi_l_perp: Array,
    epsilon_c: float,
) -> float:
    """Compute local calibration scalar ``c_l``.

    Same role as ``c_g`` but for the orthogonalized local features.

    Args:
        phi_l_perp: Orthogonalized local features of shape ``(n, m_l)``.
        epsilon_c: Minimum calibration scaling.

    Returns:
        Calibration scalar ``c_l > 0``.
    """
    n = phi_l_perp.shape[0]
    trace = float(np.trace(phi_l_perp.T @ phi_l_perp))
    return np.sqrt(trace / n + epsilon_c)


def calibrate_global_features(phi_g: Array, c_g: float) -> Array:
    """Scale global features by calibration constant.

    Args:
        phi_g: Global features of shape ``(n, r_g)`` or ``(r_g,)``.
        c_g: Calibration scalar.

    Returns:
        Calibrated features ``phi_g / c_g``.
    """
    return phi_g / c_g


def calibrate_local_features(phi_l_perp: Array, c_l: float) -> Array:
    """Scale local features by calibration constant.

    Args:
        phi_l_perp: Local features of shape ``(n, m_l)`` or ``(m_l,)``.
        c_l: Calibration scalar.

    Returns:
        Calibrated features ``phi_l_perp / c_l``.
    """
    return phi_l_perp / c_l
