"""Feature calibration and stability.

Implements Section 7:
    c_g = sqrt(1/n * tr(Phi_g^T Phi_g) + epsilon_c)
    c_l = sqrt(1/n * tr((Phi_l_perp)^T Phi_l_perp) + epsilon_c)
    bar_phi_g(x) = phi_g(x) / c_g
    bar_phi_l_perp(x) = phi_l_perp(x) / c_l
"""

import numpy as np

from aware_kernel.aware.types import Array


def compute_global_calibration(
    phi_g: Array,
    epsilon_c: float,
) -> float:
    """Compute global calibration scalar c_g.

    Args:
        phi_g: Global features of shape (n, r_g).
        epsilon_c: Minimum calibration scaling.

    Returns:
        Calibration scalar c_g > 0.
    """
    n = phi_g.shape[0]
    trace = float(np.trace(phi_g.T @ phi_g))
    return np.sqrt(trace / n + epsilon_c)


def compute_local_calibration(
    phi_l_perp: Array,
    epsilon_c: float,
) -> float:
    """Compute local calibration scalar c_l.

    Args:
        phi_l_perp: Orthogonalized local features of shape (n, m_l).
        epsilon_c: Minimum calibration scaling.

    Returns:
        Calibration scalar c_l > 0.
    """
    n = phi_l_perp.shape[0]
    trace = float(np.trace(phi_l_perp.T @ phi_l_perp))
    return np.sqrt(trace / n + epsilon_c)


def calibrate_global_features(phi_g: Array, c_g: float) -> Array:
    """Scale global features by calibration constant.

    Args:
        phi_g: Global features of shape (n, r_g) or (r_g,).
        c_g: Calibration scalar.

    Returns:
        Calibrated features.
    """
    return phi_g / c_g


def calibrate_local_features(phi_l_perp: Array, c_l: float) -> Array:
    """Scale local features by calibration constant.

    Args:
        phi_l_perp: Local features of shape (n, m_l) or (m_l,).
        c_l: Calibration scalar.

    Returns:
        Calibrated features.
    """
    return phi_l_perp / c_l
