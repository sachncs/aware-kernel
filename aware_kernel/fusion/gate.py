"""Feature fusion with global/local gating.

Implements Section 8:
    rho = sigma(a)
    phi(x) = [sqrt(rho) * bar_phi_g(x); sqrt(1 - rho) * bar_phi_l_perp(x)]
"""

import numpy as np

from aware_kernel.aware.types import Array


def sigmoid(a: float) -> float:
    """Sigmoid function mapping real line to (0, 1).

    Args:
        a: Logit parameter.

    Returns:
        Sigmoid value in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-a))


def compute_gate(a: float) -> float:
    """Compute fusion gate rho = sigma(a).

    Args:
        a: Logit parameter.

    Returns:
        Gate value rho in (0, 1).
    """
    return sigmoid(a)


def fuse_features(
    phi_g: Array,
    phi_l_perp: Array,
    rho: float,
) -> Array:
    """Fuse global and local features with gate rho.

    Args:
        phi_g: Calibrated global features of shape (n, r_g) or (r_g,).
        phi_l_perp: Calibrated local features of shape (n, m_l) or (m_l,).
        rho: Fusion gate in [0, 1].

    Returns:
        Fused features of shape (n, r_g + m_l) or (r_g + m_l,).
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    sqrt_rho = np.sqrt(rho)
    sqrt_one_minus_rho = np.sqrt(1.0 - rho)

    if phi_g.ndim == 1:
        scaled_g = sqrt_rho * phi_g
        scaled_l = sqrt_one_minus_rho * phi_l_perp
        return np.concatenate([scaled_g, scaled_l])

    scaled_g = sqrt_rho * phi_g
    scaled_l = sqrt_one_minus_rho * phi_l_perp
    return np.concatenate([scaled_g, scaled_l], axis=1)


def split_fused_features(
    phi: Array,
    r_g: int,
) -> tuple[Array, Array]:
    """Split fused features back into global and local components.

    Args:
        phi: Fused features of shape (n, r_g + m_l) or (r_g + m_l,).
        r_g: Global feature dimension.

    Returns:
        Tuple of (phi_g_part, phi_l_part).
    """
    if phi.ndim == 1:
        return phi[:r_g], phi[r_g:]
    return phi[:, :r_g], phi[:, r_g:]
