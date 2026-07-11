"""Feature fusion with global/local gating.

Implements Section 8 of the method blueprint: fusing calibrated global
and local features via a logistic sigmoid gate.

The fusion step combines the two calibrated feature blocks into a single
feature vector:

    ``rho = sigma(a)``
    ``phi(x) = [sqrt(rho) * bar_phi_g(x); sqrt(1 - rho) * bar_phi_l_perp(x)]``

where ``sigma`` is the logistic sigmoid function and ``a`` is a learnable
logit parameter.  The square-root weighting ensures that the L2 norms of
the two blocks are balanced when ``rho = 0.5``.

The gate ``rho`` controls the relative contribution of global (smooth,
general) and local (residual-correcting, specific) features.  During
training, ``rho`` is initialized to 0.5 (equal weighting) and can be
updated as part of the discrete refresh.

Design rationale
----------------
The square-root parameterization ``sqrt(rho)`` and ``sqrt(1-rho)`` is
preferred over direct weighting because it preserves the normalization:
``rho + (1-rho) = 1`` in the squared-norm sense.  This means the total
feature norm is approximately constant regardless of ``rho``, which
improves the stability of the ridge regression.
"""

import numpy as np

from aware_kernel.aware.types import Array


def sigmoid(a: float) -> float:
    """Sigmoid function mapping the real line to ``(0, 1)``.

    Computes ``sigma(a) = 1 / (1 + exp(-a))``.

    Args:
        a: Logit parameter.  Can be any real number.

    Returns:
        Sigmoid value in ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-a))


def compute_gate(a: float) -> float:
    """Compute fusion gate ``rho = sigma(a)``.

    This is a thin wrapper around ``sigmoid`` for clarity in the fusion
    pipeline.

    Args:
        a: Logit parameter.

    Returns:
        Gate value ``rho`` in ``(0, 1)``.
    """
    return sigmoid(a)


def fuse_features(
    phi_g: Array,
    phi_l_perp: Array,
    rho: float,
) -> Array:
    """Fuse global and local features with gate ``rho``.

    Computes the weighted concatenation:

        ``phi = [sqrt(rho) * phi_g; sqrt(1-rho) * phi_l_perp]``

    Supports both single-sample ``(m,)`` and batch ``(n, m)`` inputs.

    Args:
        phi_g: Calibrated global features of shape ``(n, r_g)`` or
            ``(r_g,)``.
        phi_l_perp: Calibrated local features of shape ``(n, m_l)`` or
            ``(m_l,)``.
        rho: Fusion gate in ``[0, 1]``.

    Returns:
        Fused features of shape ``(n, r_g + m_l)`` or ``(r_g + m_l,)``.

    Raises:
        ValueError: If ``rho`` is outside ``[0, 1]``.
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

    Inverse of ``fuse_features``: extracts the first ``r_g`` dimensions
    as global features and the remainder as local features.

    Args:
        phi: Fused features of shape ``(n, r_g + m_l)`` or
            ``(r_g + m_l,)``.
        r_g: Global feature dimension.

    Returns:
        Tuple of ``(phi_g_part, phi_l_part)``.
    """
    if phi.ndim == 1:
        return phi[:r_g], phi[r_g:]
    return phi[:, :r_g], phi[:, r_g:]
