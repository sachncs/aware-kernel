"""Fused feature builder combining global and local blocks.

Provides the ``FusedFeatureBuilder`` class that orchestrates the
calibration and gating stages of the fusion pipeline.  The builder
holds precomputed calibration constants and gate value, which are
refreshed when the discrete basis is updated.

The fusion pipeline is:

1. Calibrate global features: ``bar_phi_g = phi_g / c_g``
2. Calibrate local features: ``bar_phi_l = phi_l_perp / c_l``
3. Fuse: ``phi = [sqrt(rho) * bar_phi_g; sqrt(1-rho) * bar_phi_l]``

The calibration constants ``c_g`` and ``c_l`` are computed from the
feature traces at refresh time and frozen until the next refresh.
This avoids recomputing them every step while ensuring they remain
synchronized with the current basis.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array
from aware_kernel.fusion.calibration import (
    calibrate_global_features,
    calibrate_local_features,
    compute_global_calibration,
    compute_local_calibration,
)
from aware_kernel.fusion.gate import compute_gate, fuse_features


class FusedFeatureBuilder:
    """Builder for fused global + local features.

    Holds calibration constants ``c_g``, ``c_l`` and gate value ``rho``
    that are refreshed at discrete refresh boundaries.  Between refreshes,
    these values are frozen to provide stable feature scaling.

    The builder is instantiated during the refresh pipeline and used
    throughout the continuous training phase to construct features for
    the ridge solver.

    Attributes:
        c_g: Global calibration scalar.
        c_l: Local calibration scalar.
        rho: Fusion gate value.
    """

    def __init__(
        self,
        c_g: float,
        c_l: float,
        rho: float,
    ) -> None:
        """Initialize with precomputed calibration values.

        Args:
            c_g: Global calibration scalar.  Must be positive.
            c_l: Local calibration scalar.  Must be positive.
            rho: Fusion gate value in ``(0, 1)``.
        """
        self.c_g = c_g
        self.c_l = c_l
        self.rho = rho

    @classmethod
    def from_features(
        cls,
        phi_g: Array,
        phi_l_perp: Array,
        a: float = 0.0,
        epsilon_c: float = 1e-8,
    ) -> "FusedFeatureBuilder":
        """Build a FusedFeatureBuilder from feature matrices.

        Computes calibration scalars from the feature traces and the
        gate value from the logit parameter.

        Args:
            phi_g: Global features of shape ``(n, r_g)``.
            phi_l_perp: Orthogonalized local features of shape
                ``(n, m_l)``.
            a: Logit parameter for the gate.  Default ``0.0`` gives
                ``rho = 0.5`` (equal weighting).
            epsilon_c: Minimum calibration scaling to prevent collapse.

        Returns:
            Initialized ``FusedFeatureBuilder``.
        """
        c_g = compute_global_calibration(phi_g, epsilon_c)
        c_l = compute_local_calibration(phi_l_perp, epsilon_c)
        rho = compute_gate(a)
        return cls(c_g=c_g, c_l=c_l, rho=rho)

    def build(self, phi_g: Array, phi_l_perp: Array) -> Array:
        """Build fused features for a batch or single sample.

        Applies calibration and gating to produce the final feature
        vector used by the ridge solver.

        Args:
            phi_g: Global features of shape ``(n, r_g)`` or ``(r_g,)``.
            phi_l_perp: Orthogonalized local features of shape
                ``(n, m_l)`` or ``(m_l,)``.

        Returns:
            Fused features of shape ``(n, r_g + m_l)`` or
            ``(r_g + m_l,)``.
        """
        bar_phi_g = calibrate_global_features(phi_g, self.c_g)
        bar_phi_l = calibrate_local_features(phi_l_perp, self.c_l)
        return fuse_features(bar_phi_g, bar_phi_l, self.rho)
