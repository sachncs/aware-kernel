"""Fused feature builder combining global and local blocks."""

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

    Holds calibration constants and gate value, refreshed on trigger.
    """

    def __init__(
        self,
        c_g: float,
        c_l: float,
        rho: float,
    ) -> None:
        """Initialize with precomputed calibration values.

        Args:
            c_g: Global calibration scalar.
            c_l: Local calibration scalar.
            rho: Fusion gate value.
        """
        self._c_g = c_g
        self._c_l = c_l
        self._rho = rho

    @classmethod
    def from_features(
        cls,
        phi_g: Array,
        phi_l_perp: Array,
        a: float = 0.0,
        epsilon_c: float = 1e-8,
    ) -> "FusedFeatureBuilder":
        """Build a FusedFeatureBuilder from feature matrices.

        Args:
            phi_g: Global features of shape (n, r_g).
            phi_l_perp: Orthogonalized local features of shape (n, m_l).
            a: Logit parameter for gate (default 0.0 gives rho=0.5).
            epsilon_c: Minimum calibration scaling.

        Returns:
            Initialized FusedFeatureBuilder.
        """
        c_g = compute_global_calibration(phi_g, epsilon_c)
        c_l = compute_local_calibration(phi_l_perp, epsilon_c)
        rho = compute_gate(a)
        return cls(c_g=c_g, c_l=c_l, rho=rho)

    def build(self, phi_g: Array, phi_l_perp: Array) -> Array:
        """Build fused features for a batch or single sample.

        Args:
            phi_g: Global features.
            phi_l_perp: Orthogonalized local features.

        Returns:
            Fused features.
        """
        bar_phi_g = calibrate_global_features(phi_g, self._c_g)
        bar_phi_l = calibrate_local_features(phi_l_perp, self._c_l)
        return fuse_features(bar_phi_g, bar_phi_l, self._rho)

    @property
    def c_g(self) -> float:
        """Global calibration constant."""
        return self._c_g

    @property
    def c_l(self) -> float:
        """Local calibration constant."""
        return self._c_l

    @property
    def rho(self) -> float:
        """Fusion gate."""
        return self._rho
