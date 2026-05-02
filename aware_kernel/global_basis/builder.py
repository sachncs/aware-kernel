"""Global feature builder following the FeatureBuilder protocol."""

from typing import Optional

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array, FeatureBuilder
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.utils.sampling import kmeans_pp


class GlobalFeatureBuilder:
    """Builder for global Nystr\"om features."""

    def __init__(
        self,
        basis: NystromGlobalBasis,
    ) -> None:
        """Initialize with a precomputed global basis.

        Args:
            basis: Precomputed Nystr\"omGlobalBasis.
        """
        self._basis = basis

    @classmethod
    def from_data(
        cls,
        U_data: Array,
        m_g: int,
        config: NumericsConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> "GlobalFeatureBuilder":
        """Build a GlobalFeatureBuilder by selecting landmarks from data.

        Args:
            U_data: Projected embeddings of shape (n, d).
            m_g: Number of landmarks.
            config: Numerics configuration.
            rng: Optional random generator.

        Returns:
            Initialized GlobalFeatureBuilder.
        """
        basis = NystromGlobalBasis.from_data(U_data, m_g, config, rng)
        return cls(basis)

    def build(self, u: Array) -> Array:
        """Build global features for projected embeddings.

        Args:
            u: Projected embeddings of shape (n, d) or (d,).

        Returns:
            Global features of shape (n, r_g) or (r_g,).
        """
        return self._basis.build_features(u)

    @property
    def basis(self) -> NystromGlobalBasis:
        """Access underlying global basis."""
        return self._basis
