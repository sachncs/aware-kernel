"""Global feature builder following the FeatureBuilder protocol.

Provides a convenient wrapper around ``NystromGlobalBasis`` that satisfies
the ``FeatureBuilder`` protocol defined in ``aware_kernel.aware.types``.

The ``GlobalFeatureBuilder`` is used by the fusion stage to construct
global features from projected embeddings without directly coupling to
the Nyström implementation details.
"""

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array
from aware_kernel.global_basis.nystrom import NystromGlobalBasis


class GlobalFeatureBuilder:
    """Builder for global Nystr\"om features.

    Wraps a ``NystromGlobalBasis`` and exposes the ``FeatureBuilder``
    protocol interface.  This decouples the fusion stage from the
    specific Nyström implementation, allowing alternative global basis
    strategies (e.g., random Fourier features) to be substituted.

    Attributes:
        basis: The underlying Nystr\"om global basis.
    """

    def __init__(
        self,
        basis: NystromGlobalBasis,
    ) -> None:
        """Initialize with a precomputed global basis.

        Args:
            basis: Precomputed ``NystromGlobalBasis`` containing
                landmarks, whitening map, and eigendecomposition.
        """
        self.basis = basis

    @classmethod
    def from_data(
        cls,
        U_data: Array,
        m_g: int,
        config: NumericsConfig,
        rng: np.random.Generator | None = None,
    ) -> "GlobalFeatureBuilder":
        """Build a GlobalFeatureBuilder by selecting landmarks from data.

        Convenience factory that performs landmark selection, kernel
        computation, and whitening map construction in one call.

        Args:
            U_data: Projected embeddings of shape ``(n, d)``.
            m_g: Number of landmarks.  Must be ``<= n``.
            config: Numerical stability configuration.
            rng: Optional random generator for reproducibility.

        Returns:
            Initialized ``GlobalFeatureBuilder``.
        """
        basis = NystromGlobalBasis.from_data(U_data, m_g, config, rng)
        return cls(basis)

    def build(self, u: Array) -> Array:
        """Build global features for projected embeddings.

        Delegates to ``NystromGlobalBasis.build_features``.

        Args:
            u: Projected embeddings of shape ``(n, d)`` or ``(d,)``.

        Returns:
            Global features of shape ``(n, r_g)`` or ``(r_g,)``.
        """
        return self.basis.build_features(u)
