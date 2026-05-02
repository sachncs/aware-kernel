"""Nystr\"om global basis: landmark selection and feature building.

Implements Section 4: global Nystr\"om block with landmark set Z.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array
from aware_kernel.global_basis.whitening import build_whitening_map
from aware_kernel.utils.sampling import kmeans_pp


class NystromGlobalBasis:
    """Nystr\"om global basis with soft-truncated whitening.

    Attributes:
        Z: Landmarks of shape (m_g, d).
        M_g: Whitening map of shape (m_g, r_g).
        U: Eigenvectors of W.
        eigenvalues: Clipped eigenvalues of W.
        r_g: Retained rank.
    """

    def __init__(
        self,
        Z: Array,
        M_g: Array,
        U: Array,
        eigenvalues: Array,
        r_g: int,
    ) -> None:
        """Initialize from precomputed components.

        Args:
            Z: Landmarks.
            M_g: Whitening map.
            U: Eigenvectors.
            eigenvalues: Clipped eigenvalues.
            r_g: Retained rank.
        """
        self.Z = Z
        self.M_g = M_g
        self.U = U
        self.eigenvalues = eigenvalues
        self.r_g = r_g

    @classmethod
    def from_landmarks(
        cls,
        Z: Array,
        config: NumericsConfig,
    ) -> "NystromGlobalBasis":
        """Build a Nystr\"omGlobalBasis from landmarks.

        Args:
            Z: Landmarks of shape (m_g, d).
            config: Numerics configuration.

        Returns:
            Initialized NystromGlobalBasis.
        """
        from aware_kernel.global_basis.whitening import compute_kernel_on_landmarks

        W = compute_kernel_on_landmarks(Z)
        U, eigenvalues, M_g, r_g = build_whitening_map(W, config)
        return cls(Z=Z, M_g=M_g, U=U, eigenvalues=eigenvalues, r_g=r_g)

    @classmethod
    def from_data(
        cls,
        U_data: Array,
        m_g: int,
        config: NumericsConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> "NystromGlobalBasis":
        """Build a Nystr\"omGlobalBasis by selecting landmarks from data.

        Args:
            U_data: Projected embeddings of shape (n, d).
            m_g: Number of landmarks to select.
            config: Numerics configuration.
            rng: Optional random generator.

        Returns:
            Initialized NystromGlobalBasis.
        """
        if rng is None:
            rng = np.random.default_rng()

        Z = kmeans_pp(U_data, k=m_g, rng=rng)
        return cls.from_landmarks(Z, config)

    def compute_kernel_vector(self, u_query: Array) -> Array:
        """Compute the RBF kernel vector k(u_query, Z).

        Args:
            u_query: Single projected embedding of shape (d,) or batch (n, d).

        Returns:
            Kernel vector of shape (m_g,) or batch (n, m_g).
        """
        if u_query.ndim == 1:
            sq_dists = np.sum((self.Z - u_query) ** 2, axis=1)
        else:
            sq_dists = (
                np.sum(self.Z**2, axis=1).reshape(1, -1)
                + np.sum(u_query**2, axis=1).reshape(-1, 1)
                - 2.0 * (u_query @ self.Z.T)
            )
        return np.exp(-sq_dists)

    def build_features(self, u_query: Array) -> Array:
        """Build global features phi_g(u_query) = k(u_query, Z) M_g.

        Args:
            u_query: Projected embeddings of shape (d,) or (n, d).

        Returns:
            Global features of shape (r_g,) or (n, r_g).
        """
        k_vec = self.compute_kernel_vector(u_query)
        if u_query.ndim == 1:
            return k_vec @ self.M_g
        return k_vec @ self.M_g
