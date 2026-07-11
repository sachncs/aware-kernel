"""Nystr\"om global basis: landmark selection and feature building.

Implements Section 4 of the method blueprint: the global Nystr\"om block
with landmark set ``Z``.

The global basis provides a low-rank approximation to the full kernel
matrix by selecting a subset of ``m_g`` landmark points and constructing
a whitened RBF kernel feature map.  The feature map is:

    ``phi_g(u) = k(u, Z) @ M_g``

where ``k(u, Z)`` is the RBF kernel vector between the query point and
all landmarks, and ``M_g`` is the soft-truncated spectral whitening map.

The Nystr\"om approximation has theoretical guarantees on the approximation
error as a function of the number of landmarks and the spectral decay of
the kernel matrix (see Drineas & Mahoney, 2005).

Algorithm
---------
1. Select ``m_g`` landmarks via k-means++ from the projected embedding
   space.
2. Compute the kernel-on-landmarks matrix ``W = k(Z, Z)``.
3. Eigendecompose ``W`` and apply soft-truncated whitening to obtain
   ``M_g``.
4. For each query, compute ``phi_g(u) = k(u, Z) @ M_g``.

Complexity
----------
* Landmark selection: O(n * m_g * d) via k-means++.
* Kernel computation: O(n * m_g * d) per query batch.
* Feature construction: O(n * m_g * r_g) per query batch.

Dependencies
------------
* ``aware_kernel.global_basis.whitening`` -- Whitening map construction.
* ``aware_kernel.utils.sampling`` -- k-means++ landmark selection.
"""

import numpy as np

from aware_kernel.aware.config import NumericsConfig
from aware_kernel.aware.types import Array
from aware_kernel.global_basis.whitening import build_whitening_map
from aware_kernel.utils.sampling import kmeans_pp


class NystromGlobalBasis:
    """Nystr\"om global basis with soft-truncated whitening.

    Constructs a low-rank kernel feature map from a set of landmark
    points using the Nystr\"om method.  The feature map is:

        ``phi_g(u) = k(u, Z) @ M_g``

    where ``Z`` are landmarks, ``k`` is the RBF kernel, and ``M_g`` is
    the whitening map obtained from soft-truncated eigendecomposition of
    the kernel-on-landmarks matrix.

    The whitening map ``M_g`` simultaneously:
    * Reduces the feature dimension from ``m_g`` to ``r_g`` (the retained
      rank).
    * Normalizes the feature covariance to improve conditioning of the
      downstream ridge regression.

    Attributes:
        Z: Landmarks of shape ``(m_g, d)``.
        M_g: Whitening map of shape ``(m_g, r_g)``.
        U: Eigenvectors of ``W`` of shape ``(m_g, m_g)``.
        eigenvalues: Clipped eigenvalues of ``W`` of shape ``(m_g,)``.
        r_g: Retained rank (number of eigenvalues above ``tau_eig``).

    References
        Drineas, P. & Mahoney, M. W. (2005).  *On the Nystr\"om Method
        for Approximating a Gram Matrix for Improved Kernel-Based
        Learning.*  JMLR, 6, 2153--2175.
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
            Z: Landmarks of shape ``(m_g, d)``.
            M_g: Whitening map of shape ``(m_g, r_g)``.
            U: Eigenvectors of shape ``(m_g, m_g)``.
            eigenvalues: Clipped eigenvalues of shape ``(m_g,)``.
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
        """Build a Nystr\"omGlobalBasis from pre-selected landmarks.

        Computes the kernel-on-landmarks matrix ``W = k(Z, Z)`` and
        constructs the whitening map via soft-truncated eigendecomposition.

        Args:
            Z: Landmarks of shape ``(m_g, d)``.
            config: Numerical stability configuration controlling
                eigenvalue clipping, epsilon scaling, and truncation.

        Returns:
            Initialized ``NystromGlobalBasis``.
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
        rng: np.random.Generator | None = None,
    ) -> "NystromGlobalBasis":
        """Build a Nystr\"omGlobalBasis by selecting landmarks from data.

        Performs the full construction pipeline:

        1. Select ``m_g`` landmarks via k-means++ from ``U_data``.
        2. Compute the kernel-on-landmarks matrix.
        3. Construct the whitening map.

        Args:
            U_data: Projected embeddings of shape ``(n, d)``.
            m_g: Number of landmarks to select.  Must be ``<= n``.
            config: Numerical stability configuration.
            rng: Optional random generator for reproducibility.

        Returns:
            Initialized ``NystromGlobalBasis``.

        Raises:
            ValueError: If ``m_g > n``.
        """
        if rng is None:
            rng = np.random.default_rng()

        Z = kmeans_pp(U_data, k=m_g, rng=rng)
        return cls.from_landmarks(Z, config)

    def compute_kernel_vector(self, u_query: Array) -> Array:
        """Compute the RBF kernel vector ``k(u_query, Z)``.

        Uses the squared-distance expansion for efficiency:

            ``k(u, z_j) = exp(-||u - z_j||^2)``

        where ``gamma = 1.0`` (the default bandwidth).

        For batch inputs, the squared distances are computed via the
        identity ``||u - z||^2 = ||u||^2 + ||z||^2 - 2 u^T z``, which
        avoids materializing the full ``(n, m_g, d)`` difference tensor.

        Args:
            u_query: Single projected embedding of shape ``(d,)`` or
                batch of shape ``(n, d)``.

        Returns:
            Kernel vector of shape ``(m_g,)`` or batch ``(n, m_g)``.
        """
        if u_query.ndim == 1:
            sq_dists = np.sum((self.Z - u_query) ** 2, axis=1)
        else:
            # Efficient batch kernel via squared-distance identity:
            # ||u_i - z_j||^2 = ||u_i||^2 + ||z_j||^2 - 2 u_i^T z_j
            sq_dists = (
                np.sum(self.Z**2, axis=1).reshape(1, -1)
                + np.sum(u_query**2, axis=1).reshape(-1, 1)
                - 2.0 * (u_query @ self.Z.T)
            )
        result: Array = np.exp(-sq_dists)
        return result

    def build_features(self, u_query: Array) -> Array:
        """Build global features ``phi_g(u_query) = k(u_query, Z) @ M_g``.

        This is the main entry point for constructing global features.
        It computes the kernel vector and applies the whitening map in a
        single operation.

        Args:
            u_query: Projected embeddings of shape ``(d,)`` or
                ``(n, d)``.

        Returns:
            Global features of shape ``(r_g,)`` or ``(n, r_g)``.
        """
        k_vec = self.compute_kernel_vector(u_query)
        if u_query.ndim == 1:
            return k_vec @ self.M_g
        return k_vec @ self.M_g
