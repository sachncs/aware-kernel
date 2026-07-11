"""Sampling utilities for landmark and anchor selection.

Implements two sampling strategies used in the aware-kernel pipeline:

* **k-means++** (``kmeans_pp``): Probabilistic sampling proportional to
  squared distance from the nearest selected center.  Used for initial
  landmark selection and as the baseline for coverage-based anchor
  sampling.
* **Farthest point sampling** (``farthest_point_sampling``): Greedy
  deterministic selection of the point farthest from all previously
  selected points.  Provides maximal coverage but is slower than
  k-means++.

Both methods produce ``k`` landmarks from a dataset of ``n`` points
in O(n * k * d) time.

Algorithm: k-means++
--------------------
1. Choose the first center uniformly at random.
2. For each remaining center:
   a. Compute the squared distance from each point to its nearest
      selected center.
   b. Sample the next center proportional to these distances.
3. Return the selected centers.

This is the standard k-means++ initialization (Arthur & Vassilvitskii,
2007), which provides a provable O(log k)-approximation to the optimal
k-center clustering.

Algorithm: Farthest point sampling
-----------------------------------
1. Choose the first point uniformly at random.
2. For each remaining point:
   a. Compute the minimum squared distance to all selected points.
   b. Select the point with the maximum minimum distance.
3. Return the selected points.

FPS is greedy and deterministic (given the initial point), providing
maximal coverage of the data manifold.

References
    Arthur, D. & Vassilvitskii, S. (2007).  *k-means++: The Advantages
    of Careful Seeding.*  SODA '07.
"""

import numpy as np

from aware_kernel.aware.types import Array


def kmeans_pp(
    x: Array,
    k: int,
    rng: np.random.Generator | None = None,
) -> Array:
    """Select ``k`` landmarks via k-means++ seeding.

    Implements the k-means++ initialization algorithm, which selects
    landmarks with probability proportional to their squared distance
    from the nearest already-selected landmark.  This produces well-
    spread landmarks that cover the data manifold.

    Time complexity: O(n * k * d) where ``n`` is the number of points,
    ``k`` is the number of landmarks, and ``d`` is the dimensionality.

    Args:
        x: Data matrix of shape ``(n, d)``.
        k: Number of landmarks to select.
        rng: Optional random generator for reproducibility.

    Returns:
        Selected landmarks of shape ``(k, d)``.

    Raises:
        ValueError: If ``k > n``.
    """
    n, d = x.shape
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    if rng is None:
        rng = np.random.default_rng()

    landmarks = np.zeros((k, d), dtype=x.dtype)
    indices = np.zeros(k, dtype=int)

    # First center chosen uniformly at random.
    indices[0] = rng.integers(0, n)
    landmarks[0] = x[indices[0]]

    # Distance from each point to its closest selected center.
    # Initialized to infinity so the first update correctly captures
    # the distance to the first center.
    distances = np.full(n, np.inf)

    for i in range(1, k):
        # Update distances: for each point, track the minimum squared
        # distance to any selected center.
        diff = x - landmarks[i - 1]
        dists_i = np.sum(diff**2, axis=1)
        distances = np.minimum(distances, dists_i)

        # Sample next center proportional to squared distance.
        total = np.sum(distances)
        if total == 0:
            # All remaining points are duplicates of selected centers.
            # Fall back to selecting from the unselected points.
            remaining = np.setdiff1d(np.arange(n), indices[:i])
            if len(remaining) == 0:
                # No unselected points remain; duplicate the first center.
                indices[i] = indices[0]
                landmarks[i] = landmarks[0]
            else:
                indices[i] = remaining[0]
                landmarks[i] = x[indices[i]]
        else:
            probs = distances / total
            indices[i] = rng.choice(n, p=probs)
            landmarks[i] = x[indices[i]]

    return landmarks


def farthest_point_sampling(
    x: Array,
    k: int,
    rng: np.random.Generator | None = None,
) -> Array:
    """Select ``k`` landmarks via farthest point sampling (greedy).

    Selects landmarks by iteratively choosing the point farthest from
    all previously selected points.  This provides maximal coverage
    of the data manifold.

    Time complexity: O(n * k * d).  Same asymptotic complexity as
    k-means++, but without the probabilistic component.

    Args:
        x: Data matrix of shape ``(n, d)``.
        k: Number of landmarks to select.
        rng: Optional random generator for the initial point.

    Returns:
        Selected landmarks of shape ``(k, d)``.

    Raises:
        ValueError: If ``k > n``.
    """
    n, d = x.shape
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    if rng is None:
        rng = np.random.default_rng()

    landmarks = np.zeros((k, d), dtype=x.dtype)
    indices = np.zeros(k, dtype=int)

    # First point chosen uniformly at random.
    indices[0] = rng.integers(0, n)
    landmarks[0] = x[indices[0]]

    # Minimum squared distance from each point to any selected point.
    distances = np.full(n, np.inf)

    for i in range(1, k):
        diff = x - landmarks[i - 1]
        dists_i = np.sum(diff**2, axis=1)
        distances = np.minimum(distances, dists_i)

        # Select the point with the maximum minimum distance (greedy).
        next_idx = int(np.argmax(distances))
        indices[i] = next_idx
        landmarks[i] = x[next_idx]

    return landmarks
