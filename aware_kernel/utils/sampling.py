"""Sampling utilities for landmark and anchor selection.

Implements k-means++ initialization and farthest point sampling (FPS)
for use in global basis landmark selection.
"""

import numpy as np

from aware_kernel.aware.types import Array


def kmeans_pp(
    x: Array,
    k: int,
    rng: np.random.Generator | None = None,
) -> Array:
    """Select k landmarks via k-means++ seeding.

    Args:
        x: Data matrix of shape (n, d).
        k: Number of landmarks to select.
        rng: Optional random generator for reproducibility.

    Returns:
        Selected landmarks of shape (k, d).
    """
    n, d = x.shape
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    if rng is None:
        rng = np.random.default_rng()

    landmarks = np.zeros((k, d), dtype=x.dtype)
    indices = np.zeros(k, dtype=int)

    # First center chosen uniformly at random
    indices[0] = rng.integers(0, n)
    landmarks[0] = x[indices[0]]

    # Distance from each point to its closest selected center
    distances = np.full(n, np.inf)

    for i in range(1, k):
        # Update distances
        diff = x - landmarks[i - 1]
        dists_i = np.sum(diff**2, axis=1)
        distances = np.minimum(distances, dists_i)

        # Sample next center proportional to squared distance
        total = np.sum(distances)
        if total == 0:
            # All remaining points are duplicates of selected centers
            remaining = np.setdiff1d(np.arange(n), indices[:i])
            if len(remaining) == 0:
                # Duplicate existing centers to fill quota
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
    """Select k landmarks via farthest point sampling (greedy).

    Args:
        x: Data matrix of shape (n, d).
        k: Number of landmarks to select.
        rng: Optional random generator for reproducibility.

    Returns:
        Selected landmarks of shape (k, d).
    """
    n, d = x.shape
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")

    if rng is None:
        rng = np.random.default_rng()

    landmarks = np.zeros((k, d), dtype=x.dtype)
    indices = np.zeros(k, dtype=int)

    # First point chosen uniformly at random
    indices[0] = rng.integers(0, n)
    landmarks[0] = x[indices[0]]

    distances = np.full(n, np.inf)

    for i in range(1, k):
        diff = x - landmarks[i - 1]
        dists_i = np.sum(diff**2, axis=1)
        distances = np.minimum(distances, dists_i)

        next_idx = int(np.argmax(distances))
        indices[i] = next_idx
        landmarks[i] = x[next_idx]

    return landmarks
