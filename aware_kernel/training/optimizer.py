"""Gradient-based outer-loop optimizer for the projection matrix R.

Implements Section 10: computes gradients of the outer objective with respect
to R and performs a gradient descent step. The objective includes the ridge
prediction loss, Frobenius regularizer on R, orthogonality penalty, and
diversity penalty.

Because the chain from R -> embeddings -> features is complex, this optimizer
uses central finite differences on a mini-batch while holding the discrete
state fixed. This is standard for bilevel optimization research scaffolds.
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.config import TrainingConfig
from aware_kernel.aware.state import FullState
from aware_kernel.aware.types import Array
from aware_kernel.embedding.projector import Projector
from aware_kernel.training.objectives import compute_outer_objective


class OuterObjectiveOptimizer:
    """Optimizer for the continuous outer objective over R.

    Updates R via gradient descent using finite-difference gradients
    evaluated on mini-batches while holding the discrete basis fixed.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        lambda_r: float = 0.0,
        lambda_orth: float = 0.0,
        gamma_div: float = 0.0,
        fd_epsilon: float = 1e-5,
    ) -> None:
        """Initialize optimizer.

        Args:
            lr: Learning rate for gradient descent on R.
            lambda_r: Weight for ||R||_F^2.
            lambda_orth: Weight for ||R^T R - I||_F^2.
            gamma_div: Weight for diversity penalty D(Phi_g, Phi_l_perp).
            fd_epsilon: Finite-difference perturbation magnitude.
        """
        self._lr = lr
        self._lambda_r = lambda_r
        self._lambda_orth = lambda_orth
        self._gamma_div = gamma_div
        self._fd_epsilon = fd_epsilon

    def _evaluate_objective(
        self,
        R: Array,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
        config: TrainingConfig,
    ) -> float:
        """Evaluate outer objective for a given R, holding discrete state fixed.

        Args:
            R: Candidate projection matrix.
            state: Current full state (discrete state is reused).
            X_batch: Mini-batch inputs.
            y_batch: Mini-batch targets.
            config: Training configuration.

        Returns:
            Scalar outer objective value.
        """
        from aware_kernel.training.loop import TrainingLoop

        loop = TrainingLoop(config)

        # Build features with candidate R
        embedder = state.continuous.theta.get("embedder") if state.continuous.theta else None
        if embedder is None:
            raise RuntimeError("Embedder not found in state")

        embeddings = embedder.embed(X_batch)
        projector = Projector(R)
        U = projector.transform(embeddings)

        phi = loop._build_fused_features(U, state.discrete)

        # Solve ridge on mini-batch
        from aware_kernel.solver.ridge import DirectRidgeSolver

        solver = DirectRidgeSolver(lambda_reg=config.lambda_reg)
        w = solver.solve(phi, y_batch)

        # Rebuild phi_g and phi_l_perp for diversity penalty
        # For efficiency, skip the expensive orthogonalization in the
        # continuous update and approximate using the current discrete state's
        # fused feature splitting heuristic.
        r_g = state.discrete.M_g.shape[1] if state.discrete.M_g is not None else 0
        m_l = state.discrete.A.shape[0] if state.discrete.A is not None else 0

        if phi.shape[1] == r_g + m_l:
            phi_g = phi[:, :r_g]
            phi_l = phi[:, r_g:]
            # Approximate phi_l_perp as phi_l (continuous update doesn't refresh)
            # This is a reasonable approximation during continuous optimization
            phi_l_perp = phi_l
        else:
            # Fallback: zero features
            phi_g = np.zeros((phi.shape[0], 1))
            phi_l_perp = np.zeros((phi.shape[0], 1))

        return compute_outer_objective(
            y=y_batch,
            phi=phi,
            w=w,
            R=R,
            phi_g=phi_g,
            phi_l_perp=phi_l_perp,
            lambda_r=self._lambda_r,
            lambda_orth=self._lambda_orth,
            gamma_div=self._gamma_div,
        )

    def _compute_gradient_fd(
        self,
        R: Array,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
        config: TrainingConfig,
        rng: np.random.Generator,
    ) -> Array:
        """Compute gradient of outer objective w.r.t. R via central finite differences.

        For efficiency, this uses a simultaneous perturbation approach:
        sample a random direction matrix D with entries in {-1, +1},
        evaluate objective at R + eps * D and R - eps * D,
        and approximate the directional derivative.

        This is repeated for `n_dir` random directions and averaged.

        Args:
            R: Current projection matrix.
            state: Current state.
            X_batch: Mini-batch inputs.
            y_batch: Mini-batch targets.
            config: Training configuration.
            rng: Random generator.

        Returns:
            Gradient matrix with same shape as R.
        """
        d1, d2 = R.shape
        n_dir = max(1, min(5, d1 * d2 // 100))  # Scale directions with size

        grad = np.zeros_like(R)
        for _ in range(n_dir):
            D = rng.choice([-1.0, 1.0], size=R.shape)
            D = D / np.linalg.norm(D, "fro")

            R_plus = R + self._fd_epsilon * D
            R_minus = R - self._fd_epsilon * D

            obj_plus = self._evaluate_objective(R_plus, state, X_batch, y_batch, config)
            obj_minus = self._evaluate_objective(R_minus, state, X_batch, y_batch, config)

            grad += (obj_plus - obj_minus) / (2.0 * self._fd_epsilon) * D

        return grad / n_dir

    def step(
        self,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
        config: TrainingConfig,
        rng: np.random.Generator,
    ) -> FullState:
        """Perform one gradient descent step on R.

        Args:
            state: Current training state.
            X_batch: Mini-batch inputs.
            y_batch: Mini-batch targets.
            config: Training configuration.
            rng: Random generator.

        Returns:
            Updated state with new R.
        """
        R = state.continuous.R
        if R is None:
            raise RuntimeError("R is not initialized in continuous state")

        grad = self._compute_gradient_fd(R, state, X_batch, y_batch, config, rng)
        new_R = R - self._lr * grad

        new_continuous = state.continuous.copy_with(R=new_R)
        return state.copy_with(continuous=new_continuous)
