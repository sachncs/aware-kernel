"""Gradient-based outer-loop optimizer for the projection matrix R.

Implements Section 10 of the method blueprint: computing gradients of the
outer objective with respect to ``R`` and performing gradient descent.

Because the chain ``R -> embeddings -> features -> coefficients -> loss``
is complex and non-differentiable (due to the discrete ridge solve),
this optimizer uses **central finite differences** on a mini-batch to
estimate the gradient.  This is a standard approach in bilevel optimization
research scaffolds.

The gradient is estimated via simultaneous perturbation:

1. Sample a random direction matrix ``D`` with entries in ``{-1, +1}``.
2. Evaluate the objective at ``R + eps * D`` and ``R - eps * D``.
3. Approximate the directional derivative as ``(obj_plus - obj_minus) / (2 * eps)``.
4. Repeat for ``n_dir`` random directions and average.

This yields a stochastic gradient estimate that is unbiased (on average)
and computationally tractable for moderate ``d``.

Design rationale
----------------
The simultaneous perturbation approach (Spall, 1992) is preferred over
element-wise finite differences because it requires only 2 objective
evaluations per direction (regardless of the dimension of ``R``), making
it feasible for high-dimensional projection matrices.

The number of directions ``n_dir`` scales with the problem size:
``n_dir = max(1, min(5, d^2 / 100))``, providing more gradient
estimates for larger matrices.
"""

import numpy as np

from aware_kernel.aware.config import TrainingConfig
from aware_kernel.aware.state import FullState
from aware_kernel.aware.types import Array
from aware_kernel.embedding.projector import Projector
from aware_kernel.training.objectives import compute_outer_objective


class OuterObjectiveOptimizer:
    """Optimizer for the continuous outer objective over ``R``.

    Updates ``R`` via gradient descent using finite-difference gradients
    evaluated on mini-batches while holding the discrete basis fixed.

    The gradient is estimated using simultaneous perturbation finite
    differences (SPSA-style), which requires only 2 objective evaluations
    per random direction regardless of the dimension of ``R``.

    Attributes:
        lr: Learning rate for gradient descent.
        lambda_r: Frobenius regularizer weight.
        lambda_orth: Orthogonality penalty weight.
        gamma_div: Diversity penalty weight.
        fd_epsilon: Finite-difference perturbation magnitude.
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
            lr: Learning rate for gradient descent on ``R``.
            lambda_r: Weight for ``||R||_F^2`` regularizer.
            lambda_orth: Weight for ``||R^T R - I||_F^2`` orthogonality
                penalty.
            gamma_div: Weight for diversity penalty
                ``D(Phi_g, Phi_l_perp)``.
            fd_epsilon: Finite-difference perturbation magnitude.
                Smaller values give more accurate gradients but are more
                sensitive to numerical noise.
        """
        self.lr = lr
        self.lambda_r = lambda_r
        self.lambda_orth = lambda_orth
        self.gamma_div = gamma_div
        self.fd_epsilon = fd_epsilon

    def _evaluate_objective(
        self,
        R: Array,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
        config: TrainingConfig,
    ) -> float:
        """Evaluate outer objective for a given ``R``, holding discrete state fixed.

        Constructs a temporary ``TrainingLoop`` to build fused features
        with the candidate ``R``, then solves ridge regression and
        evaluates the full outer objective.

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

        # Build features with candidate R.
        embedder = (
            state.continuous.theta.get("embedder") if state.continuous.theta else None
        )
        if embedder is None:
            raise RuntimeError("Embedder not found in state")

        embeddings = embedder.embed(X_batch)
        projector = Projector(R)
        U = projector.transform(embeddings)

        phi = loop._build_fused_features(U, state.discrete)

        # Solve ridge on mini-batch to get coefficients for this R.
        from aware_kernel.solver.ridge import DirectRidgeSolver

        solver = DirectRidgeSolver(lambda_reg=config.lambda_reg)
        w = solver.solve(phi, y_batch)

        # Split fused features into global and local for the diversity
        # penalty.  This is an approximation: during continuous updates
        # we skip the full orthogonalization and use the fused feature
        # splitting heuristic.
        r_g = state.discrete.M_g.shape[1] if state.discrete.M_g is not None else 0
        m_l = state.discrete.A.shape[0] if state.discrete.A is not None else 0

        if phi.shape[1] == r_g + m_l:
            phi_g = phi[:, :r_g]
            phi_l = phi[:, r_g:]
            # Approximate phi_l_perp as phi_l (orthogonalization is not
            # re-computed during continuous updates).
            phi_l_perp = phi_l
        else:
            # Fallback: zero features if dimensions don't match.
            phi_g = np.zeros((phi.shape[0], 1))
            phi_l_perp = np.zeros((phi.shape[0], 1))

        return compute_outer_objective(
            y=y_batch,
            phi=phi,
            w=w,
            R=R,
            phi_g=phi_g,
            phi_l_perp=phi_l_perp,
            lambda_r=self.lambda_r,
            lambda_orth=self.lambda_orth,
            gamma_div=self.gamma_div,
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
        """Compute gradient of outer objective w.r.t. ``R`` via central finite differences.

        Uses simultaneous perturbation: sample random direction matrices
        ``D`` with entries in ``{-1, +1}``, evaluate the objective at
        ``R + eps * D`` and ``R - eps * D``, and average the directional
        derivatives.

        The number of directions scales with the matrix size:
        ``n_dir = max(1, min(5, d^2 / 100))``.

        Args:
            R: Current projection matrix of shape ``(d, d)``.
            state: Current state.
            X_batch: Mini-batch inputs.
            y_batch: Mini-batch targets.
            config: Training configuration.
            rng: Random generator for direction sampling.

        Returns:
            Gradient matrix of same shape as ``R``.
        """
        d1, d2 = R.shape
        # Scale the number of random directions with the matrix size.
        # More directions improve gradient accuracy for larger matrices.
        n_dir = max(1, min(5, d1 * d2 // 100))

        grad = np.zeros_like(R)
        for _ in range(n_dir):
            # Random direction with entries in {-1, +1}, normalized.
            D = rng.choice([-1.0, 1.0], size=R.shape)
            D = D / np.linalg.norm(D, "fro")

            R_plus = R + self.fd_epsilon * D
            R_minus = R - self.fd_epsilon * D

            obj_plus = self._evaluate_objective(R_plus, state, X_batch, y_batch, config)
            obj_minus = self._evaluate_objective(
                R_minus, state, X_batch, y_batch, config
            )

            # Central difference approximation of the directional derivative.
            grad += (obj_plus - obj_minus) / (2.0 * self.fd_epsilon) * D

        result: Array = grad / n_dir
        return result

    def step(
        self,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
        config: TrainingConfig,
        rng: np.random.Generator,
    ) -> FullState:
        """Perform one gradient descent step on ``R``.

        Computes the finite-difference gradient estimate and applies a
        single gradient descent update: ``R_new = R - lr * grad``.

        Args:
            state: Current training state.
            X_batch: Mini-batch inputs.
            y_batch: Mini-batch targets.
            config: Training configuration.
            rng: Random generator for gradient estimation.

        Returns:
            Updated state with new ``R``.

        Raises:
            RuntimeError: If ``R`` is not initialized in continuous state.
        """
        R = state.continuous.R
        if R is None:
            raise RuntimeError("R is not initialized in continuous state")

        grad = self._compute_gradient_fd(R, state, X_batch, y_batch, config, rng)
        new_R = R - self.lr * grad

        new_continuous = state.continuous.copy_with(R=new_R)
        return state.copy_with(continuous=new_continuous)
