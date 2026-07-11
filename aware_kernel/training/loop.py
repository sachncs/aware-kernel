"""Training loop integrating continuous updates and refresh decisions.

Implements Section 3.2 (per-step loop) and Section 3.3 (discrete refresh
pipeline) of the method blueprint.

The ``TrainingLoop`` is the central orchestrator of the aware-kernel
training process.  It manages:

1. **Initialization**: Creates the embedder, projection matrix, initial
   discrete basis (via the refresh pipeline), and solves for the initial
   ridge coefficients.
2. **Per-step loop**: For each training step:
   a. Sample a mini-batch.
   b. Perform a continuous update on ``R`` via the outer-loop optimizer.
   c. Evaluate refresh trigger and execute refresh if needed.
   d. Increment the step counter.
3. **Evaluation**: Compute prediction RMSE on held-out data.

The loop integrates all major subsystems: embedding, projection, basis
construction, feature fusion, ridge solving, refresh control, and budget
tracking.

Design rationale
----------------
The loop is designed to be stateful but not stateful in a way that
prevents inspection.  All state is encapsulated in ``FullState`` objects,
which are passed through the loop and can be inspected at any point.
The loop itself holds only configuration and lightweight references to
subsystem instances (solver, optimizer, budget accountant).
"""

import numpy as np

from aware_kernel.aware.config import TrainingConfig
from aware_kernel.aware.state import ContinuousState, DiscreteState, FullState
from aware_kernel.aware.types import Array
from aware_kernel.embedding.embedder import DenseEmbedder
from aware_kernel.embedding.projector import Projector
from aware_kernel.fusion.builder import FusedFeatureBuilder
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.inference.predictor import Predictor
from aware_kernel.local_corrective.sparse_features import build_local_features
from aware_kernel.refresh.budget import BudgetAccountant
from aware_kernel.refresh.controller import should_refresh, transition_state
from aware_kernel.refresh.drift import compute_drift
from aware_kernel.refresh.pipeline import run_refresh_pipeline
from aware_kernel.solver.ridge import DirectRidgeSolver
from aware_kernel.training.callbacks import Callback
from aware_kernel.training.optimizer import OuterObjectiveOptimizer


class TrainingLoop:
    """Main training loop for aware-kernel.

    Orchestrates the hybrid continuous-discrete training process,
    including initialization, per-step updates, refresh decisions,
    and evaluation.

    Attributes:
        config: Training configuration.
        callbacks: List of training callbacks.
        rng: Random generator for reproducibility.
        solver: Direct ridge solver for normal equations.
        optimizer: Outer-loop optimizer for ``R``.
        budget: Budget accountant for refresh cost tracking.
        refresh_count: Number of refreshes performed so far.
    """

    def __init__(
        self,
        config: TrainingConfig,
        callbacks: list[Callback] | None = None,
    ) -> None:
        """Initialize training loop.

        Args:
            config: Master training configuration.
            callbacks: Optional list of callbacks for monitoring and
                checkpointing.
        """
        self.config = config
        self.callbacks = callbacks or []
        self.rng = np.random.default_rng(config.seed)
        self.solver = DirectRidgeSolver(
            lambda_reg=config.lambda_reg,
            kappa_threshold=config.numerics.kappa_threshold,
        )
        self.optimizer = OuterObjectiveOptimizer(
            lr=config.lr,
            lambda_r=config.lambda_r,
            lambda_orth=config.lambda_orth,
            gamma_div=0.0
            if config.ablation.disable_diversity_penalty
            else config.gamma_div,
            fd_epsilon=config.fd_epsilon,
        )
        self.budget = BudgetAccountant(total_budget=config.total_refresh_budget)
        self.refresh_count = 0
        self.R_ref: Array | None = None

    def initialize_state(
        self,
        X: Array,
        y: Array,
    ) -> FullState:
        """Initialize full training state from data.

        Performs the complete initialization sequence:

        1. Create a ``DenseEmbedder`` with random weights.
        2. Initialize ``R`` to the identity matrix.
        3. Project embeddings through ``R``.
        4. Run the initial refresh pipeline to build the discrete basis.
        5. Build fused features and solve for initial ridge coefficients.

        Args:
            X: Input data of shape ``(n, input_dim)``.
            y: Targets of shape ``(n,)``.

        Returns:
            Initialized ``FullState`` with all parameters set.
        """
        n, input_dim = X.shape
        embedder = DenseEmbedder(
            input_dim=input_dim,
            output_dim=self.config.embedding_dim,
            rng=self.rng,
        )
        R = np.eye(self.config.embedding_dim)
        continuous = ContinuousState(theta={"embedder": embedder}, R=R)

        # Project embeddings through the identity to get the initial
        # projected representation for basis construction.
        embeddings = embedder.embed(X)
        projector = Projector(R)
        U = projector.transform(embeddings)

        # Run the initial refresh pipeline to build the discrete basis
        # (landmarks, anchors, whitening map, calibration, gate).
        discrete = run_refresh_pipeline(
            FullState(continuous=continuous, step=0),
            U,
            y,
            self.config,
            self.rng,
        )

        # Build fused features and solve for initial coefficients.
        phi = self._build_fused_features(U, discrete)
        w = self.solver.solve(phi, y)

        return FullState(
            continuous=continuous,
            discrete=discrete,
            step=0,
            w=w,
        )

    def continuous_update(
        self,
        state: FullState,
        X_batch: Array,
        y_batch: Array,
    ) -> FullState:
        """Perform a continuous update step on the projection matrix ``R``.

        Uses the outer-objective optimizer to compute a gradient descent
        step on ``R`` while holding the discrete basis fixed.  The
        gradient is estimated via central finite differences on the
        mini-batch.

        Args:
            state: Current state.
            X_batch: Mini-batch of inputs of shape ``(batch_size, d)``.
            y_batch: Mini-batch of targets of shape ``(batch_size,)``.

        Returns:
            Updated state with new ``R``.
        """
        new_state = self.optimizer.step(
            state=state,
            X_batch=X_batch,
            y_batch=y_batch,
            config=self.config,
            rng=self.rng,
        )
        return new_state

    def maybe_refresh(
        self,
        state: FullState,
        X_val: Array,
        y_val: Array,
        val_gain: float = 1.0,
    ) -> FullState:
        """Evaluate refresh trigger and execute refresh if needed.

        Checks all five refresh conditions (drift, cooldown, warmup,
        hysteresis, budget) and, if triggered, executes the full
        discrete refresh pipeline.

        Args:
            state: Current state.
            X_val: Validation inputs for refresh pipeline.
            y_val: Validation targets for refresh pipeline.
            val_gain: Estimated validation gain (``Delta L_val``).

        Returns:
            Updated state (possibly with new discrete basis).
        """
        if self.config.ablation.disable_refresh:
            return state

        # Compute drift against the reference R from the last refresh.
        # If no refresh has occurred yet, approximate drift as a small
        # value proportional to the step count.
        if self.R_ref is not None and state.continuous.R is not None:
            drift = compute_drift(state.continuous.R, self.R_ref)
        else:
            drift = 0.001 * state.step

        # Adjust controller config for ablations.
        effective_t_cool = (
            0 if self.config.ablation.disable_cooldown else self.config.refresh.t_cool
        )
        effective_b_t = (
            1 if self.config.ablation.disable_hysteresis else state.discrete.b_t
        )

        trigger = should_refresh(
            state=state.copy_with(discrete=state.discrete.copy_with(b_t=effective_b_t)),
            drift=drift,
            val_gain=val_gain,
            refresh_cost=self.config.refresh_cost,
            config=self.config.refresh.copy_with(t_cool=effective_t_cool),
        )

        if trigger and self.budget.remaining >= self.config.refresh_cost:
            # Execute full refresh pipeline on validation data.
            embedder = (
                state.continuous.theta.get("embedder")
                if state.continuous.theta
                else None
            )
            if embedder is None:
                return state
            if state.continuous.R is None:
                return state
            embeddings = embedder.embed(X_val)
            projector = Projector(state.continuous.R)
            U_val = projector.transform(embeddings)

            new_discrete = run_refresh_pipeline(
                state,
                U_val,
                y_val,
                self.config,
                self.rng,
            )

            # Rebuild features for validation data and re-solve.
            phi_val = self._build_fused_features(U_val, new_discrete)
            new_w = self.solver.solve(phi_val, y_val)

            new_state = state.copy_with(
                discrete=new_discrete,
                w=new_w,
            )
            new_state = transition_state(new_state, refreshed=True)

            # Record cost and update reference R for drift computation.
            self.budget.record_refresh(self.config.refresh_cost)
            self.refresh_count += 1
            if state.continuous.R is None:
                return state
            self.R_ref = state.continuous.R.copy()

            for cb in self.callbacks:
                cb.on_refresh(new_state.step, new_state)
            return new_state

        return state

    def _build_fused_features(
        self,
        U: Array,
        discrete: DiscreteState,
    ) -> Array:
        """Build fused features from projected embeddings and discrete state.

        Constructs global features via the Nyström basis, local features
        via sparse k-NN RBF, and fuses them with calibration and gating.

        Args:
            U: Projected embeddings of shape ``(n, d)``.
            discrete: Discrete basis state.

        Returns:
            Fused features of shape ``(n, m)`` where ``m = r_g + m_l``.
        """
        if discrete.Z is None or discrete.A is None or discrete.M_g is None:
            raise ValueError("Discrete state is not initialized")

        # Build global features via the Nyström basis.
        # We construct a temporary NystromGlobalBasis from the stored
        # landmarks and whitening map (the eigenvectors are not needed
        # for feature construction).
        global_basis = NystromGlobalBasis(
            Z=discrete.Z,
            M_g=discrete.M_g,
            U=np.zeros((discrete.Z.shape[0], discrete.Z.shape[0])),
            eigenvalues=np.zeros(discrete.Z.shape[0]),
            r_g=discrete.M_g.shape[1],
        )
        phi_g = global_basis.build_features(U)

        # Build local features via sparse k-NN RBF.
        phi_l, _ = build_local_features(
            U,
            discrete.A,
            tau=self.config.refresh.tau_local,
            k=self.config.refresh.k_local,
        )

        # Fuse with calibration and gating.
        # Note: orthogonalization is skipped in the per-step feature
        # construction for efficiency.  The full orthogonalization is
        # performed during the refresh pipeline.
        builder = FusedFeatureBuilder(
            c_g=discrete.c_g,
            c_l=discrete.c_l,
            rho=discrete.rho,
        )
        return builder.build(phi_g, phi_l)

    def evaluate(
        self,
        state: FullState,
        X: Array,
        y: Array,
    ) -> dict[str, float]:
        """Evaluate current model on data.

        Computes the RMSE of the current predictor on the given data.

        Args:
            state: Current training state.
            X: Inputs of shape ``(n, d)``.
            y: Targets of shape ``(n,)``.

        Returns:
            Dictionary containing at least ``{"rmse": float}``.
        """
        if state.w is None:
            return {"rmse": float("inf")}

        embedder = (
            state.continuous.theta.get("embedder") if state.continuous.theta else None
        )
        if embedder is None:
            return {"rmse": float("inf")}

        if state.continuous.R is None:
            return {"rmse": float("inf")}

        embeddings = embedder.embed(X)
        projector = Projector(state.continuous.R)
        U = projector.transform(embeddings)

        phi = self._build_fused_features(U, state.discrete)
        predictor = Predictor(w=state.w)
        y_pred = predictor.predict(phi)

        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        return {"rmse": rmse}
