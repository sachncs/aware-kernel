"""Training loop integrating continuous updates and refresh decisions.

Implements Section 3.2 (per-step loop) and Section 3.3 (discrete refresh pipeline).
"""

from typing import List, Optional

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
from aware_kernel.memory.base import BaseMemoryAccumulator
from aware_kernel.memory.cached import CachedMemoryAccumulator
from aware_kernel.memory.streamed import StreamedMemoryAccumulator
from aware_kernel.refresh.budget import BudgetAccountant
from aware_kernel.refresh.controller import should_refresh, transition_state
from aware_kernel.refresh.drift import compute_drift
from aware_kernel.refresh.pipeline import run_refresh_pipeline
from aware_kernel.solver.ridge import DirectRidgeSolver
from aware_kernel.training.callbacks import Callback
from aware_kernel.training.optimizer import OuterObjectiveOptimizer


class TrainingLoop:
    """Main training loop for aware-kernel."""

    def __init__(
        self,
        config: TrainingConfig,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """Initialize training loop.

        Args:
            config: Training configuration.
            callbacks: Optional list of callbacks.
        """
        self._config = config
        self._callbacks = callbacks or []
        self._rng = np.random.default_rng(config.seed)
        self._solver = DirectRidgeSolver(
            lambda_reg=config.lambda_reg,
            kappa_threshold=config.numerics.kappa_threshold,
        )
        self._optimizer = OuterObjectiveOptimizer(
            lr=config.lr,
            lambda_r=config.lambda_r,
            lambda_orth=config.lambda_orth,
            gamma_div=0.0 if config.ablation.disable_diversity_penalty else config.gamma_div,
            fd_epsilon=config.fd_epsilon,
        )
        self._budget = BudgetAccountant(total_budget=config.total_refresh_budget)
        self._refresh_count = 0

    def initialize_state(
        self,
        X: Array,
        y: Array,
    ) -> FullState:
        """Initialize full training state from data.

        Args:
            X: Input data of shape (n, input_dim).
            y: Targets of shape (n,).

        Returns:
            Initialized FullState.
        """
        n, input_dim = X.shape
        embedder = DenseEmbedder(
            input_dim=input_dim,
            output_dim=self._config.embedding_dim,
            rng=self._rng,
        )
        R = np.eye(self._config.embedding_dim)
        continuous = ContinuousState(theta={"embedder": embedder}, R=R)

        # Initial projection
        embeddings = embedder.embed(X)
        projector = Projector(R)
        U = projector.transform(embeddings)

        # Run initial refresh pipeline to get discrete state
        discrete = run_refresh_pipeline(
            FullState(continuous=continuous, step=0),
            U,
            y,
            self._config,
            self._rng,
        )

        # Build initial features
        phi = self._build_fused_features(U, discrete)

        # Solve for initial coefficients
        w = self._solver.solve(phi, y)

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
        """Perform a continuous update step on representation parameters.

        Uses the outer-objective optimizer to compute a gradient descent step
        on R while holding the discrete basis fixed.

        Args:
            state: Current state.
            X_batch: Mini-batch of inputs.
            y_batch: Mini-batch of targets.

        Returns:
            Updated state with new R.
        """
        new_state = self._optimizer.step(
            state=state,
            X_batch=X_batch,
            y_batch=y_batch,
            config=self._config,
            rng=self._rng,
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

        Args:
            state: Current state.
            X_val: Validation inputs.
            y_val: Validation targets.
            val_gain: Estimated validation gain.

        Returns:
            Updated state (possibly with new discrete basis).
        """
        if self._config.ablation.disable_refresh:
            return state

        # Compute actual drift against the reference R at last refresh
        reference_R = state.continuous.R
        # The discrete state stores no reference R, so we approximate by
        # comparing current R to identity (initial) if no refresh has occurred.
        # In practice, the pipeline should be extended to snapshot R at refresh.
        # For now, use a proxy: condition-number-based drift or step-based drift.
        # We'll use a proper drift metric when R_ref is available.
        if hasattr(self, "_R_ref"):
            drift = compute_drift(state.continuous.R, self._R_ref)
        else:
            drift = 0.001 * state.step

        # Adjust controller config for ablations
        effective_t_cool = 0 if self._config.ablation.disable_cooldown else self._config.refresh.t_cool
        effective_b_t = 1 if self._config.ablation.disable_hysteresis else state.discrete.b_t

        trigger = should_refresh(
            state=state.copy_with(
                discrete=state.discrete.copy_with(b_t=effective_b_t)
            ),
            drift=drift,
            val_gain=val_gain,
            refresh_cost=self._config.refresh_cost,
            config=self._config.refresh.copy_with(t_cool=effective_t_cool),
        )

        if trigger and self._budget.remaining >= self._config.refresh_cost:
            # Execute full refresh pipeline on validation data
            embedder = state.continuous.theta.get("embedder") if state.continuous.theta else None
            if embedder is None:
                return state
            embeddings = embedder.embed(X_val)
            projector = Projector(state.continuous.R)
            U_val = projector.transform(embeddings)

            new_discrete = run_refresh_pipeline(
                state,
                U_val,
                y_val,
                self._config,
                self._rng,
            )

            # Rebuild features for all validation data and re-solve
            phi_val = self._build_fused_features(U_val, new_discrete)
            new_w = self._solver.solve(phi_val, y_val)

            new_state = state.copy_with(
                discrete=new_discrete,
                w=new_w,
            )
            new_state = transition_state(new_state, refreshed=True)

            self._budget.record_refresh(self._config.refresh_cost)
            self._refresh_count += 1
            self._R_ref = state.continuous.R.copy()

            for cb in self._callbacks:
                cb.on_refresh(new_state.step, new_state)
            return new_state

        return state

    def _build_fused_features(
        self,
        U: Array,
        discrete: DiscreteState,
    ) -> Array:
        """Build fused features from projected embeddings and discrete state.

        Args:
            U: Projected embeddings of shape (n, d).
            discrete: Discrete basis state.

        Returns:
            Fused features of shape (n, m).
        """
        if discrete.Z is None or discrete.A is None or discrete.M_g is None:
            raise ValueError("Discrete state is not initialized")

        # Global features
        global_basis = NystromGlobalBasis(
            Z=discrete.Z,
            M_g=discrete.M_g,
            U=np.zeros((discrete.Z.shape[0], discrete.Z.shape[0])),
            eigenvalues=np.zeros(discrete.Z.shape[0]),
            r_g=discrete.M_g.shape[1],
        )
        phi_g = global_basis.build_features(U)

        # Local features
        phi_l, _ = build_local_features(
            U,
            discrete.A,
            tau=self._config.refresh.tau_local,
            k=self._config.refresh.k_local,
        )

        # For simplicity, skip orthogonalization in the stand-in loop
        # In production, this would orthogonalize and calibrate
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
    ) -> dict:
        """Evaluate current model on data.

        Args:
            state: Current state.
            X: Inputs.
            y: Targets.

        Returns:
            Dictionary of metrics.
        """
        if state.w is None:
            return {"rmse": float("inf")}

        embedder = state.continuous.theta.get("embedder") if state.continuous.theta else None
        if embedder is None:
            return {"rmse": float("inf")}

        embeddings = embedder.embed(X)
        projector = Projector(state.continuous.R)
        U = projector.transform(embeddings)

        phi = self._build_fused_features(U, state.discrete)
        predictor = Predictor(w=state.w)
        y_pred = predictor.predict(phi)

        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        return {"rmse": rmse}
