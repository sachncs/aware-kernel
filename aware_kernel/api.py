"""Public API: sklearn-compatible estimator for aware-kernel.

This module provides ``AwareKernelEstimator``, a drop-in scikit-learn
estimator that wraps the full aware-kernel training pipeline.  It bridges
the library's internal ``TrainingLoop`` with the familiar ``fit``/``predict``/``score``
interface, making aware-kernel compatible with sklearn utilities like
``GridSearchCV``, ``Pipeline``, and ``cross_val_score``.

Example::

    from aware_kernel import AwareKernelEstimator

    model = AwareKernelEstimator(embedding_dim=8, m_g=32, m_l=8, lambda_reg=1e-2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

All hyperparameters map directly to ``TrainingConfig`` fields.  Ablation
flags (``disable_refresh``, ``disable_hysteresis``, etc.) allow controlled
ablation experiments from the public API without modifying internal config.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from aware_kernel.aware.config import (
    AblationConfig,
    MemoryMode,
    RefreshConfig,
    TrainingConfig,
)
from aware_kernel.training.callbacks import Callback, LoggingCallback
from aware_kernel.training.loop import TrainingLoop


class AwareKernelEstimator(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Sklearn-compatible estimator for refresh-aware hybrid kernel learning.

    This estimator implements a complete fit/predict interface backed by the
    internal ``TrainingLoop``.  It separates continuous parameters (updated every
    step via ridge regression on mini-batches) from discrete basis parameters
    (refreshed adaptively based on drift detection).

    The estimator satisfies the sklearn ``BaseEstimator`` and ``RegressorMixin``
    interfaces, enabling use with ``GridSearchCV``, ``Pipeline``, and standard
    cross-validation utilities.  The ``coef_`` attribute exposes the final ridge
    solution weights for introspection.

    Attributes:
        embedding_dim_: Resolved embedding dimension after fitting.
        state_: Final ``FullState`` containing all continuous and discrete
            parameters after training.
        config_: ``TrainingConfig`` constructed from hyperparameters.
        coef_: Ridge solution weights ``w`` of shape ``(m_fused,)`` where
            ``m_fused = m_g + 2 * m_l``.

    Design notes:
        - The estimator constructs ``TrainingConfig`` lazily in ``_build_config``
          so that hyperparameters remain mutable until ``fit`` is called.
        - Mini-batch size is fixed at 32 for computational efficiency on
          moderate datasets; larger batches may be added via config in the future.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        m_g: int = 512,
        m_l: int = 128,
        lambda_reg: float = 1e-3,
        memory_mode: str = "cached",
        max_steps: int = 1000,
        eval_freq: int = 10,
        seed: int | None = None,
        delta_hi: float = 0.1,
        t_cool: int = 50,
        t_warmup: int = 10,
        gamma_cost: float = 0.01,
        alpha_a: float = 0.5,
        tau_local: float = 0.1,
        k_local: int = 5,
        log_interval: int = 0,
        lr: float = 1e-4,
        lambda_r: float = 0.0,
        lambda_orth: float = 0.0,
        gamma_div: float = 0.0,
        fd_epsilon: float = 1e-5,
        total_refresh_budget: float = float("inf"),
        refresh_cost: float = 1.0,
        disable_refresh: bool = False,
        disable_hysteresis: bool = False,
        disable_cooldown: bool = False,
        disable_residual_aware_anchors: bool = False,
        disable_orthogonalization: bool = False,
        disable_diversity_penalty: bool = False,
        static_scaling: bool = False,
    ) -> None:
        """Initialize estimator.

        Args:
            embedding_dim: Dimension of continuous embedding space.
            m_g: Global basis rank budget (landmarks).
            m_l: Local corrective rank budget (anchors).
            lambda_reg: Ridge regularization parameter.
            memory_mode: "cached" or "streamed".
            max_steps: Maximum training steps.
            eval_freq: Evaluation frequency in steps.
            seed: Random seed for reproducibility.
            delta_hi: High threshold for drift metric to trigger refresh.
            t_cool: Minimum steps between refreshes (cooldown).
            t_warmup: Minimum step before first refresh allowed.
            gamma_cost: Validation gain threshold scaled by refresh cost.
            alpha_a: Mix weight for residual-aware anchor sampling.
            tau_local: Bandwidth for local sparse features.
            k_local: Number of nearest neighbors for sparse features.
            log_interval: If > 0, adds a LoggingCallback at this interval.
            lr: Learning rate for outer-loop optimizer on R.
            lambda_r: Frobenius regularizer weight on R.
            lambda_orth: Orthogonality penalty weight on R.
            gamma_div: Diversity penalty weight in outer objective.
            fd_epsilon: Finite-difference perturbation magnitude.
            total_refresh_budget: Total amortized refresh budget (inf disables).
            refresh_cost: Fixed cost per refresh for budget accounting.
            disable_refresh: Ablation: skip all discrete refreshes.
            disable_hysteresis: Ablation: force b_t = 1 permanently.
            disable_cooldown: Ablation: set effective cooldown to 0.
            disable_residual_aware_anchors: Ablation: use coverage-only sampling.
            disable_orthogonalization: Ablation: skip local orthogonalization.
            disable_diversity_penalty: Ablation: set gamma_div = 0.
            static_scaling: Ablation: freeze calibration scalars after first refresh.
        """
        self.embedding_dim = embedding_dim
        self.m_g = m_g
        self.m_l = m_l
        self.lambda_reg = lambda_reg
        self.memory_mode = memory_mode
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.seed = seed
        self.delta_hi = delta_hi
        self.t_cool = t_cool
        self.t_warmup = t_warmup
        self.gamma_cost = gamma_cost
        self.alpha_a = alpha_a
        self.tau_local = tau_local
        self.k_local = k_local
        self.log_interval = log_interval
        self.lr = lr
        self.lambda_r = lambda_r
        self.lambda_orth = lambda_orth
        self.gamma_div = gamma_div
        self.fd_epsilon = fd_epsilon
        self.total_refresh_budget = total_refresh_budget
        self.refresh_cost = refresh_cost
        self.disable_refresh = disable_refresh
        self.disable_hysteresis = disable_hysteresis
        self.disable_cooldown = disable_cooldown
        self.disable_residual_aware_anchors = disable_residual_aware_anchors
        self.disable_orthogonalization = disable_orthogonalization
        self.disable_diversity_penalty = disable_diversity_penalty
        self.static_scaling = static_scaling

    def _build_config(self) -> TrainingConfig:
        """Construct ``TrainingConfig`` from estimator hyperparameters.

        Maps the flat sklearn-style hyperparameters to the nested config
        structure used internally.  This separation keeps the public API
        flat (sklearn convention) while the internals use structured configs.

        Returns:
            TrainingConfig with all fields populated from hyperparameters.
        """
        mode = (
            MemoryMode.CACHED
            if self.memory_mode.lower() == "cached"
            else MemoryMode.STREAMED
        )
        refresh = RefreshConfig(
            delta_hi=self.delta_hi,
            t_cool=self.t_cool,
            t_warmup=self.t_warmup,
            gamma_cost=self.gamma_cost,
            alpha_a=self.alpha_a,
            tau_local=self.tau_local,
            k_local=self.k_local,
        )
        ablation = AblationConfig(
            disable_refresh=self.disable_refresh,
            disable_hysteresis=self.disable_hysteresis,
            disable_cooldown=self.disable_cooldown,
            disable_residual_aware_anchors=self.disable_residual_aware_anchors,
            disable_orthogonalization=self.disable_orthogonalization,
            disable_diversity_penalty=self.disable_diversity_penalty,
            static_scaling=self.static_scaling,
        )
        return TrainingConfig(
            embedding_dim=self.embedding_dim,
            m_g=self.m_g,
            m_l=self.m_l,
            lambda_reg=self.lambda_reg,
            memory_mode=mode,
            max_steps=self.max_steps,
            eval_freq=self.eval_freq,
            seed=self.seed,
            refresh=refresh,
            ablation=ablation,
            lr=self.lr,
            lambda_r=self.lambda_r,
            lambda_orth=self.lambda_orth,
            gamma_div=self.gamma_div,
            fd_epsilon=self.fd_epsilon,
            total_refresh_budget=self.total_refresh_budget,
            refresh_cost=self.refresh_cost,
        )

    def fit(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        callbacks: list[Callback] | None = None,
    ) -> "AwareKernelEstimator":
        """Fit the aware-kernel model.

        Runs the full training pipeline: embeds inputs, builds the global
        Nystr\"om basis, constructs local corrective features, and alternates
        between continuous ridge updates and adaptive discrete refreshes.

        Args:
            X: Training inputs of shape ``(n_samples, n_features)``.
            y: Training targets of shape ``(n_samples,)``.
            callbacks: Optional list of ``Callback`` instances for logging
                or custom monitoring during training.

        Returns:
            Self, the fitted estimator (enables method chaining).

        Side effects:
            Sets ``self.state_``, ``self.config_``, ``self.embedding_dim_``,
            and ``self.coef_`` as fitted attributes.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, y_numeric=True)
        self.config_ = self._build_config()
        cb = list(callbacks) if callbacks else []
        if self.log_interval > 0:
            cb.append(LoggingCallback(log_interval=self.log_interval))

        loop = TrainingLoop(self.config_, callbacks=cb)
        state = loop.initialize_state(X, y)

        for step in range(1, self.config_.max_steps + 1):
            batch_size = min(32, X.shape[0])
            batch_idx = np.random.default_rng(self.config_.seed).integers(
                0, X.shape[0], size=batch_size
            )
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            state = loop.continuous_update(state, X_batch, y_batch)
            state = loop.maybe_refresh(state, X, y)
            state = state.copy_with(step=step)

            if step % self.config_.eval_freq == 0:
                metrics = loop.evaluate(state, X, y)
                for callback in cb:
                    callback.on_eval(step, metrics)

        self.state_ = state
        self.embedding_dim_ = self.config_.embedding_dim
        self.coef_ = state.w
        return self

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict on new data.

        Reconstructs the full feature pipeline (embed → project → fuse)
        using the fitted state, then applies the ridge solution ``w``.

        Args:
            X: Inputs of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        check_is_fitted(self, attributes=["state_", "config_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        loop = TrainingLoop(self.config_)
        loop.evaluate(self.state_, X, np.zeros(X.shape[0]))
        # evaluate returns dict; we need actual predictions. Use the loop's
        # internal build logic manually since evaluate doesn't expose preds.
        embedder = (
            self.state_.continuous.theta.get("embedder")
            if self.state_.continuous.theta
            else None
        )
        if embedder is None:
            raise RuntimeError("Embedder not found in fitted state")
        from aware_kernel.embedding.projector import Projector
        from aware_kernel.inference.predictor import Predictor

        embeddings = embedder.embed(X)
        if self.state_.continuous.R is None:
            raise RuntimeError("Projection matrix R not found in fitted state")
        projector = Projector(self.state_.continuous.R)
        U = projector.transform(embeddings)
        phi = loop._build_fused_features(U, self.state_.discrete)
        if self.state_.w is None:
            raise RuntimeError("Ridge coefficients w not found in fitted state")
        predictor = Predictor(w=self.state_.w)
        return np.asarray(predictor.predict(phi))

    def score(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
        """Return the coefficient of determination R^2.

        Higher is better.  A score of 1.0 indicates perfect prediction;
        0.0 means the model predicts the mean of ``y``; negative scores
        indicate worse-than-mean performance.

        Args:
            X: Test inputs of shape ``(n_samples, n_features)``.
            y: True targets of shape ``(n_samples,)``.

        Returns:
            R^2 score (capped at 1.0).
        """
        from aware_kernel.evaluation.metrics import compute_r2

        y_pred = self.predict(X)
        return compute_r2(y, y_pred)
