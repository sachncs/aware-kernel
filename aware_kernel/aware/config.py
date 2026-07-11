"""Configuration dataclasses for aware-kernel.

All hyperparameters and numerical thresholds are centralized here to ensure
consistency across modules.  The configuration hierarchy mirrors the logical
structure of the method:

* ``NumericsConfig`` -- Controls numerical stability (eigenvalue clipping,
  epsilon scaling, condition-number thresholds, precision).
* ``RefreshConfig`` -- Governs the adaptive refresh controller (drift
  threshold, cooldown, warmup, budget, anchor sampling mix).
* ``AblationConfig`` -- Boolean switches for controlled component removal,
  corresponding to the ablation study in Section 7 of the method blueprint.
* ``TrainingConfig`` -- Top-level master configuration that composes the above
  sub-configs and adds training-specific parameters (rank budgets,
  regularization, learning rate, outer-loop weights).

All dataclasses are **frozen** (immutable) to prevent accidental mutation
during training.  Each provides a ``copy_with`` helper that returns a new
instance with selected fields overridden -- this is the idiomatic way to
adjust configuration at runtime without violating immutability.

Design rationale
----------------
Centralizing configuration avoids the "scattered magic number" problem
common in research codebases.  Every numerical constant that affects
correctness or stability lives in exactly one place and is documented
with its purpose.  The ``AblationConfig`` switches enable reproducible
ablation experiments without code duplication.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryMode(Enum):
    """Operating mode for feature memory accumulation.

    Controls how normal equations are accumulated during training:

    * ``CACHED`` -- Stores the full feature matrix ``Phi`` explicitly.
      Memory O(n * m), enables direct normal-equation construction and
      feature reconstruction.  Preferred for small-to-medium datasets.
    * ``STREAMED`` -- Accumulates ``S = Phi^T Phi`` and ``b = Phi^T y``
      directly.  Memory O(m^2), discards individual samples.  Preferred
      for large datasets where storing ``Phi`` is prohibitive.

    Both modes produce identical coefficients when given the same data
    and seed (verified by parity tests).
    """

    CACHED = "cached"
    STREAMED = "streamed"


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical stability and precision configuration.

    Centralizes all thresholds that control the trade-off between
    accuracy and numerical robustness.  These defaults were tuned on
    synthetic benchmarks to work across a range of dataset scales.

    Attributes:
        tau_eig: Eigenvalue threshold for soft truncation in the whitening
            map.  Eigenvalues below ``tau_eig`` are down-weighted via the
            soft-truncation formula rather than being hard-clipped, which
            avoids discontinuities in the feature map.
        alpha_epsilon: Scale factor for the dataset-scale epsilon used in
            whitening: ``epsilon = alpha_epsilon * tr(W) / m_g``.  This
            makes the stabilization epsilon proportional to the average
            eigenvalue of the kernel matrix.
        epsilon_c: Minimum calibration scaling to prevent feature collapse.
            Ensures ``c_g, c_l > 0`` even when feature traces are near
            zero.
        lambda_min: Floor on ridge regularization for SPD normal equations.
            The effective regularization is ``max(lambda_reg, lambda_min)``.
        eta_o: Ridge regularizer scale factor for the orthogonalization
            matrix ``P_g = Phi_g (Phi_g^T Phi_g + eta_o * I)^{-1} Phi_g^T``.
            Prevents singularity when the global feature matrix is
            rank-deficient.
        beta: Scale factor for the orthogonalization regularizer, used in
            conjunction with ``eta_o``.
        kappa_threshold: Maximum acceptable condition number for the normal
            equations matrix.  If exceeded, a ``ConditioningError`` is
            raised.
        precision: Default precision for accumulations (``"float32"`` or
            ``"float64"``).  Float64 is recommended for most use cases.
    """

    tau_eig: float = 1e-6
    alpha_epsilon: float = 1e-5
    epsilon_c: float = 1e-8
    lambda_min: float = 1e-6
    eta_o: float = 1e-4
    beta: float = 1e-3
    kappa_threshold: float = 1e12
    precision: str = "float64"

    def __post_init__(self) -> None:
        if self.precision not in {"float32", "float64"}:
            raise ValueError(f"precision must be float32 or float64, got {self.precision}")

    def copy_with(self, **kwargs) -> "NumericsConfig":
        """Return a new NumericsConfig with the specified fields overridden.

        This is the only way to modify a frozen configuration instance.
        All unspecified fields retain their current values.

        Args:
            **kwargs: Fields to override (e.g., ``tau_eig=1e-5``).

        Returns:
            A new ``NumericsConfig`` with updated values.
        """
        current = {
            "tau_eig": self.tau_eig,
            "alpha_epsilon": self.alpha_epsilon,
            "epsilon_c": self.epsilon_c,
            "lambda_min": self.lambda_min,
            "eta_o": self.eta_o,
            "beta": self.beta,
            "kappa_threshold": self.kappa_threshold,
            "precision": self.precision,
        }
        current.update(kwargs)
        return NumericsConfig(**current)


@dataclass(frozen=True)
class RefreshConfig:
    """Refresh controller configuration.

    Governs the adaptive refresh mechanism that determines *when* discrete
    basis parameters (landmarks, anchors, whitening maps) should be
    recomputed.  The refresh decision combines five conditions:

    1. **Drift threshold** (``delta_hi``): Representation drift must
       exceed this value.
    2. **Cooldown** (``t_cool``): Sufficient steps must have elapsed
       since the last refresh.
    3. **Warmup** (``t_warmup``): The training step must exceed this
       minimum before any refresh is allowed.
    4. **Hysteresis** (``b_t``): The hysteresis flag must be active.
    5. **Budget** (``gamma_cost * refresh_cost``): The estimated
       validation gain must justify the amortized refresh cost.

    Attributes:
        delta_hi: High threshold for the relative Frobenius-norm drift
            metric ``Delta = ||R_t - R_{t_r}||_F / ||R_{t_r}||_F``.
        t_cool: Minimum number of training steps between consecutive
            refreshes.
        t_warmup: Minimum training step before the first refresh is
            permitted.
        gamma_cost: Validation gain threshold scaled by refresh cost.
            A refresh only triggers if ``Delta L_val > gamma_cost * C_refresh``.
        alpha_a: Mix weight for residual-aware anchor sampling.  ``0.0``
            means pure coverage-based sampling (k-means++), ``1.0`` means
            pure residual-based sampling.
        tau_local: Bandwidth parameter for the local sparse RBF features.
            Controls the locality of the corrective basis.
        k_local: Number of nearest neighbors for the k-NN sparse feature
            computation.  Must be ``<= m_l``.
    """

    delta_hi: float = 0.1
    t_cool: int = 50
    t_warmup: int = 10
    gamma_cost: float = 0.01
    alpha_a: float = 0.5
    tau_local: float = 0.1
    k_local: int = 5

    def copy_with(self, **kwargs) -> "RefreshConfig":
        """Return a new RefreshConfig with the specified fields overridden.

        Args:
            **kwargs: Fields to override.

        Returns:
            A new ``RefreshConfig`` with updated values.
        """
        current = {
            "delta_hi": self.delta_hi,
            "t_cool": self.t_cool,
            "t_warmup": self.t_warmup,
            "gamma_cost": self.gamma_cost,
            "alpha_a": self.alpha_a,
            "tau_local": self.tau_local,
            "k_local": self.k_local,
        }
        current.update(kwargs)
        return RefreshConfig(**current)


@dataclass(frozen=True)
class AblationConfig:
    """Ablation switches for controlled component removal.

    Implements the ablation study requirements from Section 7 of the method
    blueprint.  Each flag disables a specific algorithmic component while
    keeping the rest of the pipeline intact, enabling rigorous
    component-by-component evaluation.

    Design rationale
        Ablations are implemented as configuration flags rather than
        separate code paths to avoid branching complexity and to ensure
        that ablated experiments share the exact same code as full-model
        experiments.

    Attributes:
        disable_refresh: If ``True``, skip all discrete refreshes and
            maintain a static basis throughout training.
        disable_hysteresis: If ``True``, force ``b_t = 1`` permanently,
            disabling the hysteresis dampening mechanism.
        disable_cooldown: If ``True``, set effective cooldown to 0,
            allowing refreshes at every step (subject to drift threshold).
        disable_residual_aware_anchors: If ``True``, use pure
            coverage-based (k-means++) anchor sampling instead of the
            residual-aware blend.
        disable_orthogonalization: If ``True``, skip the local-to-global
            orthogonalization step, allowing local features to overlap
            with the global subspace.
        disable_diversity_penalty: If ``True``, set ``gamma_div = 0``,
            removing the diversity penalty from the outer objective.
        static_scaling: If ``True``, freeze calibration scalars ``c_g``
            and ``c_l`` after the first refresh, preventing them from
            updating with the evolving feature statistics.
    """

    disable_refresh: bool = False
    disable_hysteresis: bool = False
    disable_cooldown: bool = False
    disable_residual_aware_anchors: bool = False
    disable_orthogonalization: bool = False
    disable_diversity_penalty: bool = False
    static_scaling: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Master training configuration.

    Top-level configuration that composes all sub-configurations and adds
    training-specific parameters.  This is the single object passed to
    ``TrainingLoop`` and ``AwareKernelEstimator``.

    Attributes:
        embedding_dim: Dimension of the continuous embedding space.
            Larger values increase expressivity but also increase the cost
            of projection and basis construction.
        m_g: Global basis rank budget (number of landmarks).  Controls
            the capacity of the Nyström global feature map.
        m_l: Local corrective rank budget (number of anchors).  Must be
            ``<= 0.25 * m_g`` to ensure the local basis remains a small
            correction rather than a competing representation.
        lambda_reg: Ridge regularization parameter for the normal
            equations.  Must be ``>= numerics.lambda_min``.
        memory_mode: Cached or streamed feature accumulation.
        numerics: Numerical stability configuration.
        refresh: Refresh controller configuration.
        ablation: Ablation switches.
        max_steps: Maximum number of training steps.
        eval_freq: Evaluation frequency in steps.  Metrics are computed
            every ``eval_freq`` steps.
        seed: Random seed for reproducibility.  ``None`` for non-
            deterministic behavior.
        lr: Learning rate for the outer-loop gradient descent on the
            projection matrix ``R``.
        lambda_r: Frobenius regularizer weight on ``R``.  Encourages
            small-norm projections.
        lambda_orth: Orthogonality penalty weight on ``R``.  Penalizes
            deviation from ``R^T R = I``.
        gamma_div: Diversity penalty weight in the outer objective.
            Encourages global and local features to span complementary
            subspaces.
        fd_epsilon: Finite-difference perturbation magnitude for gradient
            estimation in the outer-loop optimizer.
        total_refresh_budget: Total amortized refresh budget over the
            training horizon.  ``float("inf")`` disables budgeting.
        refresh_cost: Fixed cost per refresh for budget accounting.
    """

    embedding_dim: int = 64
    m_g: int = 512
    m_l: int = 128
    lambda_reg: float = 1e-3
    memory_mode: MemoryMode = MemoryMode.CACHED
    numerics: NumericsConfig = field(default_factory=lambda: NumericsConfig())
    refresh: RefreshConfig = field(default_factory=lambda: RefreshConfig())
    ablation: AblationConfig = field(default_factory=lambda: AblationConfig())
    max_steps: int = 1000
    eval_freq: int = 10
    seed: Optional[int] = None
    lr: float = 1e-4
    lambda_r: float = 0.0
    lambda_orth: float = 0.0
    gamma_div: float = 0.0
    fd_epsilon: float = 1e-5
    total_refresh_budget: float = float("inf")
    refresh_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.m_l > 0.25 * self.m_g:
            raise ValueError(
                f"Local rank m_l ({self.m_l}) must be <= 0.25 * m_g ({0.25 * self.m_g})"
            )
        if self.lambda_reg < self.numerics.lambda_min:
            raise ValueError(
                f"lambda_reg ({self.lambda_reg}) must be >= lambda_min "
                f"({self.numerics.lambda_min})"
            )
