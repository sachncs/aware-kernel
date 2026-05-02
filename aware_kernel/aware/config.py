"""Configuration dataclasses for aware-kernel.

All hyperparameters and numerical thresholds are centralized here
to ensure consistency across modules.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryMode(Enum):
    """Operating mode for feature memory."""

    CACHED = "cached"
    STREAMED = "streamed"


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical stability and precision configuration.

    Attributes:
        tau_eig: Eigenvalue threshold for soft truncation.
        alpha_epsilon: Scale factor for dataset-scale epsilon.
        epsilon_c: Minimum calibration scaling to prevent collapse.
        lambda_min: Floor on ridge regularization for SPD normal equations.
        eta_o: Orthogonalization ridge regularizer scale factor.
        beta: Scale factor for orthogonalization regularizer.
        kappa_threshold: Maximum acceptable condition number.
        precision: Default precision for accumulations ("float32" or "float64").
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
        """Return a new NumericsConfig with updated fields."""
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

    Attributes:
        delta_hi: High threshold for drift metric to trigger refresh.
        t_cool: Minimum steps between refreshes (cooldown).
        t_warmup: Minimum step before first refresh allowed.
        gamma_cost: Validation gain threshold scaled by refresh cost.
        alpha_a: Mix weight for residual-aware anchor sampling (0=coverage, 1=residual).
        tau_local: Bandwidth for local sparse features.
        k_local: Number of nearest neighbors for sparse features.
    """

    delta_hi: float = 0.1
    t_cool: int = 50
    t_warmup: int = 10
    gamma_cost: float = 0.01
    alpha_a: float = 0.5
    tau_local: float = 0.1
    k_local: int = 5

    def copy_with(self, **kwargs) -> "RefreshConfig":
        """Return a new RefreshConfig with updated fields."""
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
    """Ablations for controlled component removal.

    Implements Section 7 ablation requirements:
        - disable_refresh: skip all discrete refreshes (static basis).
        - disable_hysteresis: force b_t = 1 permanently.
        - disable_cooldown: set effective cooldown to 0.
        - disable_residual_aware_anchors: use coverage-only (kmeans++) sampling.
        - disable_orthogonalization: skip local orthogonalization against global.
        - disable_diversity_penalty: set gamma_div = 0 in outer objective.
        - static_scaling: freeze calibration scalars after first refresh.
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

    Attributes:
        embedding_dim: Dimension of continuous embedding space.
        m_g: Global basis rank budget (landmarks).
        m_l: Local corrective rank budget (anchors).
        lambda_reg: Ridge regularization parameter.
        memory_mode: Cached or streamed feature accumulation.
        numerics: Numerical stability configuration.
        refresh: Refresh controller configuration.
        ablation: Ablation switches.
        max_steps: Maximum training steps.
        eval_freq: Evaluation frequency in steps.
        seed: Random seed for reproducibility.
        lr: Learning rate for outer-loop optimizer on R.
        lambda_r: Frobenius regularizer weight on R.
        lambda_orth: Orthogonality penalty weight on R.
        gamma_div: Diversity penalty weight in outer objective.
        fd_epsilon: Finite-difference perturbation magnitude.
        total_refresh_budget: Total amortized refresh budget (inf disables budgeting).
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
