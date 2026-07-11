"""State containers for continuous and discrete parameters.

The aware-kernel method maintains two logically distinct groups of parameters:

* **Continuous** (``ContinuousState``): The projection matrix ``R`` and
  embedding parameters ``theta``.  These are updated every training step
  via gradient descent in the outer-loop optimizer.
* **Discrete** (``DiscreteState``): Landmarks ``Z``, anchors ``A``,
  whitening map ``M_g``, calibration scalars ``c_g``/``c_l``, normalizers
  ``d``, and controller metadata (``t_r``, ``b_t``, ``rho``).  These are
  refreshed adaptively when representation drift exceeds a threshold.

``FullState`` composes both into a single immutable snapshot that is
threaded through the training loop.

Design rationale
----------------
All state containers are **frozen dataclasses** with ``copy_with`` helper
methods.  This immutability ensures that refresh boundaries are explicit:
a refresh produces a *new* ``DiscreteState`` rather than mutating the old
one, which eliminates a class of subtle bugs where stale references leak
into subsequent computation.  The ``copy_with`` pattern is used instead of
``dataclasses.replace`` to maintain consistency across the codebase.

Thread safety
-------------
Frozen dataclasses are inherently safe to share across threads because they
cannot be mutated after construction.  The ``copy_with`` method returns a
new instance, so concurrent readers never observe partial writes.
"""

from dataclasses import dataclass, field
from typing import Any

from aware_kernel.aware.types import Array


@dataclass(frozen=True)
class ContinuousState:
    """Continuously updated representation parameters.

    Holds the embedding parameters ``theta`` and the projection matrix
    ``R`` that are updated every training step via the outer-loop
    optimizer (gradient descent with finite-difference gradients).

    Attributes:
        theta: Embedding parameters stored as an opaque dictionary.
            The default ``DenseEmbedder`` is stored under the key
            ``"embedder"``.  Using a dict allows custom embedders to
            store arbitrary state without modifying this container.
        R: Projection matrix of shape ``(embedding_dim, embedding_dim)``.
            Initialized to the identity matrix and updated by the
            ``OuterObjectiveOptimizer``.  The projection is applied
            after L2-normalization: ``u = R * normalize(e)``.
    """

    theta: dict[Any, Any] | None = None
    R: Array | None = None

    def copy_with(self, **kwargs: Any) -> "ContinuousState":
        """Return a new ContinuousState with the specified fields overridden.

        Args:
            **kwargs: Fields to override (e.g., ``R=new_R``).

        Returns:
            A new ``ContinuousState`` with updated values.
        """
        current: dict[str, Any] = {
            "theta": self.theta,
            "R": self.R,
        }
        current.update(kwargs)
        return ContinuousState(**current)


@dataclass(frozen=True)
class DiscreteState:
    """Discrete basis state refreshed on trigger.

    Holds all parameters that are recomputed during a discrete refresh:
    landmarks, anchors, whitening map, calibration scalars, normalizers,
    and refresh controller metadata.

    The discrete state is the output of ``run_refresh_pipeline`` and is
    consumed by the feature builder and ridge solver.  It is replaced
    wholesale during a refresh -- individual fields are never updated
    independently.

    Attributes:
        Z: Global landmarks of shape ``(m_g, embedding_dim)``.  Selected
            via k-means++ from the projected embedding space.
        A: Local anchors of shape ``(m_l, embedding_dim)``.  Selected
            via residual-aware sampling (blend of coverage and residual
            weights).
        M_g: Whitening map for the global basis of shape ``(m_g, r_g)``.
            Constructed from the soft-truncated eigendecomposition of the
            kernel-on-landmarks matrix.
        c_g: Global calibration scalar.  Normalizes the trace of the
            global feature Gram matrix to prevent scale drift.
        c_l: Local calibration scalar.  Same role as ``c_g`` for local
            features.
        d: Local normalization denominators of shape ``(m_l,)``.
            Computed as ``d_j = sum_i s_j(x_i)^2 + eta`` to ensure
            unit-variance local features.
        t_r: Step index of the last refresh.  Used by the cooldown
            condition in the refresh controller.
        b_t: Hysteresis flag (``1`` = active, ``0`` = inactive).
            Prevents refresh churn by requiring the flag to be active
            before a refresh can trigger.
        rho: Fusion gate value in ``(0, 1)``.  Controls the balance
            between global and local features: ``phi = [sqrt(rho) *
            bar_phi_g, sqrt(1-rho) * bar_phi_l_perp]``.
    """

    Z: Array | None = None
    A: Array | None = None
    M_g: Array | None = None
    c_g: float = 1.0
    c_l: float = 1.0
    d: Array | None = None
    t_r: int = 0
    b_t: int = 1
    rho: float = 0.5

    def copy_with(self, **kwargs: Any) -> "DiscreteState":
        """Return a new DiscreteState with the specified fields overridden.

        Args:
            **kwargs: Fields to override.

        Returns:
            A new ``DiscreteState`` with updated values.
        """
        current: dict[str, Any] = {
            "Z": self.Z,
            "A": self.A,
            "M_g": self.M_g,
            "c_g": self.c_g,
            "c_l": self.c_l,
            "d": self.d,
            "t_r": self.t_r,
            "b_t": self.b_t,
            "rho": self.rho,
        }
        current.update(kwargs)
        return DiscreteState(**current)


@dataclass(frozen=True)
class FullState:
    """Complete training state passed through the loop.

    Composes continuous and discrete state into a single immutable
    snapshot that flows through the training loop.  Each training step
    produces a new ``FullState`` via ``copy_with``, ensuring that no
    step can accidentally corrupt a previous step's state.

    Attributes:
        continuous: Continuous representation state (``theta``, ``R``).
        discrete: Discrete basis state (``Z``, ``A``, ``M_g``, etc.).
        step: Current training step index (0-based).
        w: Current ridge coefficients of shape ``(m,)`` where
            ``m = r_g + m_l``.  ``None`` before the first solve.
    """

    continuous: ContinuousState = field(default_factory=ContinuousState)
    discrete: DiscreteState = field(default_factory=DiscreteState)
    step: int = 0
    w: Array | None = None

    def copy_with(self, **kwargs: Any) -> "FullState":
        """Return a new FullState with the specified fields overridden.

        Args:
            **kwargs: Fields to override.

        Returns:
            A new ``FullState`` with updated values.
        """
        current: dict[str, Any] = {
            "continuous": self.continuous,
            "discrete": self.discrete,
            "step": self.step,
            "w": self.w,
        }
        current.update(kwargs)
        return FullState(**current)
