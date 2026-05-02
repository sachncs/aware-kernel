"""State containers for continuous and discrete parameters.

State objects are preferred to be immutable or treated as snapshots
to simplify reasoning about refresh boundaries.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from aware_kernel.aware.types import Array


@dataclass(frozen=True)
class ContinuousState:
    """Continuously updated representation parameters.

    Attributes:
        theta: Embedding parameters (opaque; kept as a dict or array here).
        R: Projection matrix of shape (embedding_dim, embedding_dim).
    """

    theta: Optional[dict] = None
    R: Optional[Array] = None

    def copy_with(self, **kwargs) -> "ContinuousState":
        """Return a new ContinuousState with updated fields."""
        current = {
            "theta": self.theta,
            "R": self.R,
        }
        current.update(kwargs)
        return ContinuousState(**current)


@dataclass(frozen=True)
class DiscreteState:
    """Discrete basis state refreshed on trigger.

    Attributes:
        Z: Global landmarks of shape (m_g, embedding_dim).
        A: Local anchors of shape (m_l, embedding_dim).
        M_g: Whitening map for global basis of shape (m_g, r_g).
        c_g: Global calibration scalar.
        c_l: Local calibration scalar.
        d: Local normalization denominators of shape (m_l,).
        t_r: Step index of last refresh.
        b_t: Hysteresis flag (1 = active, 0 = inactive).
        rho: Fusion gate value.
    """

    Z: Optional[Array] = None
    A: Optional[Array] = None
    M_g: Optional[Array] = None
    c_g: float = 1.0
    c_l: float = 1.0
    d: Optional[Array] = None
    t_r: int = 0
    b_t: int = 1
    rho: float = 0.5

    def copy_with(self, **kwargs) -> "DiscreteState":
        """Return a new DiscreteState with updated fields."""
        current = {
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

    Attributes:
        continuous: Continuous representation state.
        discrete: Discrete basis state.
        step: Current training step.
        w: Current ridge coefficients of shape (m,) where m = r_g + m_l.
    """

    continuous: ContinuousState = field(default_factory=ContinuousState)
    discrete: DiscreteState = field(default_factory=DiscreteState)
    step: int = 0
    w: Optional[Array] = None

    def copy_with(self, **kwargs) -> "FullState":
        """Return a new FullState with updated fields."""
        current = {
            "continuous": self.continuous,
            "discrete": self.discrete,
            "step": self.step,
            "w": self.w,
        }
        current.update(kwargs)
        return FullState(**current)
