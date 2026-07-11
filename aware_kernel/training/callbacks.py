"""Training callbacks for logging and checkpointing.

Provides a base ``Callback`` class and concrete implementations for
logging and checkpointing.  Callbacks are invoked by the ``TrainingLoop``
at specific points during training:

* ``on_step_end``: Called after each training step with the current
  state and metrics.
* ``on_refresh``: Called when a discrete refresh triggers.
* ``on_eval``: Called during evaluation with the computed metrics.

Callbacks enable monitoring, early stopping, and model persistence
without modifying the training loop itself.

Design rationale
----------------
The callback pattern is chosen over direct logging in the training loop
because it:

1. Separates concerns (training logic vs. monitoring).
2. Allows multiple callbacks to be composed.
3. Makes it easy to add custom monitoring without subclassing the loop.
"""

import logging

from aware_kernel.aware.state import FullState

logger = logging.getLogger(__name__)


class Callback:
    """Base callback interface.

    Subclass this and override any of the hook methods to receive
    notifications during training.  All methods are no-ops by default,
    so subclasses only need to implement the hooks they care about.
    """

    def on_step_end(
        self, step: int, state: FullState, metrics: dict[str, float]
    ) -> None:
        """Called at the end of each training step.

        Args:
            step: Current training step (1-indexed).
            state: Current training state.
            metrics: Dictionary of metrics computed at this step.
        """
        pass

    def on_refresh(self, step: int, state: FullState) -> None:
        """Called when a discrete refresh triggers.

        Args:
            step: Current training step.
            state: State after the refresh.
        """
        pass

    def on_eval(self, step: int, metrics: dict[str, float]) -> None:
        """Called during evaluation.

        Args:
            step: Current training step.
            metrics: Dictionary of evaluation metrics.
        """
        pass


class LoggingCallback(Callback):
    """Callback that prints metrics to stdout at regular intervals.

    Useful for monitoring training progress in interactive sessions
    or log files.

    Attributes:
        log_interval: Log metrics every N steps.
    """

    def __init__(self, log_interval: int = 10) -> None:
        """Initialize logging callback.

        Args:
            log_interval: Log metrics every N steps.  Default ``10``.
        """
        self.log_interval = log_interval

    def on_step_end(
        self, step: int, state: FullState, metrics: dict[str, float]
    ) -> None:
        """Log step metrics at intervals."""
        if step % self.log_interval == 0:
            logger.info("Step %d: %s", step, metrics)

    def on_refresh(self, step: int, state: FullState) -> None:
        """Log refresh notification."""
        logger.info("Step %d: Refresh triggered", step)

    def on_eval(self, step: int, metrics: dict[str, float]) -> None:
        """Log evaluation metrics."""
        logger.info("Eval at step %d: %s", step, metrics)


class CheckpointCallback(Callback):
    """Callback that saves state checkpoints at regular intervals.

    Currently a placeholder for production serialization.  In a real
    deployment, this would serialize ``FullState`` to disk using
    ``pickle`` or ``numpy.savez``.

    Attributes:
        save_interval: Save every N steps.
        path_prefix: Prefix for checkpoint file paths.
    """

    def __init__(
        self, save_interval: int = 100, path_prefix: str = "checkpoint"
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            save_interval: Save every N steps.  Default ``100``.
            path_prefix: Prefix for checkpoint file paths.
        """
        self.save_interval = save_interval
        self.path_prefix = path_prefix

    def on_step_end(
        self, step: int, state: FullState, metrics: dict[str, float]
    ) -> None:
        """Save checkpoint at intervals.

        Placeholder: in production, serialize state to disk.
        """
        if step % self.save_interval == 0:
            # Placeholder: in production, serialize state to disk
            pass

    def on_refresh(self, step: int, state: FullState) -> None:
        """Optionally save checkpoint on refresh."""
        pass
