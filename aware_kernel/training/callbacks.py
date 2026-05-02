"""Training callbacks for logging and checkpointing."""

from typing import Any, Optional

import numpy as np

from aware_kernel.aware.state import FullState


class Callback:
    """Base callback interface."""

    def on_step_end(self, step: int, state: FullState, metrics: dict) -> None:
        """Called at the end of each training step."""
        pass

    def on_refresh(self, step: int, state: FullState) -> None:
        """Called when a refresh triggers."""
        pass

    def on_eval(self, step: int, metrics: dict) -> None:
        """Called during evaluation."""
        pass


class LoggingCallback(Callback):
    """Simple callback that prints metrics."""

    def __init__(self, log_interval: int = 10) -> None:
        """Initialize logging callback.

        Args:
            log_interval: Print metrics every N steps.
        """
        self._log_interval = log_interval

    def on_step_end(self, step: int, state: FullState, metrics: dict) -> None:
        """Print step metrics at intervals."""
        if step % self._log_interval == 0:
            print(f"Step {step}: {metrics}")

    def on_refresh(self, step: int, state: FullState) -> None:
        """Print refresh notification."""
        print(f"Step {step}: Refresh triggered")

    def on_eval(self, step: int, metrics: dict) -> None:
        """Print evaluation metrics."""
        print(f"Eval at step {step}: {metrics}")


class CheckpointCallback(Callback):
    """Callback that saves state checkpoints."""

    def __init__(self, save_interval: int = 100, path_prefix: str = "checkpoint") -> None:
        """Initialize checkpoint callback.

        Args:
            save_interval: Save every N steps.
            path_prefix: Prefix for checkpoint file paths.
        """
        self._save_interval = save_interval
        self._path_prefix = path_prefix

    def on_step_end(self, step: int, state: FullState, metrics: dict) -> None:
        """Save checkpoint at intervals."""
        if step % self._save_interval == 0:
            # Placeholder: in production, serialize state to disk
            pass

    def on_refresh(self, step: int, state: FullState) -> None:
        """Optionally save checkpoint on refresh."""
        pass
