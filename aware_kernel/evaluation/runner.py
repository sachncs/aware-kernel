"""Experiment runner for comparing models on synthetic datasets.

Provides a lightweight framework to evaluate aware-kernel against
baselines with controlled random seeds and reproducible splits.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from aware_kernel.aware.types import Array
from aware_kernel.evaluation.datasets import split_train_test
from aware_kernel.evaluation.metrics import compute_all_metrics


@dataclass
class ExperimentResult:
    """Result container for a single model on a single dataset.

    Attributes:
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        metrics: Dictionary of evaluation metrics.
        fit_time_seconds: Wall-clock time for fitting.
        predict_time_seconds: Wall-clock time for prediction.
    """

    model_name: str
    dataset_name: str
    metrics: dict = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0


class ExperimentRunner:
    """Run a suite of models on a suite of datasets."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize runner.

        Args:
            seed: Global random seed for reproducibility.
        """
        self._seed = seed
        self._rng: Optional[np.random.Generator] = None

    def _get_rng(self) -> np.random.Generator:
        """Return a deterministic RNG, creating one if necessary."""
        if self._rng is None:
            self._rng = np.random.default_rng(self._seed)
        return self._rng

    def run_experiment(
        self,
        model_name: str,
        model: object,
        dataset_name: str,
        X: Array,
        y: Array,
        test_size: float = 0.2,
    ) -> ExperimentResult:
        """Run a single model on a single dataset.

        Args:
            model_name: Human-readable model name.
            model: Model object with ``fit(X, y)`` and ``predict(X)`` methods.
            dataset_name: Human-readable dataset name.
            X: Full input matrix.
            y: Full target vector.
            test_size: Fraction of data to hold out for testing.

        Returns:
            ExperimentResult with metrics and timings.
        """
        rng = self._get_rng()
        X_train, X_test, y_train, y_test = split_train_test(
            X, y, test_size=test_size, rng=rng
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        predict_time = time.perf_counter() - t0

        metrics = compute_all_metrics(y_test, y_pred)

        return ExperimentResult(
            model_name=model_name,
            dataset_name=dataset_name,
            metrics=metrics,
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
        )

    def run_suite(
        self,
        models: Dict[str, Callable[[], object]],
        datasets: Dict[str, Callable[[np.random.Generator], tuple]],
        test_size: float = 0.2,
    ) -> List[ExperimentResult]:
        """Run all model factories on all dataset factories.

        Args:
            models: Dictionary mapping model name to a factory callable
                that returns a model instance.
            datasets: Dictionary mapping dataset name to a factory callable
                that takes an RNG and returns (X, y).
            test_size: Fraction of data to hold out for testing.

        Returns:
            List of ExperimentResult objects.
        """
        results: List[ExperimentResult] = []
        rng = self._get_rng()

        for dataset_name, dataset_factory in datasets.items():
            dataset_result = dataset_factory(rng)
            if len(dataset_result) == 3:
                X, y, *_ = dataset_result
            else:
                X, y = dataset_result

            for model_name, model_factory in models.items():
                model = model_factory()
                result = self.run_experiment(
                    model_name=model_name,
                    model=model,
                    dataset_name=dataset_name,
                    X=X,
                    y=y,
                    test_size=test_size,
                )
                results.append(result)

        return results


def format_results_table(results: List[ExperimentResult]) -> str:
    """Format experiment results as a markdown table.

    Args:
        results: List of ExperimentResult objects.

    Returns:
        Markdown-formatted table string.
    """
    lines = [
        "| Model | Dataset | RMSE | MAE | R^2 | Fit (s) | Predict (s) |",
        "|-------|---------|------|-----|-----|---------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r.model_name} | {r.dataset_name} | "
            f"{r.metrics.get('rmse', float('nan')):.6f} | "
            f"{r.metrics.get('mae', float('nan')):.6f} | "
            f"{r.metrics.get('r2', float('nan')):.6f} | "
            f"{r.fit_time_seconds:.4f} | {r.predict_time_seconds:.4f} |"
        )
    return "\n".join(lines)
