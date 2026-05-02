"""Unit tests for aware_kernel.evaluation modules."""

import numpy as np
import pytest

from aware_kernel.evaluation.baselines import (
    NystromRidgeBaseline,
    RandomFeatureBaseline,
    RidgeBaseline,
)
from aware_kernel.evaluation.datasets import (
    make_heteroscedastic_regression,
    make_high_dim_regression,
    make_linear_regression,
    make_polynomial_regression,
    split_train_test,
)
from aware_kernel.evaluation.metrics import (
    compute_all_metrics,
    compute_mae,
    compute_max_abs_error,
    compute_r2,
    compute_rmse,
)
from aware_kernel.evaluation.runner import ExperimentResult, ExperimentRunner, format_results_table


class TestMakeLinearRegression:
    """Tests for make_linear_regression."""

    def test_shapes(self, rng: np.random.Generator) -> None:
        """Output shapes should match requested dimensions."""
        X, y, true_w = make_linear_regression(rng, n_samples=100, n_features=5)
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert true_w.shape == (5,)

    def test_nonzero_weights(self, rng: np.random.Generator) -> None:
        """True weights should generally be nonzero."""
        _, _, true_w = make_linear_regression(rng, n_samples=50, n_features=5)
        assert np.count_nonzero(true_w) > 0


class TestMakePolynomialRegression:
    """Tests for make_polynomial_regression."""

    def test_shapes(self, rng: np.random.Generator) -> None:
        """Output shapes should match requested dimensions."""
        X, y = make_polynomial_regression(rng, n_samples=100, degree=3)
        assert X.shape == (100, 1)
        assert y.shape == (100,)

    def test_non_constant(self, rng: np.random.Generator) -> None:
        """Y should not be constant."""
        X, y = make_polynomial_regression(rng, n_samples=100, degree=3)
        assert np.std(y) > 0.0


class TestMakeHighDimRegression:
    """Tests for make_high_dim_regression."""

    def test_shapes(self, rng: np.random.Generator) -> None:
        """Output shapes should match requested dimensions."""
        X, y, true_w = make_high_dim_regression(
            rng, n_samples=100, n_features=50, n_informative=5
        )
        assert X.shape == (100, 50)
        assert y.shape == (100,)
        assert true_w.shape == (50,)

    def test_sparsity(self, rng: np.random.Generator) -> None:
        """True weights should have exactly n_informative nonzero entries."""
        _, _, true_w = make_high_dim_regression(
            rng, n_samples=50, n_features=50, n_informative=5
        )
        assert np.count_nonzero(true_w) == 5


class TestMakeHeteroscedasticRegression:
    """Tests for make_heteroscedastic_regression."""

    def test_shapes(self, rng: np.random.Generator) -> None:
        """Output shapes should match requested dimensions."""
        X, y = make_heteroscedastic_regression(rng, n_samples=100)
        assert X.shape == (100, 1)
        assert y.shape == (100,)


class TestSplitTrainTest:
    """Tests for split_train_test."""

    def test_split_sizes(self, rng: np.random.Generator) -> None:
        """Train and test sizes should sum to total."""
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, rng=rng)
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20

    def test_disjoint(self, rng: np.random.Generator) -> None:
        """Train and test sets should be disjoint."""
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.3, rng=rng)
        train_set = set(map(tuple, X_train))
        test_set = set(map(tuple, X_test))
        assert train_set.isdisjoint(test_set)


class TestComputeRmse:
    """Tests for compute_rmse."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction should yield zero RMSE."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_positive(self) -> None:
        """Imperfect prediction should yield positive RMSE."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        assert compute_rmse(y_true, y_pred) == pytest.approx(1.0)


class TestComputeMae:
    """Tests for compute_mae."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction should yield zero MAE."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0, abs=1e-10)

    def test_basic(self) -> None:
        """Basic MAE calculation."""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, -1.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(1.0)


class TestComputeR2:
    """Tests for compute_r2."""

    def test_perfect_prediction(self) -> None:
        """Perfect prediction should yield R^2 = 1."""
        y = np.array([1.0, 2.0, 3.0])
        assert compute_r2(y, y) == pytest.approx(1.0)

    def test_constant_true(self) -> None:
        """Constant ground truth should yield R^2 = 1 by convention."""
        y = np.array([2.0, 2.0, 2.0])
        y_pred = np.array([1.0, 3.0, 2.0])
        assert compute_r2(y, y_pred) == pytest.approx(1.0)


class TestComputeMaxAbsError:
    """Tests for compute_max_abs_error."""

    def test_basic(self) -> None:
        """Max absolute error should return largest deviation."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.0, 3.0])
        assert compute_max_abs_error(y_true, y_pred) == pytest.approx(1.0)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_keys(self) -> None:
        """Result should contain expected keys."""
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_all_metrics(y, y)
        assert set(metrics.keys()) == {"rmse", "mae", "r2", "max_abs_error"}


class TestRidgeBaseline:
    """Tests for RidgeBaseline."""

    def test_fit_predict(self, rng: np.random.Generator) -> None:
        """Should fit and predict on linear data."""
        X, y, _ = make_linear_regression(rng, n_samples=100, n_features=5)
        model = RidgeBaseline(lambda_reg=1e-2)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit should raise RuntimeError."""
        model = RidgeBaseline()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))


class TestNystromRidgeBaseline:
    """Tests for NystromRidgeBaseline."""

    def test_fit_predict(self, rng: np.random.Generator) -> None:
        """Should fit and predict on linear data."""
        X, y, _ = make_linear_regression(rng, n_samples=100, n_features=5)
        model = NystromRidgeBaseline(m_g=16, lambda_reg=1e-2, seed=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit should raise RuntimeError."""
        model = NystromRidgeBaseline()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))


class TestRandomFeatureBaseline:
    """Tests for RandomFeatureBaseline."""

    def test_fit_predict(self, rng: np.random.Generator) -> None:
        """Should fit and predict on linear data."""
        X, y, _ = make_linear_regression(rng, n_samples=100, n_features=5)
        model = RandomFeatureBaseline(n_features=50, lambda_reg=1e-2, seed=42)
        model.fit(X, y)
        y_pred = model.predict(X)
        assert y_pred.shape == (100,)

    def test_predict_before_fit_raises(self) -> None:
        """Predicting before fit should raise RuntimeError."""
        model = RandomFeatureBaseline()
        with pytest.raises(RuntimeError):
            model.predict(np.zeros((5, 3)))


class TestExperimentRunner:
    """Tests for ExperimentRunner."""

    def test_run_experiment(self, rng: np.random.Generator) -> None:
        """Should produce a valid ExperimentResult."""
        X, y, _ = make_linear_regression(rng, n_samples=100, n_features=5)
        runner = ExperimentRunner(seed=42)
        model = RidgeBaseline(lambda_reg=1e-2)
        result = runner.run_experiment("ridge", model, "linear", X, y, test_size=0.2)
        assert result.model_name == "ridge"
        assert result.dataset_name == "linear"
        assert "rmse" in result.metrics
        assert result.fit_time_seconds >= 0.0
        assert result.predict_time_seconds >= 0.0

    def test_run_suite(self, rng: np.random.Generator) -> None:
        """Should produce one result per model-dataset pair."""
        models = {
            "ridge": lambda: RidgeBaseline(lambda_reg=1e-2),
        }
        datasets = {
            "linear": lambda r: make_linear_regression(r, n_samples=100, n_features=5),
        }
        runner = ExperimentRunner(seed=42)
        results = runner.run_suite(models, datasets)
        assert len(results) == 1


class TestFormatResultsTable:
    """Tests for format_results_table."""

    def test_contains_headers(self) -> None:
        """Output should contain markdown table headers."""
        result = ExperimentResult(
            model_name="m", dataset_name="d", metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9, "max_abs_error": 2.0}
        )
        table = format_results_table([result])
        assert "| Model |" in table
        assert "| Dataset |" in table

    def test_contains_data(self) -> None:
        """Output should contain the result data."""
        result = ExperimentResult(
            model_name="m", dataset_name="d", metrics={"rmse": 1.0, "mae": 0.5, "r2": 0.9, "max_abs_error": 2.0}
        )
        table = format_results_table([result])
        assert "m | d" in table
