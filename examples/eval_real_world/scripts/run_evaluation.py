"""Main evaluation script for aware-kernel real-world experiments.

Usage:
    cd examples/eval_real_world
    python -m scripts.run_evaluation \
        --datasets Diabetes CaliforniaHousing \
        --tiers Small Medium \
        --n_seeds 5 \
        --output_dir results

Or run the full sweep:
    python -m scripts.run_evaluation \
        --datasets all \
        --tiers all \
        --n_seeds 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

# Add repo root and eval_real_world to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loaders.loaders import DATASET_LOADERS

from aware_kernel import AwareKernelEstimator
from aware_kernel.aware.config import (
    AblationConfig,
    MemoryMode,
)
from aware_kernel.evaluation.baselines import (
    NystromRidgeBaseline,
    RandomFeatureBaseline,
    RidgeBaseline,
)
from aware_kernel.evaluation.metrics import compute_all_metrics
from aware_kernel.training.loop import TrainingLoop


# ---------------------------------------------------------------------------
# Budget tiers
# ---------------------------------------------------------------------------
@dataclass
class BudgetTier:
    """Preset budget tier for evaluation."""

    name: str
    m_g: int
    m_l: int
    max_steps: int
    embedding_dim: int
    total_refresh_budget: float
    refresh_cost: float
    memory_mode: MemoryMode = MemoryMode.CACHED


SMALL = BudgetTier("Small", 128, 32, 200, 16, 5.0, 1.0)
MEDIUM = BudgetTier("Medium", 512, 128, 1000, 64, 20.0, 1.0)
LARGE = BudgetTier("Large", 2048, 512, 2000, 128, 50.0, 1.0)
TIER_MAP = {"Small": SMALL, "Medium": MEDIUM, "Large": LARGE}

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def preprocess_and_split(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test with optional standard scaling."""
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    if standardize:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train)
        X_val = scaler_x.transform(X_val)
        X_test = scaler_x.transform(X_test)
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SingleRunResult:
    rmse: float
    mae: float
    r2: float
    train_time_sec: float
    predict_time_sec: float
    peak_mem_mb: float
    refresh_count: int
    condition_proxy: float


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    tier: str
    runs: list[SingleRunResult] = field(default_factory=list)

    def mean_std(self, attr: str) -> tuple[float, float]:
        vals = [getattr(r, attr) for r in self.runs]
        return float(np.mean(vals)), float(np.std(vals))

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "tier": self.tier,
            "rmse": f"{self.mean_std('rmse')[0]:.6f} ± {self.mean_std('rmse')[1]:.6f}",
            "mae": f"{self.mean_std('mae')[0]:.6f} ± {self.mean_std('mae')[1]:.6f}",
            "r2": f"{self.mean_std('r2')[0]:.6f} ± {self.mean_std('r2')[1]:.6f}",
            "train_time_sec": f"{self.mean_std('train_time_sec')[0]:.4f} ± {self.mean_std('train_time_sec')[1]:.4f}",
            "predict_time_sec": f"{self.mean_std('predict_time_sec')[0]:.6f} ± {self.mean_std('predict_time_sec')[1]:.6f}",
            "peak_mem_mb": f"{self.mean_std('peak_mem_mb')[0]:.2f} ± {self.mean_std('peak_mem_mb')[1]:.2f}",
            "refresh_count": f"{self.mean_std('refresh_count')[0]:.1f} ± {self.mean_std('refresh_count')[1]:.1f}",
            "condition_proxy": f"{self.mean_std('condition_proxy')[0]:.2e} ± {self.mean_std('condition_proxy')[1]:.2e}",
            "stability_rmse_var": self.mean_std("rmse")[1],
            "stability_time_var": self.mean_std("train_time_sec")[1],
            "stability_cond_var": self.mean_std("condition_proxy")[1],
        }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_aware_kernel(
    tier: BudgetTier,
    lambda_reg: float,
    seed: int,
    ablation: AblationConfig | None = None,
) -> AwareKernelEstimator:
    return AwareKernelEstimator(
        embedding_dim=tier.embedding_dim,
        m_g=tier.m_g,
        m_l=tier.m_l,
        lambda_reg=lambda_reg,
        memory_mode=tier.memory_mode.value,
        max_steps=tier.max_steps,
        eval_freq=max(1, tier.max_steps // 20),
        seed=seed,
        total_refresh_budget=tier.total_refresh_budget,
        refresh_cost=tier.refresh_cost,
        lr=1e-4,
        lambda_r=1e-4,
        lambda_orth=1e-4,
        gamma_div=1e-3,
        fd_epsilon=1e-5,
        disable_refresh=ablation.disable_refresh if ablation else False,
        disable_hysteresis=ablation.disable_hysteresis if ablation else False,
        disable_cooldown=ablation.disable_cooldown if ablation else False,
        disable_residual_aware_anchors=ablation.disable_residual_aware_anchors
        if ablation
        else False,
        disable_orthogonalization=ablation.disable_orthogonalization
        if ablation
        else False,
        disable_diversity_penalty=ablation.disable_diversity_penalty
        if ablation
        else False,
        static_scaling=ablation.static_scaling if ablation else False,
    )


def make_nystrom(
    tier: BudgetTier, lambda_reg: float, seed: int
) -> NystromRidgeBaseline:
    return NystromRidgeBaseline(m_g=tier.m_g, lambda_reg=lambda_reg, seed=seed)


def make_rff(tier: BudgetTier, lambda_reg: float, seed: int) -> RandomFeatureBaseline:
    n_features = min(2000, max(tier.m_g, 512))
    return RandomFeatureBaseline(
        n_features=n_features, gamma=1.0, lambda_reg=lambda_reg, seed=seed
    )


def make_ridge(lambda_reg: float) -> RidgeBaseline:
    return RidgeBaseline(lambda_reg=lambda_reg)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_lambda_reg(
    factory: Callable[[float], object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: list[float] | None = None,
) -> float:
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best_rmse = float("inf")
    best_lam = lambdas[0]
    for lam in lambdas:
        model = factory(lam)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - y_pred) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_lam = lam
    return best_lam


# ---------------------------------------------------------------------------
# Condition proxy
# ---------------------------------------------------------------------------
def compute_condition_proxy(
    estimator: AwareKernelEstimator, X: np.ndarray, lambda_reg: float
) -> float:
    if estimator.state_ is None or estimator.state_.w is None:
        return float("inf")
    loop = TrainingLoop(estimator.config_)
    embedder = (
        estimator.state_.continuous.theta.get("embedder")
        if estimator.state_.continuous.theta
        else None
    )
    if embedder is None:
        return float("inf")
    embeddings = embedder.embed(X)
    from aware_kernel.embedding.projector import Projector

    projector = Projector(estimator.state_.continuous.R)
    U = projector.transform(embeddings)
    phi = loop._build_fused_features(U, estimator.state_.discrete)
    gram = phi.T @ phi + lambda_reg * np.eye(phi.shape[1])
    try:
        eigs = np.linalg.eigvalsh(gram)
        return float(np.max(eigs) / (np.min(eigs) + 1e-15))
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------
def run_single_aware_kernel(
    model: AwareKernelEstimator, X_train, y_train, X_test, y_test
) -> SingleRunResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = compute_all_metrics(y_test, y_pred)
    cond = compute_condition_proxy(model, X_test, model.lambda_reg)
    refresh_count = 1 if model.state_ and model.state_.discrete.t_r > 0 else 0

    return SingleRunResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        peak_mem_mb=peak / (1024 * 1024),
        refresh_count=refresh_count,
        condition_proxy=cond,
    )


def run_single_baseline(model, X_train, y_train, X_test, y_test) -> SingleRunResult:
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    metrics = compute_all_metrics(y_test, y_pred)
    return SingleRunResult(
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        peak_mem_mb=peak / (1024 * 1024),
        refresh_count=0,
        condition_proxy=float("nan"),
    )


# ---------------------------------------------------------------------------
# Experiment suite
# ---------------------------------------------------------------------------
def run_experiment_suite(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    tier: BudgetTier,
    seeds: list[int],
    run_ablations: bool = True,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []

    # AwareKernel full
    exp = ExperimentResult(dataset=dataset_name, model="AwareKernel", tier=tier.name)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
        best_lam = tune_lambda_reg(
            lambda lam, _seed=seed: make_aware_kernel(tier, lam, _seed),
            X_tr,
            y_tr,
            X_val,
            y_val,
            lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
        )
        model = make_aware_kernel(tier, best_lam, seed)
        exp.runs.append(run_single_aware_kernel(model, X_tr, y_tr, X_te, y_te))
    results.append(exp)

    # Baselines
    baseline_factories = {
        "Ridge": lambda lam, s: make_ridge(lam),
        "Nystrom": lambda lam, s: make_nystrom(tier, lam, s),
        "RFF": lambda lam, s: make_rff(tier, lam, s),
    }
    for bname, factory in baseline_factories.items():
        exp = ExperimentResult(dataset=dataset_name, model=bname, tier=tier.name)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
            best_lam = tune_lambda_reg(
                lambda lam, _seed=seed, _factory=factory: _factory(lam, _seed),
                X_tr,
                y_tr,
                X_val,
                y_val,
                lambdas=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            )
            model = factory(best_lam, seed)
            exp.runs.append(run_single_baseline(model, X_tr, y_tr, X_te, y_te))
        results.append(exp)

    # Ablations
    if run_ablations:
        ablation_configs = {
            "AK-NoRefresh": AblationConfig(disable_refresh=True),
            "AK-NoHysteresis": AblationConfig(disable_hysteresis=True),
            "AK-NoCooldown": AblationConfig(disable_cooldown=True),
            "AK-NoResidAnchors": AblationConfig(disable_residual_aware_anchors=True),
            "AK-NoOrthog": AblationConfig(disable_orthogonalization=True),
            "AK-NoDivPenalty": AblationConfig(disable_diversity_penalty=True),
            "AK-StaticScaling": AblationConfig(static_scaling=True),
        }
        for abname, abcfg in ablation_configs.items():
            exp = ExperimentResult(dataset=dataset_name, model=abname, tier=tier.name)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                X_tr, X_val, X_te, y_tr, y_val, y_te = preprocess_and_split(X, y, rng)
                best_lam = tune_lambda_reg(
                    lambda lam, _seed=seed, _abcfg=abcfg: make_aware_kernel(
                        tier, lam, _seed, ablation=_abcfg
                    ),
                    X_tr,
                    y_tr,
                    X_val,
                    y_val,
                    lambdas=[1e-4, 1e-3, 1e-2, 1e-1],
                )
                model = make_aware_kernel(tier, best_lam, seed, ablation=abcfg)
                exp.runs.append(run_single_aware_kernel(model, X_tr, y_tr, X_te, y_te))
            results.append(exp)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def results_to_markdown(results: list[ExperimentResult]) -> str:
    lines = [
        "| Dataset | Model | Tier | RMSE | MAE | R^2 | Train(s) | PeakMem(MB) | Refresh | Cond(kappa) |",
        "|---------|-------|------|------|-----|-----|----------|-------------|---------|-------------|",
    ]
    for exp in results:
        d = exp.to_dict()
        lines.append(
            f"| {d['dataset']} | {d['model']} | {d['tier']} | {d['rmse']} | {d['mae']} | {d['r2']} | {d['train_time_sec']} | {d['peak_mem_mb']} | {d['refresh_count']} | {d['condition_proxy']} |"
        )
    return "\n".join(lines)


def stability_to_markdown(results: list[ExperimentResult]) -> str:
    lines = [
        "| Dataset | Model | Tier | Var(RMSE) | Var(Time) | Var(Cond) |",
        "|---------|-------|------|-----------|-----------|-----------|",
    ]
    for exp in results:
        d = exp.to_dict()
        lines.append(
            f"| {d['dataset']} | {d['model']} | {d['tier']} | {d['stability_rmse_var']:.6e} | {d['stability_time_var']:.6e} | {d['stability_cond_var']:.6e} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run aware-kernel evaluation suite.")
    parser.add_argument(
        "--datasets", nargs="+", default=["Diabetes"], help="Dataset names or 'all'."
    )
    parser.add_argument(
        "--tiers", nargs="+", default=["Small"], help="Budget tier names or 'all'."
    )
    parser.add_argument(
        "--n_seeds", type=int, default=5, help="Number of random seeds."
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory."
    )
    parser.add_argument(
        "--no_ablations", action="store_true", help="Skip ablation runs."
    )
    args = parser.parse_args()

    datasets = (
        list(DATASET_LOADERS.keys()) if args.datasets == ["all"] else args.datasets
    )
    tiers = [
        TIER_MAP[t]
        for t in (["Small", "Medium", "Large"] if args.tiers == ["all"] else args.tiers)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ExperimentResult] = []

    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"WARNING: Unknown dataset '{dataset_name}', skipping.")
            continue
        print(f"\n=== Dataset: {dataset_name} ===")
        rng = np.random.default_rng(0)
        X, y = DATASET_LOADERS[dataset_name](rng)
        print(f"  Loaded n={X.shape[0]}, d={X.shape[1]}")

        for tier in tiers:
            print(f"\n  -- Tier: {tier.name} --")
            seeds = [42 + i for i in range(args.n_seeds)]
            tier_results = run_experiment_suite(
                dataset_name, X, y, tier, seeds, run_ablations=not args.no_ablations
            )
            all_results.extend(tier_results)
            for exp in tier_results:
                rmse_m, rmse_s = exp.mean_std("rmse")
                print(
                    f"    {exp.model:22s} RMSE={rmse_m:.4f}±{rmse_s:.4f}  time={exp.mean_std('train_time_sec')[0]:.2f}s"
                )

    # Write outputs
    md_path = output_dir / "results.md"
    with open(md_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write("## Mean ± Std Metrics\n\n")
        f.write(results_to_markdown(all_results))
        f.write("\n\n## Stability-over-Refits\n\n")
        f.write(stability_to_markdown(all_results))
    print(f"\nWrote markdown: {md_path}")

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].to_dict().keys())
        writer.writeheader()
        for exp in all_results:
            writer.writerow(exp.to_dict())
    print(f"Wrote CSV: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
