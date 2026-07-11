<p align="center">
  <h1 align="center">AwareKernel</h1>
  <p align="center">Refresh-aware hybrid continuous-discrete low-rank kernel learning for scalable, adaptive kernel regression.</p>
  <p align="center">
    <a href="#installation"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/sachncs/aware-kernel/actions"><img src="https://img.shields.io/github/actions/workflow/status/sachncs/aware-kernel/ci.yml?branch=master" alt="CI"></a>
    <a href="https://github.com/sachncs/aware-kernel/stargazers"><img src="https://img.shields.io/github/stars/sachncs/aware-kernel" alt="Stars"></a>
    <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-strict-green.svg" alt="Checked with mypy"></a>
  </p>
</p>

**aware-kernel** is a Python library that implements refresh-aware hybrid
continuous-discrete low-rank kernel regression. It separates model parameters
into continuous (updated every step) and discrete (refreshed adaptively)
groups, enabling efficient training with dynamic basis adaptation.

---

## Features

- **Explicit-feature kernel system** — PSD guarantee (`K = Phi Phi^T >= 0`) with rank bound `rank(K) <= r_g + m_l`
- **Global Nyström basis** — Landmark selection via k-means++ with soft-truncated spectral whitening
- **Local corrective features** — Residual-aware anchor sampling and k-NN sparse RBF features
- **Residual orthogonalization** — Ensures local features live in the global nullspace
- **Feature calibration and fusion** — Trace-based normalization and logistic gate balancing global/local contributions
- **Refresh controller** — Drift-aware trigger with cooldown, warmup, hysteresis, and amortized budget
- **Two memory modes** — Cached O(nm) and streamed O(m^2) for different scale regimes
- **Numerical stabilization** — Eigenvalue clipping, soft spectral truncation, and Cholesky with jitter fallback
- **Sklearn-compatible API** — `fit`, `predict`, `score` with `GridSearchCV` and `Pipeline` support

---

## Installation

### From source

```bash
git clone https://github.com/sachncs/aware-kernel.git
cd aware-kernel
pip install -e .
```

### With dev dependencies

```bash
pip install -e ".[dev]"
```

**Requirements**: Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, scikit-learn >= 1.3

---

## Quick Start

### Python API

```python
import numpy as np
from aware_kernel import AwareKernelEstimator

# Generate synthetic data
rng = np.random.default_rng(42)
X_train = rng.standard_normal((200, 4))
y_train = X_train[:, 0] + 0.5 * X_train[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

X_test = rng.standard_normal((50, 4))
y_test = X_test[:, 0] + 0.5 * X_test[:, 1] ** 2 + 0.1 * rng.standard_normal(50)

# Fit the model
model = AwareKernelEstimator(
    embedding_dim=4,
    m_g=32,
    m_l=8,
    lambda_reg=1e-2,
    max_steps=50,
    seed=42,
)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"R^2 score: {model.score(X_test, y_test):.4f}")
```

### Memory Modes

```python
# Cached (default) - O(nm) memory, simpler
model = AwareKernelEstimator(memory_mode="cached")

# Streamed - O(m^2) memory, scales to larger datasets
model = AwareKernelEstimator(memory_mode="streamed")
```

### Ablation Studies

```python
# Disable specific components for ablation
model = AwareKernelEstimator(
    disable_refresh=True,            # No discrete refreshes
    disable_orthogonalization=True,  # Skip orthogonalization
    disable_diversity_penalty=True,  # Remove diversity regularization
)
```

---

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Dimension of continuous embedding space |
| `m_g` | 512 | Global basis rank budget (landmarks) |
| `m_l` | 128 | Local corrective rank budget (anchors) |
| `lambda_reg` | 1e-3 | Ridge regularization parameter |
| `memory_mode` | `"cached"` | `"cached"` or `"streamed"` |
| `max_steps` | 1000 | Maximum training steps |
| `seed` | `None` | Random seed for reproducibility |

### Refresh Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_hi` | 0.1 | Drift threshold to trigger refresh |
| `t_cool` | 50 | Minimum steps between refreshes |
| `t_warmup` | 10 | Minimum step before first refresh |
| `gamma_cost` | 0.01 | Validation gain threshold scaled by refresh cost |

### Outer-Loop Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate for gradient descent on R |
| `lambda_r` | 0.0 | Frobenius regularizer weight |
| `lambda_orth` | 0.0 | Orthogonality penalty weight |
| `gamma_div` | 0.0 | Diversity penalty weight |
| `fd_epsilon` | 1e-5 | Finite-difference perturbation magnitude |

See [docs/getting-started.md](docs/getting-started.md) for detailed configuration options.

---

## Project Structure

```
aware-kernel/
├── aware_kernel/              # SDK package
│   ├── __init__.py            # Public API exports
│   ├── api.py                 # Sklearn-compatible estimator
│   ├── _version.py            # PEP 440 version
│   ├── aware/                 # Core types, config, state, exceptions
│   │   ├── config.py          # TrainingConfig, NumericsConfig, etc.
│   │   ├── state.py           # ContinuousState, DiscreteState, FullState
│   │   ├── types.py           # Array, protocols (Embedder, Solver, etc.)
│   │   └── exceptions.py      # ConditioningError, BudgetExceededError
│   ├── embedding/             # Dense embedder and projection matrix
│   │   ├── embedder.py        # DenseEmbedder (f_theta(x))
│   │   └── projector.py       # Projector (normalize + R-projection)
│   ├── global_basis/          # Nyström landmark selection and whitening
│   │   ├── nystrom.py         # NystromGlobalBasis
│   │   ├── whitening.py       # Spectral whitening map
│   │   └── builder.py         # GlobalFeatureBuilder
│   ├── local_corrective/      # Anchor sampling, sparse features
│   │   ├── anchors.py         # Residual-aware anchor selection
│   │   ├── sparse_features.py # k-NN sparse RBF features
│   │   └── orthogonalizer.py  # Ridge-regularized nullspace projection
│   ├── fusion/                # Calibration, gating, fused feature building
│   │   ├── calibration.py     # Trace-based feature normalization
│   │   ├── gate.py            # Logistic sigmoid gate
│   │   └── builder.py         # FusedFeatureBuilder
│   ├── solver/                # Ridge regression (Cholesky / PCG)
│   │   ├── ridge.py           # DirectRidgeSolver, IterativeRidgeSolver
│   │   ├── normal_eq.py       # Normal equation assembly
│   │   └── preconditioner.py  # Diagonal Jacobi preconditioner
│   ├── memory/                # Cached and streamed accumulators
│   │   ├── base.py            # BaseMemoryAccumulator
│   │   ├── cached.py          # CachedMemoryAccumulator
│   │   └── streamed.py        # StreamedMemoryAccumulator
│   ├── refresh/               # Drift, budget, controller, refresh pipeline
│   │   ├── controller.py      # Five-condition decision logic
│   │   ├── drift.py           # Relative Frobenius-norm drift
│   │   ├── pipeline.py        # Seven-step discrete refresh pipeline
│   │   └── budget.py          # BudgetAccountant
│   ├── training/              # Training loop, objectives, callbacks
│   │   ├── loop.py            # TrainingLoop
│   │   ├── objectives.py      # Bilevel outer-loop objectives
│   │   ├── optimizer.py       # OuterObjectiveOptimizer (SPSA)
│   │   └── callbacks.py       # LoggingCallback, CheckpointCallback
│   ├── inference/             # Mean and variance prediction
│   │   └── predictor.py       # Predictor (Bayesian posterior)
│   ├── evaluation/            # Datasets, baselines, metrics
│   │   ├── baselines.py       # Ridge, Nyström, RFF baselines
│   │   ├── datasets.py        # Synthetic regression generators
│   │   ├── metrics.py         # RMSE, MAE, R^2, max abs error
│   │   └── runner.py          # ExperimentRunner
│   └── utils/                 # Linear algebra, numerics, sampling
│       ├── linalg.py          # Safe Cholesky, PCG, Frobenius norm
│       ├── numerics.py        # Eigenvalue clipping, soft truncation
│       └── sampling.py        # k-means++, farthest point sampling
├── tests/                     # Test suite (unit, numerical, integration)
├── examples/                  # Real-world evaluation examples
├── docs/                      # Documentation
└── pyproject.toml             # Build & tool config
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Lint
ruff check aware_kernel/ tests/

# Format
ruff format aware_kernel/ tests/

# Type check
mypy aware_kernel/

# All checks
pytest && ruff check aware_kernel/ tests/ && mypy aware_kernel/
```

### Code Style

- Line length: 100
- Quotes: double (`"`)
- Formatting: ruff (auto-format with `ruff format`)
- Type hints: required on all public signatures
- Docstrings: Google-style with "what" and "why"
- No semi-private naming (`_foo`) — all identifiers are public

### Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add residual-aware anchor selection
fix: handle edge case in drift computation
docs: add comprehensive docstrings across all modules
refactor: convert semi-private attributes to public API
test: add parity tests for cached vs streamed memory
chore: update ruff config
```

---

## Architecture

The method separates model parameters into two groups:

- **Continuous parameters** (`theta`, `R`): Updated every training step via gradient descent
- **Discrete parameters** (`Z`, `A`, `M_g`, `c_g`, `c_l`, `rho`): Refreshed only when drift exceeds a threshold

This hybrid approach makes the method efficient for streaming/large-batch settings, because the expensive discrete refresh is triggered adaptively rather than every step.

### Mathematical Guarantees

1. **PSD kernel**: `K = Phi Phi^T` is positive semidefinite
2. **Rank bound**: `rank(K) <= r_g + m_l`
3. **SPD normal equations**: `S = Phi^T Phi + lambda I` is symmetric positive definite
4. **Orthogonalization**: `Phi_g^T Phi_l_perp ~= 0` up to ridge regularization
5. **Calibration stability**: calibration scalars bounded away from zero

See [docs/architecture.md](docs/architecture.md) for full design rationale and extension points.

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.10+ |
| Numerical | [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) |
| Lint/Format | [ruff](https://docs.astral.sh/ruff/) |
| Type Check | [mypy](https://mypy-lang.org/) (strict) |
| Testing | [pytest](https://docs.pytest.org/) + pytest-cov |
| Build | [Hatchling](https://hatch.pypa.io/) |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features and milestones.

- **v0.1.0** — Current release: core implementation, sklearn API, test suite
- **v0.2.0** — GPU solver backend (CuPy), learned embedding functions
- **v1.0.0** — Stable API, PyPI release, streaming/incremental fit

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Pull request process
- Coding standards
- Test expectations

## Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## Security

Report vulnerabilities to **sachncs@gmail.com** — see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Sachin
