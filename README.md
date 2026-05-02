# AwareKernel: Refresh-Aware Hybrid Continuous-Discrete Low-Rank Kernel Learning

**AwareKernel** is a research implementation of refresh-aware hybrid
continuous-discrete low-rank kernel learning. It combines a continuously
updated embedding projection with a discrete refresh pipeline for the kernel
basis, yielding scalable, adaptive kernel regression with explicit feature maps
and strong numerical guarantees.

## Features

- **Explicit-feature kernel system**: Build `phi(x)` feature maps with a
  PSD guarantee `K = Phi Phi^T >= 0`.
- **Global Nyström basis**: Soft-truncated whitening for stable, low-rank
  global approximation.
- **Local corrective features**: Residual-aware anchor sampling and k-NN
  sparsity for adaptive local refinement.
- **Residual orthogonalization**: `Phi_l_perp = (I - P_g) Phi_l` ensures
  local features live in the global nullspace.
- **Feature calibration and fusion**: Gate `rho = sigma(a)` balances global
  and local contributions.
- **Refresh controller**: Drift-aware trigger with cooldown, warmup,
  hysteresis, and amortized budget.
- **Two memory modes**: Cached (`O(nm)`) and streamed (`O(m^2)`) for
  different scale regimes.
- **Numerical stabilization**: Eigenvalue clipping, soft spectral truncation,
  epsilon scaling, and Cholesky with jitter fallback.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10, NumPy >= 1.24, SciPy >= 1.10, and scikit-learn >= 1.3.

## Quick Start

```python
import numpy as np
from aware_kernel import AwareKernelEstimator

# Synthetic data
rng = np.random.default_rng(42)
X = rng.standard_normal((200, 4))
y = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.1 * rng.standard_normal(200)

# Fit
model = AwareKernelEstimator(
    embedding_dim=4,
    m_g=32,
    m_l=8,
    lambda_reg=1e-2,
    max_steps=20,
    seed=42,
)
model.fit(X, y)

# Predict
X_test = rng.standard_normal((50, 4))
y_pred = model.predict(X_test)
print(f"R^2 on training data: {model.score(X, y):.4f}")
```

## Testing

Run the full test suite with coverage:

```bash
python -m pytest tests/ -v -p no:asyncio
```

Run only unit tests:

```bash
python -m pytest tests/unit/ -v -p no:asyncio
```

Run numerical invariant tests:

```bash
python -m pytest tests/numerical/ -v -p no:asyncio
```

## Architecture

The package is organized into modular phases:

| Module | Purpose |
|--------|---------|
| `aware_kernel.aware` | Configuration, state, types, and exceptions |
| `aware_kernel.embedding` | Dense embedder and projection matrix `R` |
| `aware_kernel.global_basis` | Nyström landmark selection and whitening |
| `aware_kernel.local_corrective` | Anchor sampling, sparse features, orthogonalization |
| `aware_kernel.fusion` | Calibration, gating, and fused feature building |
| `aware_kernel.solver` | Ridge regression via Cholesky or PCG |
| `aware_kernel.memory` | Cached and streamed normal-equation accumulators |
| `aware_kernel.refresh` | Drift, budget, controller, and refresh pipeline |
| `aware_kernel.training` | Training loop, objectives, and callbacks |
| `aware_kernel.inference` | Mean and variance prediction |
| `aware_kernel.evaluation` | Datasets, baselines, metrics, and experiment runner |
| `aware_kernel.api` | Sklearn-compatible `AwareKernelEstimator` |

## Mathematical Guarantees

The implementation verifies the following paper invariants:

1. **PSD kernel**: `K = Phi Phi^T` is positive semidefinite.
2. **Rank bound**: `rank(K) <= r_g + m_l`.
3. **SPD normal equations**: `S = Phi^T Phi + lambda I` is symmetric
   positive definite with conditioning checked against `kappa_threshold`.
4. **Orthogonalization**: `Phi_g^T Phi_l_perp ~= 0` up to ridge regularization.
5. **Calibration stability**: `c_g^2 tr(Phi_g^T Phi_g)` and
   `c_l^2 tr(Phi_l^T Phi_l)` are bounded away from zero.

See `docs/design.md` for full design rationale and extension points.

## License

MIT License. See `LICENSE` for details.
