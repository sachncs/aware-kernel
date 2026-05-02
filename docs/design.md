# Design Document

This document describes the architecture, invariants, and extension points of
the aware-kernel implementation.

## Overview

AwareKernel is a hybrid continuous-discrete learner for large-scale kernel
regression. The key idea is to separate representation parameters (continuous:
`theta`, `R`) from basis parameters (discrete: `Z`, `A`, `M_g`, `c_g`, `c_l`,
`d`). Continuous parameters are updated every step; discrete parameters are
refreshed only when drift exceeds a threshold, making the method efficient
for streaming or large-batch settings.

## Module Map

### `aware_kernel.aware`

Central configuration (`TrainingConfig`), state containers (`FullState`),
shared protocols (`Embedder`, `RidgeSolver`), and domain exceptions.

### `aware_kernel.embedding`

- `DenseEmbedder`: Maps input `x` to a dense continuous embedding.
- `Projector`: Applies the learned projection matrix `R` to embeddings.

### `aware_kernel.global_basis`

- `NystromGlobalBasis`: Selects landmarks `Z` and builds a whitened global
  feature map `phi_g(u) = k(u, Z) M_g`.
- `build_whitening_map`: Implements soft-truncated spectral whitening with
  eigenvalue clipping and epsilon scaling.

### `aware_kernel.local_corrective`

- `residual_aware_sample`: Chooses anchors `A` by blending coverage and residual
  weights.
- `compute_sparse_features`: Builds k-NN sparse radial features.
- `orthogonalize_local_features`: Projects local features into the global
  nullspace: `Phi_l_perp = (I - P_g) Phi_l`.

### `aware_kernel.fusion`

- `FusedFeatureBuilder`: Calibrates and gates global and local features:
  `phi = [sqrt(rho) * c_g * phi_g, sqrt(1-rho) * c_l * phi_l_perp]`.
- `compute_gate`: Logistic sigmoid gate `rho = sigma(a)`.

### `aware_kernel.solver`

- `DirectRidgeSolver`: Solves normal equations via Cholesky with conditioning
  checks and jitter fallback.
- `IterativeRidgeSolver`: PCG with diagonal preconditioner for large `m`.

### `aware_kernel.memory`

- `CachedMemoryAccumulator`: Stores `Phi` explicitly; normal equations via
  matrix multiplication (`O(nm)`).
- `StreamedMemoryAccumulator`: Accumulates `S = Phi^T Phi` and `b = Phi^T y`
  directly (`O(m^2)`).

### `aware_kernel.refresh`

- `compute_drift`: Measures representation drift.
- `should_refresh`: Evaluates trigger conditions (drift, cooldown, warmup,
  hysteresis, budget).
- `run_refresh_pipeline`: Full discrete refresh from projected embeddings.

### `aware_kernel.training`

- `TrainingLoop`: Main loop integrating initialization, continuous updates,
  refresh decisions, and evaluation.
- `objectives.py`: Outer-loop bilevel objectives including ridge loss,
  orthogonality penalty, and diversity penalty.
- `callbacks.py`: Logging and checkpoint hooks.

### `aware_kernel.inference`

- `Predictor`: Stateful mean and variance prediction from fused features.

### `aware_kernel.evaluation`

- `datasets.py`: Synthetic benchmarks (linear, polynomial, high-dimensional,
  heteroscedastic).
- `baselines.py`: Ridge, Nyström ridge, and random Fourier feature baselines.
- `metrics.py`: RMSE, MAE, R^2, and max absolute error.
- `runner.py`: Reproducible experiment runner with timing.

### `aware_kernel.api`

- `AwareKernelEstimator`: Sklearn-compatible public API wrapping `TrainingLoop`.

## Key Design Decisions

### Immutability of State

`ContinuousState`, `DiscreteState`, and `FullState` are frozen dataclasses
with `copy_with` methods. This makes refresh boundaries explicit and avoids
surprising mutations during the discrete pipeline.

### Protocols over Inheritance

Core interfaces (`Embedder`, `RidgeSolver`, `MemoryAccumulator`) are defined
as protocols. This keeps modules decoupled and makes swapping implementations
(e.g., a learned embedder or a GPU solver) straightforward.

### Numerical Hardening

All numerical thresholds are centralized in `NumericsConfig`:

- `tau_eig`: Eigenvalue floor for soft truncation.
- `alpha_epsilon`: Dataset-scale epsilon for whitening stability.
- `epsilon_c`: Minimum calibration scale to prevent feature collapse.
- `lambda_min`: Floor on ridge regularization for SPD guarantees.
- `eta_o`: Ridge regularizer for orthogonalization matrix invertibility.
- `kappa_threshold`: Maximum acceptable condition number.

These defaults were chosen to balance accuracy and stability on synthetic
benchmarks.

### Cached vs. Streamed Memory

Cached mode stores the full `Phi` matrix, which is simpler and enables direct
normal-equation construction. Streamed mode accumulates `S` and `b` online,
reducing memory from `O(nm)` to `O(m^2)`. Parity tests confirm both modes
produce identical coefficients when given the same data and seed.

### Refresh Pipeline

The refresh pipeline is intentionally modular:

1. Landmark selection (k-means++).
2. Whitening map construction (eigenvalue clip + soft truncation).
3. Residual computation (current predictor minus targets).
4. Anchor sampling (coverage/residual blend).
5. Local sparse features (k-NN RBF).
6. Orthogonalization (residual projection).
7. Calibration (scalar normalization).
8. Fusion gate estimation.

Each step can be tested in isolation, and the pipeline can be extended with
alternative sampling or whitening strategies.

## Extension Points

### Custom Embedder

Implement the `Embedder` protocol and pass it into a custom `TrainingLoop`
initializer (or extend `AwareKernelEstimator` to accept an embedder factory).

### Alternative Refresh Policy

Implement the `RefreshPolicy` protocol and replace the default
`should_refresh` logic in `TrainingLoop.maybe_refresh`.

### GPU Solver

Replace `DirectRidgeSolver` with a CuPy-backed solver that implements the
`RidgeSolver` protocol. The `TrainingLoop` and `AwareKernelEstimator` will
work unchanged.

## Testing Strategy

The test suite is organized in three tiers:

1. **Unit tests** (`tests/unit/`): Verify shapes, API contracts, error paths,
   and basic correctness for every public function.
2. **Numerical invariant tests** (`tests/numerical/`): Verify paper guarantees
   (PSD, SPD, rank bounds, orthogonality, calibration stability) with
   randomized inputs and tight tolerances.
3. **Integration tests** (`tests/integration/`): End-to-end parity,
   refresh trigger behavior, and training loop convergence on synthetic data.

Coverage is enforced at 80% (currently ~91% on full runs), with omissions for
the evaluation module since it is benchmark scaffolding rather than core
library code.
