# Getting Started

This guide walks you through installing and using AwareKernel for the first time.

## Prerequisites

- Python >= 3.10
- pip or conda package manager

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/sachncs/aware-kernel.git
cd aware-kernel
pip install -e ".[dev]"
```

### Verify installation

```python
import aware_kernel
print(aware_kernel.__version__)
```

## Quick Start

### Basic regression

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

### Using different memory modes

```python
# Cached mode (default) - stores full feature matrix, O(nm) memory
model_cached = AwareKernelEstimator(memory_mode="cached")

# Streamed mode - accumulates normal equations directly, O(m^2) memory
model_streamed = AwareKernelEstimator(memory_mode="streamed")
```

### Using callbacks for monitoring

```python
from aware_kernel.training.callbacks import LoggingCallback

# Log every 10 steps
model = AwareKernelEstimator(log_interval=10, max_steps=100)
model.fit(X_train, y_train)
```

## Configuration

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Dimension of continuous embedding space |
| `m_g` | 512 | Global basis rank (landmarks) |
| `m_l` | 128 | Local corrective rank (anchors) |
| `lambda_reg` | 1e-3 | Ridge regularization parameter |
| `max_steps` | 1000 | Maximum training steps |
| `seed` | None | Random seed for reproducibility |

### Refresh parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_hi` | 0.1 | Drift threshold to trigger refresh |
| `t_cool` | 50 | Minimum steps between refreshes |
| `t_warmup` | 10 | Minimum step before first refresh |
| `gamma_cost` | 0.01 | Validation gain threshold |
| `total_refresh_budget` | inf | Total amortized refresh budget |

### Ablation flags

| Parameter | Default | Description |
|-----------|---------|-------------|
| `disable_refresh` | False | Skip all discrete refreshes |
| `disable_hysteresis` | False | Force hysteresis = 1 permanently |
| `disable_cooldown` | False | Set effective cooldown to 0 |
| `disable_residual_aware_anchors` | False | Use coverage-only sampling |
| `disable_orthogonalization` | False | Skip local orthogonalization |
| `disable_diversity_penalty` | False | Set diversity penalty to 0 |
| `static_scaling` | False | Freeze calibration after first refresh |

## Next steps

- Read the [Architecture Guide](architecture.md) for module details
- Check the [Design Document](design.md) for mathematical foundations
- Review `examples/eval_real_world/` for real-world evaluation scripts
- See the [FAQ](faq.md) for common questions
