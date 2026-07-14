# Frequently Asked Questions

## General

### What is AwareKernel?

AwareKernel is a research implementation of **refresh-aware hybrid continuous-discrete low-rank kernel learning** for scalable, adaptive kernel regression. It combines a continuously updated embedding projection with a discrete refresh pipeline for the kernel basis.

### When should I use AwareKernel?

AwareKernel is designed for:
- Large-scale kernel regression where standard Nyström methods are too slow
- Streaming or online learning settings where data arrives sequentially
- Problems requiring adaptive kernel basis updates without full recomputation
- Research into hybrid continuous-discrete learning methods

### How does AwareKernel compare to standard Nyström ridge regression?

Standard Nyström methods rebuild the kernel basis from scratch each time. AwareKernel separates continuous parameters (updated every step) from discrete parameters (refreshed only when drift exceeds a threshold). This makes it much more efficient for streaming settings.

## Installation

### What Python versions are supported?

Python 3.10, 3.11, and 3.12 are officially supported and tested in CI.

### What are the dependencies?

- NumPy >= 1.24
- SciPy >= 1.10
- scikit-learn >= 1.3

### How do I install for development?

```bash
git clone https://github.com/sachncs/aware-kernel.git
cd aware-kernel
pip install -e ".[dev]"
```

## Usage

### How do I choose between cached and streamed memory modes?

- **Cached** (`memory_mode="cached"`): Stores the full feature matrix. Use when memory is not a constraint and you want simplicity.
- **Streamed** (`memory_mode="streamed"`): Accumulates normal equations directly. Use when `n` is very large and you cannot afford O(nm) memory.

### What do the ablation flags do?

Ablation flags disable specific components for ablation studies:

| Flag | Effect |
|------|--------|
| `disable_refresh` | Never refresh the discrete basis |
| `disable_hysteresis` | Always consider refresh (ignore hysteresis) |
| `disable_cooldown` | No minimum gap between refreshes |
| `disable_residual_aware_anchors` | Use coverage-only anchor sampling |
| `disable_orthogonalization` | Skip local feature orthogonalization |
| `disable_diversity_penalty` | Remove diversity regularization |
| `static_scaling` | Freeze calibration after first refresh |

### How do I set a random seed for reproducibility?

Pass `seed=<integer>` to the constructor:

```python
model = AwareKernelEstimator(seed=42)
```

### Can I use custom embedding functions?

Yes. Implement the `Embedder` protocol and use it with a custom `TrainingLoop`:

```python
from aware_kernel.aware.types import Embedder

class MyEmbedder:
    def embed(self, X: np.ndarray) -> np.ndarray:
        return my_custom_embedding(X)
```

See `docs/architecture.md` for details.

## Performance

### How many landmarks (`m_g`) should I use?

General guidelines:

| Dataset size | Recommended `m_g` |
|-------------|-------------------|
| < 1K samples | 32 - 64 |
| 1K - 10K samples | 128 - 512 |
| 10K - 100K samples | 512 - 2048 |
| > 100K samples | 2048+ |

Start small and increase if needed. More landmarks = better approximation but higher cost.

### How do I reduce training time?

- Reduce `max_steps`
- Reduce `m_g` (global landmarks) and `m_l` (local anchors)
- Use `disable_refresh=True` to skip discrete refresh entirely
- Increase `t_cool` to refresh less often

## Troubleshooting

### I get a conditioning warning. What does it mean?

The solver detected that the normal equation matrix is ill-conditioned (condition number exceeds `kappa_threshold`). This can happen when:
- `lambda_reg` is too small
- Features are nearly collinear
- The dataset has very different scales

Try increasing `lambda_reg` or preprocessing your data (e.g., standard scaling).

### The model seems to not converge. What should I try?

- Increase `max_steps`
- Decrease `lambda_reg` (but watch for conditioning issues)
- Adjust `delta_hi` (lower = more frequent refreshes)
- Check that your data is properly scaled
- Try different `embedding_dim` and `m_g` values

### How do I report bugs?

Please open an issue on GitHub using the bug report template. Include:
- A minimal reproducible example
- Full error traceback
- Python and package versions
- OS information

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.
