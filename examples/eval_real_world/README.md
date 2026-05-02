# Real-World Evaluation Example for Aware-Kernel

This directory implements the full **Evaluation Protocol** (`EVALUATION_PROTOCOL.md`) on a diverse collection of real-world, HuggingFace, and synthetic datasets.

---

## Structure

```
eval_real_world/
├── README.md                 # This file
├── data_loaders/
│   ├── __init__.py
│   └── loaders.py            # Dataset loaders (sklearn, OpenML, HF, synthetic)
├── scripts/
│   ├── run_evaluation.py     # Main evaluation runner
│   └── analyze_results.py    # Post-processing: plots, tables, LaTeX
└── results/                  # Output directory (created at runtime)
    ├── results.md            # Markdown tables
    ├── results.csv           # Machine-readable CSV
    ├── pareto_{dataset}.png  # Pareto front plots
    ├── ablations_{dataset}_{tier}.png
    └── results_table.tex     # LaTeX table
```

---

## Datasets

### Sklearn / OpenML (UCI Tabular)

| Dataset | Source | n | d | Type |
|---------|--------|---|---|------|
| **CaliforniaHousing** | `sklearn.datasets` | ~20,640 | 8 | Real-world regression |
| **Diabetes** | `sklearn.datasets` | 442 | 10 | Medical regression |
| **Protein** | OpenML 195 | 159 | 15 | UCI tabular (Section 2.2) |
| **Kin8nm** | OpenML 189 | 8,192 | 8 | UCI tabular (Section 2.2) |
| **YearPredictionMSD** | OpenML 227 | ~8,192 | 12 | UCI tabular (Section 2.2) |
| **NYCTaxi** | OpenML 42729 | **~581,835** | 18 | Large-scale benchmark (Section 2.2, n≥1e5) |

### HuggingFace Datasets (`inria-soda/tabular-benchmark`)

| Dataset | Config | n | d | Notes |
|---------|--------|---|---|-------|
| **WineQuality** | `reg_num_wine_quality` | ~6,497 | 11 | Tabular regression |
| **Abalone** | `reg_num_abalone` | ~4,177 | 7 | Tabular regression |
| **Superconduct** | `reg_num_superconduct` | ~21,263 | 80 | High-dimensional regression |
| **HouseSales** | `reg_num_house_sales` | ~21,613 | 15 | Tabular regression |
| **Elevators** | `reg_num_elevators` | ~16,599 | 18 | Tabular regression |
| **CPUActivity** | `reg_num_cpu_act` | ~8,192 | 21 | Tabular regression |
| **Diamonds** | `reg_cat_diamonds` | ~53,940 | 9 | Categorical features encoded |

### Synthetic

| Dataset | Generator | n | d | Purpose |
|---------|-----------|---|---|---------|
| **SyntheticSpatial** | `make_synthetic_spatial` | configurable | 2 | Stress global/local decomposition (Section 2.1) |

---

## Budget Tiers

Aligned with Section 5 of the protocol:

| Tier | `m_g` | `m_l` | `max_steps` | `embedding_dim` | `total_refresh_budget` |
|------|-------|-------|-------------|-----------------|------------------------|
| **Small** | 128 | 32 | 200 | 16 | 5.0 |
| **Medium** | 512 | 128 | 1,000 | 64 | 20.0 |
| **Large** | 2,048 | 512 | 2,000 | 128 | 50.0 |

---

## Usage

### Quick Start (small smoke test)

```bash
cd examples/eval_real_world
python -m scripts.run_evaluation \
    --datasets Diabetes CaliforniaHousing \
    --tiers Small \
    --n_seeds 3 \
    --output_dir results
```

### Full Sweep (all datasets, all tiers, all ablations)

```bash
python -m scripts.run_evaluation \
    --datasets all \
    --tiers all \
    --n_seeds 5 \
    --output_dir results
```

### Analyze Results

```bash
python -m scripts.analyze_results \
    --results results/results.csv \
    --output_dir results
```

---

## Outputs

### Metrics (per dataset × model × tier)

- **RMSE / MAE / R²** – predictive quality (Section 6.1)
- **Train Time (s)** – wall-clock fit time (Section 6.3)
- **Inference Latency (s)** – per-sample predict time (Section 6.3)
- **Peak Memory (MB)** – max resident memory via `tracemalloc` (Section 6.3)
- **Refresh Count** – number of discrete refreshes triggered (Section 6.4)
- **Condition Proxy** – `κ(ΦᵀΦ + λI)` (Section 6.4)

### Stability-over-Refits (Section 6.5)

- `Var(RMSE)` across repeated initializations
- `Var(train_time)`
- `Var(κ(ΦᵀΦ + λI))`

### Visualizations

- **Pareto plots** – RMSE vs train time / peak memory (Section 9)
- **Ablation bars** – relative RMSE change for each ablation (Section 7)

---

## Models Evaluated

### Main Method
- **AwareKernel** – full refresh-aware hybrid kernel with continuous R optimization

### Baselines (Section 4)
- **Ridge** – dense exact ridge in original feature space
- **Nyström** – static Nyström feature ridge (no refresh, no local corrective)
- **RFF** – Random Fourier Features ridge

### Ablations (Section 7)
- `AK-NoRefresh` – static basis after first refresh
- `AK-NoHysteresis` – `b_t = 1` permanently
- `AK-NoCooldown` – cooldown disabled
- `AK-NoResidAnchors` – coverage-only anchor sampling
- `AK-NoOrthog` – skip local orthogonalization
- `AK-NoDivPenalty` – `γ_div = 0`
- `AK-StaticScaling` – freeze calibration scalars after first refresh

---

## Reproducibility (Section 10)

- Fixed random seeds: `[42, 43, 44, 45, 46]` (default)
- Hardware: CPU-only (configurable to GPU via PyTorch embedder)
- Software versions logged in environment
- Exact rank/budget settings listed per tier table above
- Stopping criteria: `max_steps` with eval every `max_steps // 20`

---

## Extending

To add a new dataset, edit `data_loaders/loaders.py`:

```python
def load_my_dataset() -> Tuple[np.ndarray, np.ndarray]:
    ...
    return X, y
```

Then register it in `DATASET_LOADERS`:

```python
DATASET_LOADERS = {
    ...,
    "MyDataset": load_my_dataset,
}
```

Run with `--datasets MyDataset`.
