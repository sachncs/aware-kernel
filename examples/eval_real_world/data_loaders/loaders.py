"""Dataset loaders for real-world evaluation.

Supports three sources:
  - scikit-learn built-ins / fetch_openml (UCI tabular)
  - HuggingFace datasets (tabular-benchmark, etc.)
  - Synthetic spatial fields

All loaders return (X: np.ndarray[float64], y: np.ndarray[float64]).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from datasets import load_dataset
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.datasets import load_diabetes as _load_diabetes


# ---------------------------------------------------------------------------
# Sklearn / OpenML loaders
# ---------------------------------------------------------------------------
def load_california_housing(*_args) -> tuple[np.ndarray, np.ndarray]:
    """California Housing: n≈20_640, d=8."""
    data = fetch_california_housing()
    return data.data.astype(np.float64), data.target.astype(np.float64)


def load_diabetes(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Diabetes: n=442, d=10."""
    data = _load_diabetes()
    return data.data.astype(np.float64), data.target.astype(np.float64)


def load_protein(*_args) -> tuple[np.ndarray, np.ndarray]:
    """UCI Protein (OpenML 195): n=159, d=15."""
    data = fetch_openml(data_id=195, parser="auto", as_frame=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    return X, y


def load_kin8nm(*_args) -> tuple[np.ndarray, np.ndarray]:
    """UCI kin8nm (OpenML 189): n=8192, d=8."""
    data = fetch_openml(data_id=189, parser="auto", as_frame=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    return X, y


def load_yearpredictionmsd(*_args) -> tuple[np.ndarray, np.ndarray]:
    """YearPredictionMSD (OpenML 227): n≈8192, d=12.

    Note: this is the OpenML subsample. The full UCI version has ~515K samples.
    """
    data = fetch_openml(data_id=227, parser="auto", as_frame=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    return X, y


def load_nyc_taxi(*_args) -> tuple[np.ndarray, np.ndarray]:
    """NYC Taxi Green Dec 2016 (OpenML 42729): n≈581_835, d=18.

    Large-scale benchmark for memory / runtime stress.
    """
    data = fetch_openml(data_id=42729, parser="auto", as_frame=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# HuggingFace datasets loaders
# ---------------------------------------------------------------------------
def _load_hf_regression(config: str, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Generic loader for inria-soda/tabular-benchmark regression configs."""
    ds = load_dataset("inria-soda/tabular-benchmark", config, split="train")
    df = ds.to_pandas()

    # Separate target
    y = df[target_col].to_numpy(dtype=np.float64)
    X = df.drop(columns=[target_col]).to_numpy(dtype=np.float64)

    # Handle potential NaNs / infs
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    return X, y


def load_hf_wine_quality(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Wine Quality (HF tabular-benchmark reg_num): n≈6497, d=11."""
    return _load_hf_regression("reg_num_wine_quality", "quality")


def load_hf_abalone(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Abalone (HF tabular-benchmark reg_num): n≈4177, d=7."""
    return _load_hf_regression("reg_num_abalone", "Classnumberofrings")


def load_hf_superconduct(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Superconduct (HF tabular-benchmark reg_num): n≈21263, d=80."""
    return _load_hf_regression("reg_num_superconduct", "criticaltemp")


def load_hf_house_sales(*_args) -> tuple[np.ndarray, np.ndarray]:
    """House Sales (HF tabular-benchmark reg_num): n≈21613, d=15."""
    return _load_hf_regression("reg_num_house_sales", "price")


def load_hf_elevators(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Elevators (HF tabular-benchmark reg_num): n≈16599, d=18."""
    return _load_hf_regression("reg_num_elevators", "Elev")


def load_hf_cpu_act(*_args) -> tuple[np.ndarray, np.ndarray]:
    """CPU Activity (HF tabular-benchmark reg_num): n≈8192, d=21."""
    return _load_hf_regression("reg_num_cpu_act", "usr")


def load_hf_diamonds(*_args) -> tuple[np.ndarray, np.ndarray]:
    """Diamonds (HF tabular-benchmark reg_cat): n≈53940, d=9 (with cat enc)."""
    ds = load_dataset("inria-soda/tabular-benchmark", "reg_cat_diamonds", split="train")
    df = ds.to_pandas()
    # Encode categoricals
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes.astype(np.float64)
    y = df["price"].to_numpy(dtype=np.float64)
    X = df.drop(columns=["price"]).to_numpy(dtype=np.float64)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[valid], y[valid]


# ---------------------------------------------------------------------------
# Synthetic spatial field
# ---------------------------------------------------------------------------
def load_synthetic_spatial(
    rng: np.random.Generator,
    n: int = 5000,
    noise: float = 0.1,
    snr: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2D spatial field: smooth + local high-frequency.

    Ground-truth field:
        f(x) = sin(2*pi*x1) * cos(2*pi*x2)          (smooth global)
               + 0.5 * sin(20*pi*x1) * sin(20*pi*x2) (local high-freq)
    Noise scale derived from SNR.
    """
    X = rng.uniform(0.0, 1.0, size=(n, 2))
    signal = np.sin(2.0 * np.pi * X[:, 0]) * np.cos(
        2.0 * np.pi * X[:, 1]
    ) + 0.5 * np.sin(20.0 * np.pi * X[:, 0]) * np.sin(20.0 * np.pi * X[:, 1])
    sig_var = float(np.var(signal))
    noise_std = np.sqrt(sig_var / snr) if snr > 0 else 0.0
    y = signal + noise_std * rng.standard_normal(n)
    return X.astype(np.float64), y.astype(np.float64)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
DATASET_LOADERS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    # Sklearn / OpenML
    "CaliforniaHousing": load_california_housing,
    "Diabetes": load_diabetes,
    "Protein": load_protein,
    "Kin8nm": load_kin8nm,
    "YearPredictionMSD": load_yearpredictionmsd,
    "NYCTaxi": load_nyc_taxi,
    # HuggingFace
    "WineQuality": load_hf_wine_quality,
    "Abalone": load_hf_abalone,
    "Superconduct": load_hf_superconduct,
    "HouseSales": load_hf_house_sales,
    "Elevators": load_hf_elevators,
    "CPUActivity": load_hf_cpu_act,
    "Diamonds": load_hf_diamonds,
    # Synthetic
    "SyntheticSpatial": load_synthetic_spatial,
}
