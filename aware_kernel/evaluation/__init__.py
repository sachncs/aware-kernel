"""Evaluation datasets, baselines, metrics, and experiment runner.

This package provides the scaffolding for benchmarking aware-kernel
against baseline methods:

* ``datasets``: Synthetic regression datasets (linear, polynomial,
  high-dimensional, heteroscedastic).
* ``baselines``: Ridge, Nyström ridge, and random Fourier feature
  baselines.
* ``metrics``: RMSE, MAE, R^2, and max absolute error.
* ``runner``: Reproducible experiment runner with timing and markdown
  table formatting.
"""
