"""Inference utilities for mean and variance prediction.

This package provides the prediction stage of the pipeline:

* ``predictor``: Stateful mean and variance prediction from fused
  features using the ridge regression solution.

The predictive equations follow Bayesian linear regression:

    ``mu_*(x_*) = phi(x_*)^T w*``
    ``sigma_*^2(x_*) = phi(x_*)^T phi(x_*) - phi(x_*)^T S^{-1} phi(x_*)``
"""
