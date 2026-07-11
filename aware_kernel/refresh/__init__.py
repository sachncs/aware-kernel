"""Refresh controller, drift detection, and discrete refresh pipeline.

This package implements the adaptive refresh mechanism that determines
when discrete basis parameters should be recomputed:

* ``controller``: Five-condition decision logic (drift, cooldown,
  warmup, hysteresis, budget).
* ``drift``: Relative Frobenius-norm drift computation.
* ``pipeline``: Full seven-step discrete refresh pipeline.
* ``budget``: Amortized cost tracking and budget enforcement.
"""
