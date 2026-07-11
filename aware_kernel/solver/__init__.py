"""Ridge regression solvers and normal equation helpers.

This package implements the ridge regression stage:

* ``ridge``: Direct (Cholesky) and iterative (PCG) ridge solvers with
  conditioning checks and jitter fallback.
* ``normal_eq``: Normal equation assembly helpers for batch and
  incremental computation.
* ``preconditioner``: Diagonal Jacobi preconditioner for PCG acceleration.
"""
