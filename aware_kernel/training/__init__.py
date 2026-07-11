"""Training loop, objectives, and callbacks.

This package implements the core training orchestration:

* ``loop``: Main training loop integrating initialization, continuous
  updates, refresh decisions, and evaluation.
* ``objectives``: Bilevel outer-loop objectives including ridge loss,
  Frobenius regularizer, orthogonality penalty, and diversity penalty.
* ``optimizer``: Gradient-based outer-loop optimizer for the projection
  matrix ``R`` using finite-difference gradient estimation.
* ``callbacks``: Logging and checkpoint hooks for monitoring training.
"""
