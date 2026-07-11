"""Embedding and projection modules.

This package implements the first two stages of the aware-kernel pipeline:

* ``DenseEmbedder``: Maps raw inputs ``x`` into a dense embedding space
  via a learned linear transformation ``f_theta(x) = x @ theta + bias``.
* ``Projector``: Applies L2 normalization and projects embeddings through
  the learned matrix ``R``: ``u = R * normalize(e)``.

The embedding parameters (``theta``, ``R``) are continuous parameters
updated every training step.  The embedder satisfies the ``Embedder``
protocol, making it swappable for neural networks or other embedding
strategies.
"""
