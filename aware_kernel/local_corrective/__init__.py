"""Local corrective features and residual-aware anchor selection.

This package implements the local corrective stage, which complements
the global basis with sparse, localized features:

* ``anchors``: Residual-aware anchor selection blending coverage and
  residual weights.
* ``sparse_features``: k-NN sparse RBF features with per-anchor
  normalization.
* ``orthogonalizer``: Ridge-regularized projection into the global
  nullspace to ensure feature diversity.
"""
