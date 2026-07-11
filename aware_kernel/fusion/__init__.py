"""Feature calibration, gating, and fusion.

This package implements the fusion stage of the pipeline, which combines
calibrated global and local features into a single feature vector:

1. **Calibration** (``calibration.py``): Trace-based normalization of
   feature blocks to prevent scale dominance.
2. **Gating** (``gate.py``): Logistic sigmoid gate ``rho = sigma(a)``
   that balances global and local contributions.
3. **FusedFeatureBuilder** (``builder.py``): Orchestrates calibration
   and gating, holding precomputed constants that are refreshed
   adaptively.
"""
