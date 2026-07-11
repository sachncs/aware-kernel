"""Memory accumulators for cached and streamed modes.

This package provides two strategies for accumulating normal equations
``S = Phi^T Phi`` and ``b = Phi^T y``:

* ``CachedMemoryAccumulator``: Stores the full feature matrix ``Phi``.
  O(n * m) memory, enables direct normal-equation construction.
* ``StreamedMemoryAccumulator``: Accumulates ``S`` and ``b`` directly.
  O(m^2) memory, discards individual samples.

Both modes produce identical coefficients when given the same data and
seed (verified by parity tests).  The choice is controlled by
``TrainingConfig.memory_mode``.
"""
