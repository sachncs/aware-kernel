"""AwareKernel: refresh-aware hybrid continuous-discrete low-rank kernel learning.

This package implements the aware-kernel method for scalable, adaptive
kernel regression.  The method separates model parameters into continuous
(updated every step) and discrete (refreshed adaptively) groups, enabling
efficient training with dynamic basis adaptation.

Public API
----------
* ``AwareKernelEstimator``: Sklearn-compatible estimator with ``fit``,
  ``predict``, and ``score`` methods.

Quick start::

    from aware_kernel import AwareKernelEstimator

    model = AwareKernelEstimator(embedding_dim=8, m_g=32, m_l=8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

from aware_kernel._version import __version__
from aware_kernel.api import AwareKernelEstimator

__all__ = ["__version__", "AwareKernelEstimator"]
