"""Discrete refresh pipeline.

Implements Section 3.3 of the method blueprint: the full discrete refresh
pipeline that recomputes all discrete basis parameters.

The pipeline is triggered by the refresh controller and consists of
seven steps:

1. **Landmark selection**: Re-select ``m_g`` landmarks ``Z`` via
   k-means++ from the current projected embeddings.
2. **Kernel and whitening**: Recompute ``W = k(Z, Z)`` and the
   whitening map ``M_g`` via soft-truncated eigendecomposition.
3. **Residual computation**: Fit a global-only ridge to compute
   residuals identifying regions of poor global performance.
4. **Anchor selection**: Re-select ``m_l`` anchors ``A`` via
   residual-aware sampling (blending coverage and residual weights).
5. **Local features**: Build sparse k-NN RBF features and compute
   per-anchor normalizers.
6. **Orthogonalization**: Project local features into the nullspace
   of the global subspace.
7. **Calibration**: Compute trace-based calibration scalars and
   fusion gate.

The pipeline produces a new ``DiscreteState`` that replaces the previous
one atomically.

Design rationale
----------------
The pipeline is intentionally modular: each step is a separate function
call that can be tested in isolation.  This also allows ablation
experiments to selectively disable individual steps (e.g., skipping
orthogonalization or using pure coverage-based anchor selection).
"""

import numpy as np

from aware_kernel.aware.config import TrainingConfig
from aware_kernel.aware.state import DiscreteState, FullState
from aware_kernel.aware.types import Array
from aware_kernel.fusion.builder import FusedFeatureBuilder
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.local_corrective.anchors import (
    compute_residuals,
    residual_aware_sample,
)
from aware_kernel.local_corrective.orthogonalizer import orthogonalize_local_features
from aware_kernel.local_corrective.sparse_features import build_local_features
from aware_kernel.utils.sampling import kmeans_pp


def run_refresh_pipeline(
    state: FullState,
    U_data: Array,
    y_data: Array,
    config: TrainingConfig,
    rng: np.random.Generator | None = None,
) -> DiscreteState:
    """Execute the full discrete refresh pipeline.

    Rebuilds all discrete basis parameters from scratch using the current
    projected embeddings and targets.  The seven steps correspond to the
    pipeline described in Section 3.3 of the method blueprint.

    Args:
        state: Current training state (used for static-scaling check
            and step tracking).
        U_data: Projected embeddings of shape ``(n, d)``.
        y_data: Targets of shape ``(n,)``.
        config: Training configuration containing all sub-configs.
        rng: Optional random generator for reproducibility.

    Returns:
        New ``DiscreteState`` with refreshed basis, calibration, and
        gate values.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Re-select landmarks Z via k-means++.
    # Landmarks capture the geometry of the projected embedding space.
    Z = kmeans_pp(U_data, k=config.m_g, rng=rng)

    # Step 2: Recompute kernel-on-landmarks matrix W and whitening map M_g.
    # The whitening map encodes the spectral structure of the kernel.
    global_basis = NystromGlobalBasis.from_landmarks(Z, config.numerics)

    # Build global features for all data (needed for residual computation).
    Phi_g = global_basis.build_features(U_data)

    # Step 3: Compute residuals from global-only ridge fit.
    # Residuals identify regions where the global basis is insufficient.
    residuals = compute_residuals(Phi_g, y_data, config.lambda_reg)

    # Step 4: Re-select/update anchors A.
    if config.ablation.disable_residual_aware_anchors:
        # Ablation: pure coverage-based sampling (k-means++ only).
        A = kmeans_pp(U_data, k=config.m_l, rng=rng)
    else:
        # Build initial sparse features for coverage weight computation.
        # We need an initial anchor set to compute coverage weights,
        # which are then blended with residual weights for the final
        # anchor selection.
        initial_anchors = kmeans_pp(U_data, k=config.m_l, rng=rng)
        s_initial, _ = build_local_features(
            U_data,
            initial_anchors,
            tau=config.refresh.tau_local,
            k=config.refresh.k_local,
        )

        # Residual-aware sampling blends coverage and residual signals.
        A = residual_aware_sample(
            embeddings=U_data,
            s=s_initial,
            r=residuals,
            alpha_a=config.refresh.alpha_a,
            m_l=config.m_l,
            rng=rng,
        )

    # Step 5: Rebuild local features and normalizers.
    Phi_l, d = build_local_features(
        U_data,
        A,
        tau=config.refresh.tau_local,
        k=config.refresh.k_local,
    )

    # Step 6: Orthogonalize local block against global block.
    # This ensures local features carry only new information not
    # captured by the global basis.
    if config.ablation.disable_orthogonalization:
        Phi_l_perp = Phi_l
    else:
        eta_o = config.numerics.eta_o
        Phi_l_perp = orthogonalize_local_features(Phi_g, Phi_l, eta_o)

    # Step 7: Recompute scaling coefficients.
    if (
        config.ablation.static_scaling
        and state.discrete.c_g is not None
        and state.discrete.c_l is not None
    ):
        # Ablation: freeze calibration scalars after first refresh.
        fused_builder = FusedFeatureBuilder(
            c_g=state.discrete.c_g,
            c_l=state.discrete.c_l,
            rho=state.discrete.rho,
        )
    else:
        fused_builder = FusedFeatureBuilder.from_features(
            Phi_g,
            Phi_l_perp,
            a=0.0,
            epsilon_c=config.numerics.epsilon_c,
        )

    # Assemble the new discrete state atomically.
    new_discrete = DiscreteState(
        Z=Z,
        A=A,
        M_g=global_basis.M_g,
        c_g=fused_builder.c_g,
        c_l=fused_builder.c_l,
        d=d,
        t_r=state.step,
        b_t=1,
        rho=fused_builder.rho,
    )

    return new_discrete
