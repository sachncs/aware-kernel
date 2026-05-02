"""Discrete refresh pipeline.

Implements Section 3.3 (discrete refresh pipeline):
    1. Re-select landmarks Z
    2. Recompute W and M_g
    3. Re-select/update anchors A
    4. Rebuild local features and normalizers
    5. Orthogonalize local block against global block
    6. Recompute scaling coefficients
    7. Freeze calibration values until next refresh
"""

from typing import Optional

import numpy as np

from aware_kernel.aware.config import NumericsConfig, RefreshConfig, TrainingConfig
from aware_kernel.aware.state import DiscreteState, FullState
from aware_kernel.aware.types import Array
from aware_kernel.embedding.projector import Projector
from aware_kernel.fusion.builder import FusedFeatureBuilder
from aware_kernel.global_basis.nystrom import NystromGlobalBasis
from aware_kernel.local_corrective.anchors import compute_residuals, residual_aware_sample
from aware_kernel.local_corrective.orthogonalizer import orthogonalize_local_features
from aware_kernel.local_corrective.sparse_features import build_local_features
from aware_kernel.utils.sampling import kmeans_pp


def run_refresh_pipeline(
    state: FullState,
    U_data: Array,
    y_data: Array,
    config: TrainingConfig,
    rng: Optional[np.random.Generator] = None,
) -> DiscreteState:
    """Execute the full discrete refresh pipeline.

    Args:
        state: Current training state.
        U_data: Projected embeddings of shape (n, d).
        y_data: Targets of shape (n,).
        config: Training configuration.
        rng: Optional random generator.

    Returns:
        Updated discrete state with refreshed basis.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Step 1: Re-select landmarks Z
    Z = kmeans_pp(U_data, k=config.m_g, rng=rng)

    # Step 2: Recompute W and M_g
    global_basis = NystromGlobalBasis.from_landmarks(Z, config.numerics)

    # Build global features for all data
    Phi_g = global_basis.build_features(U_data)

    # Step 3: Compute residuals and re-select anchors A
    residuals = compute_residuals(Phi_g, y_data, config.lambda_reg)

    if config.ablation.disable_residual_aware_anchors:
        # Pure coverage-based sampling (kmeans++ only)
        A = kmeans_pp(U_data, k=config.m_l, rng=rng)
    else:
        # Build initial sparse features for coverage weights
        # Use a simple initial anchor set for coverage computation
        initial_anchors = kmeans_pp(U_data, k=config.m_l, rng=rng)
        s_initial, _ = build_local_features(
            U_data,
            initial_anchors,
            tau=config.refresh.tau_local,
            k=config.refresh.k_local,
        )

        A = residual_aware_sample(
            embeddings=U_data,
            s=s_initial,
            r=residuals,
            alpha_a=config.refresh.alpha_a,
            m_l=config.m_l,
            rng=rng,
        )

    # Step 4: Rebuild local features and normalizers
    Phi_l, d = build_local_features(
        U_data,
        A,
        tau=config.refresh.tau_local,
        k=config.refresh.k_local,
    )

    # Step 5: Orthogonalize local block against global block
    if config.ablation.disable_orthogonalization:
        Phi_l_perp = Phi_l
    else:
        eta_o = config.numerics.eta_o
        Phi_l_perp = orthogonalize_local_features(Phi_g, Phi_l, eta_o)

    # Step 6: Recompute scaling coefficients
    if config.ablation.static_scaling and state.discrete.c_g is not None and state.discrete.c_l is not None:
        # Freeze calibration scalars after first refresh
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

    # Step 7: Build updated discrete state
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
