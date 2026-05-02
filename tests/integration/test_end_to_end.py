"""End-to-end integration tests for aware-kernel."""

import numpy as np

from aware_kernel.aware.config import MemoryMode, TrainingConfig
from aware_kernel.training.loop import TrainingLoop


class TestEndToEnd:
    """End-to-end tests on synthetic data."""

    def _make_data(self, rng: np.random.Generator, n: int = 200, d: int = 4) -> tuple:
        """Create synthetic regression data."""
        X = rng.standard_normal((n, d))
        true_w = rng.standard_normal(d)
        y = X @ true_w + 0.1 * rng.standard_normal(n)
        return X, y

    def test_fit_predict_cached(self, rng: np.random.Generator) -> None:
        """Full fit/predict cycle with cached memory mode."""
        X_train, y_train = self._make_data(rng, n=200)
        X_test, y_test = self._make_data(rng, n=50)

        config = TrainingConfig(
            embedding_dim=4,
            m_g=32,
            m_l=8,
            lambda_reg=1e-2,
            memory_mode=MemoryMode.CACHED,
            seed=42,
        )
        loop = TrainingLoop(config)
        state = loop.initialize_state(X_train, y_train)

        # Run a few training steps
        for step in range(1, 11):
            state = loop.continuous_update(state, X_train[:20], y_train[:20])
            state = loop.maybe_refresh(state, X_test, y_test)
            state = state.copy_with(step=step)

        metrics = loop.evaluate(state, X_test, y_test)
        assert metrics["rmse"] >= 0.0

    def test_cached_vs_streamed_parity(self, rng: np.random.Generator) -> None:
        """Cached and streamed modes should produce same coefficients."""
        X, y = self._make_data(rng, n=200)

        config_cached = TrainingConfig(
            embedding_dim=4,
            m_g=32,
            m_l=8,
            lambda_reg=1e-2,
            memory_mode=MemoryMode.CACHED,
            seed=42,
        )
        config_streamed = TrainingConfig(
            embedding_dim=4,
            m_g=32,
            m_l=8,
            lambda_reg=1e-2,
            memory_mode=MemoryMode.STREAMED,
            seed=42,
        )

        loop_cached = TrainingLoop(config_cached)
        loop_streamed = TrainingLoop(config_streamed)

        state_c = loop_cached.initialize_state(X, y)
        state_s = loop_streamed.initialize_state(X, y)

        # Coefficients should be very close since they use the same data and seed
        np.testing.assert_allclose(state_c.w, state_s.w, atol=1e-6)

    def test_psd_invariant(self, rng: np.random.Generator) -> None:
        """Kernel matrix K = Phi Phi^T should be PSD."""
        X, y = self._make_data(rng, n=100)
        config = TrainingConfig(embedding_dim=4, m_g=32, m_l=8, lambda_reg=1e-2, seed=42)
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)

        # Build features manually
        embedder = state.continuous.theta["embedder"]
        embeddings = embedder.embed(X)
        from aware_kernel.embedding.projector import Projector
        projector = Projector(state.continuous.R)
        U = projector.transform(embeddings)
        phi = loop._build_fused_features(U, state.discrete)

        K = phi @ phi.T
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-8)

    def test_rank_bound(self, rng: np.random.Generator) -> None:
        """rank(K) <= r_g + m_l."""
        X, y = self._make_data(rng, n=100)
        config = TrainingConfig(embedding_dim=4, m_g=32, m_l=8, lambda_reg=1e-2, seed=42)
        loop = TrainingLoop(config)
        state = loop.initialize_state(X, y)

        embedder = state.continuous.theta["embedder"]
        embeddings = embedder.embed(X)
        from aware_kernel.embedding.projector import Projector
        projector = Projector(state.continuous.R)
        U = projector.transform(embeddings)
        phi = loop._build_fused_features(U, state.discrete)

        K = phi @ phi.T
        rank = np.linalg.matrix_rank(K, tol=1e-6)
        max_rank = state.discrete.M_g.shape[1] + state.discrete.A.shape[0]
        assert rank <= max_rank
