# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Comprehensive Google-style docstrings across all 34 source modules
- Enhanced module-level documentation explaining package responsibilities and architecture
- Algorithm descriptions, complexity analysis, and mathematical formulas in docstrings
- Thread-safety notes, design rationale, and recovery strategies in exception docstrings
- References to academic papers (Drineas & Mahoney 2005, Arthur & Vassilvitskii 2007, Spall 1992)
- MIT License file
- CONTRIBUTING.md with development guidelines
- CODE_OF_CONDUCT.md (Contributor Covenant v2.1)
- SECURITY.md with vulnerability reporting policy
- CHANGELOG.md following Keep a Changelog format
- .editorconfig for consistent formatting
- .gitattributes for line ending normalization
- GitHub issue templates (bug report, feature request)
- GitHub pull request template
- GitHub Actions CI workflow (lint, type-check, test)
- Dependabot configuration for automated dependency updates
- Funding configuration with placeholder URLs
- docs/getting-started.md with detailed setup instructions
- docs/architecture.md with full module documentation
- docs/deployment.md for publishing guidelines
- docs/faq.md with common questions and answers

### Changed

- Converted semi-private naming conventions to public API (removed `_` prefix from attributes with property accessors)
- Replaced `print` statements with `logging` module in training callbacks
- Updated tests to reflect public attribute naming
- Updated README.md with badges, detailed sections, and improved examples
- Updated pyproject.toml with correct repository URLs, classifiers, and metadata
- Updated .gitignore with additional Python-specific patterns

## [0.1.0] - 2026-01-01

### Added

- Initial release of aware-kernel
- `AwareKernelEstimator` sklearn-compatible API
- Refresh-aware hybrid continuous-discrete low-rank kernel learning
- Global Nyström basis with soft-truncated whitening
- Local corrective features with residual-aware anchor sampling
- Residual orthogonalization (local in global nullspace)
- Feature calibration and fusion (logistic gate)
- Refresh controller (drift-aware with cooldown, warmup, hysteresis, budget)
- Two memory modes: cached O(nm) and streamed O(m^2)
- Ridge regression solver (direct Cholesky and iterative PCG)
- Numerical stabilization (eigenvalue clipping, soft truncation, Cholesky with jitter)
- Training loop with objectives, callbacks, and outer-loop optimizer
- Inference with mean and variance prediction
- Evaluation framework with synthetic datasets, baselines, and metrics
- Comprehensive test suite (unit, numerical, integration) with ~91% coverage
- Design document (docs/design.md)
- Real-world evaluation examples (examples/eval_real_world/)
