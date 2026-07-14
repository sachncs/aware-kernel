# Contributing to AwareKernel

Thank you for your interest in contributing to AwareKernel! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Naming](#branch-naming)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Running Tests](#running-tests)
- [Documentation](#documentation)

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/aware-kernel.git
   cd aware-kernel
   ```
3. **Add** the upstream remote:
   ```bash
   git remote add upstream https://github.com/sachncs/aware-kernel.git
   ```
4. **Install** development dependencies:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Setup

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the test suite to verify everything works
pytest tests/ -v -p no:asyncio
```

## Branch Naming

Use descriptive branch names with the following prefixes:

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Adding or updating tests |
| `chore/` | Maintenance tasks |

Example: `feat/add-gpu-solver`, `fix/calibration-edge-case`

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/). Use the format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, missing semi-colons, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks (dependencies, CI, etc.) |

### Examples

```
feat(solver): add CuPy GPU solver backend

fix(refresh): correct drift threshold comparison

docs(readme): update installation instructions

refactor(memory): simplify streamed accumulator interface

test(fusion): add calibration stability edge cases

chore(deps): bump numpy to >=1.25
```

## Pull Request Process

1. **Create** a feature branch from `master`.
2. **Make** your changes following the coding standards below.
3. **Write** or update tests for your changes.
4. **Update** documentation if needed.
5. **Ensure** all checks pass:
   ```bash
   ruff check .
   black --check .
   mypy --strict aware_kernel/
   pytest tests/ -v -p no:asyncio
   ```
6. **Submit** a pull request with a clear description.
7. **Address** review feedback promptly.

### PR Guidelines

- Keep PRs focused on a single change.
- Write clear, descriptive commit messages.
- Add screenshots or plots for visual changes.
- Reference related issues with `Closes #<number>`.

## Coding Standards

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) enforced by `ruff` and `black`.
- Line length: 88 characters (enforced by `black`).
- Use type hints for all public functions.
- Use Google-style docstrings.

### Type Hints

```python
def compute_drift(R: np.ndarray, R_prev: np.ndarray) -> float:
    """Compute relative Frobenius norm drift."""
    ...
```

### Docstrings

```python
def function_name(param1: type, param2: type) -> return_type:
    """Short description of the function.

    Longer description if needed, explaining the algorithm or approach.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When something is wrong with the inputs.
    """
```

### Architecture Principles

- **Immutability**: Use frozen dataclasses with `copy_with` methods for state.
- **Protocols**: Define interfaces as `typing.Protocol` rather than ABCs.
- **Numerical stability**: Always check conditioning and use jitter fallbacks.
- **Modularity**: Keep modules decoupled; prefer composition over inheritance.

## Running Tests

```bash
# Full test suite
pytest tests/ -v -p no:asyncio

# Unit tests only
pytest tests/unit/ -v -p no:asyncio

# Numerical invariant tests
pytest tests/numerical/ -v -p no:asyncio

# Integration tests
pytest tests/integration/ -v -p no:asyncio

# With coverage
pytest tests/ --cov=aware_kernel --cov-report=term-missing

# Specific test file
pytest tests/unit/test_solver.py -v
```

## Linting and Formatting

```bash
# Lint
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Format
black .

# Check formatting
black --check .

# Type check
mypy --strict aware_kernel/
```

## Documentation

- Update `docs/` for architectural or design changes.
- Add docstrings to all public functions and classes.
- Update `CHANGELOG.md` for user-facing changes.
- Keep `README.md` examples working and up to date.

## Questions?

Open an issue with the `question` label if you have any questions about contributing.
