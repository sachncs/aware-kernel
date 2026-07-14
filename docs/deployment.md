# Deployment

This document covers how to build, test, and publish aware-kernel.

## Local Development

```bash
# Clone and install
git clone https://github.com/sachncs/aware-kernel.git
cd aware-kernel
pip install -e ".[dev]"

# Run tests
pytest tests/ -v -p no:asyncio

# Run linters
ruff check .
black --check .
mypy --strict aware_kernel/
```

## Building Distribution

```bash
# Install build tools
pip install build twine

# Build source distribution and wheel
python -m build

# Verify the build
twine check dist/*
```

This creates:

```
dist/
├── aware_kernel-0.1.0-py3-none-any.whl
└── aware_kernel-0.1.0.tar.gz
```

## Publishing to PyPI

### Test PyPI (recommended first)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ aware-kernel
```

### Production PyPI

```bash
# Upload to PyPI
twine upload dist/*
```

### Authentication

Use a PyPI API token:

```bash
# Set up ~/.pypirc or use environment variable
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-api-token-here
```

## Release Checklist

1. Update version in:
   - `aware_kernel/_version.py`
   - `pyproject.toml`
2. Update `CHANGELOG.md` with release date
3. Create a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release 0.1.0"
   git push origin v0.1.0
   ```
4. Build and upload to Test PyPI
5. Verify test installation
6. Upload to production PyPI
7. Create a GitHub Release with release notes

## Versioning

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

The version is tracked in `aware_kernel/_version.py` and referenced by `pyproject.toml`.

## Continuous Publishing

To automate releases via GitHub Actions, add a release workflow:

```yaml
name: Release

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install build twine
      - run: python -m build
      - run: twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```
