"""Single source of truth for the package version string.

This module exists to enable consistent version access across the package
and to support tooling that reads the version without importing the full
package (e.g., ``python -c "from aware_kernel._version import __version__;
print(__version__)"``).
"""

__version__ = "0.1.0"
