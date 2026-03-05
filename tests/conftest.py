"""Shared fixtures and markers for the test suite."""

import sys

import pytest

# ---------------------------------------------------------------------------
# Common skip markers for platform-specific tests
# ---------------------------------------------------------------------------

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-only (AF_UNIX sockets, POSIX permissions, or macOS APIs)",
)

skip_on_non_macos = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS-only test",
)
