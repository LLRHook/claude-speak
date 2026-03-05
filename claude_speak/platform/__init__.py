"""Platform detection and constants for claude-speak."""
import sys

MACOS = "darwin"
WINDOWS = "win32"
LINUX = "linux"


def current_platform() -> str:
    """Return the current platform identifier."""
    return sys.platform


def is_macos() -> bool:
    return sys.platform == MACOS


def is_windows() -> bool:
    return sys.platform == WINDOWS


def is_linux() -> bool:
    return sys.platform.startswith(LINUX)
