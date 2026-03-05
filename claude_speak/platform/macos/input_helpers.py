"""macOS input helpers — clipboard, paste, enter, Superwhisper detection."""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_clipboard() -> str:
    """Read the current macOS pasteboard contents via pbpaste."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return result.stdout
    except Exception:
        return ""


def set_clipboard(text: str) -> bool:
    """Write text to the macOS pasteboard via pbcopy.

    Returns True on success, False on error.
    """
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            timeout=2.0,
        )
        return True
    except Exception as e:
        logger.error("pbcopy failed: %s", e)
        return False


def paste_at_cursor() -> None:
    """Press Cmd+V via osascript to paste clipboard contents at cursor."""
    script = (
        'tell application "System Events" to keystroke "v" using {command down}'
    )
    subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )


def press_enter() -> None:
    """Press Enter via osascript to submit text."""
    script = 'tell application "System Events" to keystroke return'
    subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=True,
    )


def is_superwhisper_running() -> bool:
    """Check whether the Superwhisper app process is currently running."""
    try:
        result = subprocess.run(
            ["pgrep", "-xiq", "superwhisper"],
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False
