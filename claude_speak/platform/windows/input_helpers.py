"""Windows input helpers — clipboard, paste, enter, Superwhisper detection."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_clipboard() -> str:
    """Read clipboard text on Windows via pyperclip."""
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        return ""


def set_clipboard(text: str) -> bool:
    """Write text to clipboard on Windows via pyperclip."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.error("pyperclip copy failed: %s", e)
        return False


def paste_at_cursor() -> None:
    """Press Ctrl+V to paste on Windows via pynput."""
    from pynput.keyboard import Controller, Key
    kb = Controller()
    kb.press(Key.ctrl)
    kb.press('v')
    kb.release('v')
    kb.release(Key.ctrl)


def press_enter() -> None:
    """Press Enter on Windows via pynput."""
    from pynput.keyboard import Controller, Key
    kb = Controller()
    kb.press(Key.enter)
    kb.release(Key.enter)


def is_superwhisper_running() -> bool:
    """Superwhisper is macOS-only, always returns False on Windows."""
    return False
