"""Cross-platform input helpers — dispatches to platform-specific implementations."""
import sys

if sys.platform == "darwin":
    from .macos.input_helpers import (
        get_clipboard,
        is_superwhisper_running,
        paste_at_cursor,
        press_enter,
        set_clipboard,
    )
elif sys.platform == "win32":
    from .windows.input_helpers import (
        get_clipboard,
        is_superwhisper_running,
        paste_at_cursor,
        press_enter,
        set_clipboard,
    )
else:
    # Stubs for unsupported platforms
    def get_clipboard() -> str: return ""
    def set_clipboard(text: str) -> bool: return False
    def paste_at_cursor() -> None: pass
    def press_enter() -> None: pass
    def is_superwhisper_running() -> bool: return False

__all__ = [
    "get_clipboard",
    "is_superwhisper_running",
    "paste_at_cursor",
    "press_enter",
    "set_clipboard",
]
