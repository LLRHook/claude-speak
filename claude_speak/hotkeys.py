"""Global keyboard shortcuts — platform dispatcher.

On macOS: uses CGEventTap via pyobjc-framework-Quartz.
On Windows/Linux: the macOS HotkeyManager is imported but degrades gracefully
when Quartz is unavailable (start() returns False).

The shortcut *parser* (parse_shortcut, check_conflicts, ParsedShortcut) and
key-code constants are pure Python and available on all platforms.
"""
from __future__ import annotations

import importlib
import logging
import sys

# Force-reload the platform submodule so that test fixtures that swap
# sys.modules["Quartz"] and then reload *this* module get a fresh
# Quartz binding inside the implementation.  Without this, the cached
# submodule keeps a stale reference to the old (or missing) Quartz mock.
_IMPL_KEY = "claude_speak.platform.macos.hotkeys"
if _IMPL_KEY in sys.modules:
    importlib.reload(sys.modules[_IMPL_KEY])

from .platform.macos.hotkeys import (  # noqa: F401, E402
    _KEY_CODES,
    _MOD_COMMAND,
    _MOD_CONTROL,
    _MOD_OPTION,
    _MOD_SHIFT,
    _MODIFIER_MAP,
    _MODIFIER_MASK,
    _QUARTZ_AVAILABLE,
    _SYSTEM_SHORTCUTS,
    HotkeyManager,
    ParsedShortcut,
    check_conflicts,
    parse_shortcut,
)

logger = logging.getLogger(__name__)
