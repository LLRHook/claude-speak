"""Media key handler — platform dispatcher.

On macOS: intercepts hardware media keys via CGEventTap.
On Windows/Linux: the macOS MediaKeyHandler is imported but degrades gracefully
when Quartz is unavailable (start() returns False).
"""
from __future__ import annotations

import importlib
import sys

# Force-reload the platform submodule so that test fixtures that swap
# sys.modules["Quartz"] and then reload *this* module get a fresh
# Quartz binding inside the implementation.
_IMPL_KEY = "claude_speak.platform.macos.media_keys"
if _IMPL_KEY in sys.modules:
    importlib.reload(sys.modules[_IMPL_KEY])

from .config import PLAYING_FILE, QUEUE_DIR  # noqa: E402
from .platform.macos import media_keys as _impl  # noqa: E402
from .platform.macos.media_keys import (  # noqa: E402, F401
    KEY_STATE_DOWN,
    KEY_STATE_UP,
    MEDIA_KEY_SUBTYPE,
    NX_KEYTYPE_PLAY,
    NX_KEYTYPE_SOUND_DOWN,
    NX_KEYTYPE_SOUND_UP,
    MediaKeyHandler,
)


def _is_tts_active() -> bool:
    """Check if TTS is currently playing or has queued audio.

    This wrapper references this module's PLAYING_FILE and QUEUE_DIR so
    tests can patch them at ``claude_speak.media_keys.PLAYING_FILE``.
    """
    if PLAYING_FILE.exists():
        return True
    try:
        if QUEUE_DIR.exists():
            return any(QUEUE_DIR.iterdir())
    except OSError:
        pass
    return False


# Patch the submodule's _is_tts_active to use this module's version,
# so tests that patch claude_speak.media_keys.PLAYING_FILE affect the
# MediaKeyHandler._tap_callback path.
_impl._is_tts_active = _is_tts_active
