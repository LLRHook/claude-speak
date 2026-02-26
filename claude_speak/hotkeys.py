"""
Global keyboard shortcuts for claude-speak.

Uses macOS CGEventTap (via pyobjc-framework-Quartz) to register system-wide
hotkeys.  Degrades gracefully when pyobjc is not installed or when Accessibility
permissions have not been granted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import Quartz (pyobjc) — graceful fallback if unavailable
# ---------------------------------------------------------------------------
try:
    import Quartz  # type: ignore[import-untyped]

    _QUARTZ_AVAILABLE = True
except ImportError:
    _QUARTZ_AVAILABLE = False
    Quartz = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Modifier flag constants (mirror CGEventFlags from CoreGraphics)
# These are defined here so the parser works even without pyobjc installed.
# ---------------------------------------------------------------------------
_MOD_COMMAND = 1 << 20  # kCGEventFlagMaskCommand  0x100000
_MOD_SHIFT = 1 << 17    # kCGEventFlagMaskShift    0x020000
_MOD_CONTROL = 1 << 18  # kCGEventFlagMaskControl  0x040000
_MOD_OPTION = 1 << 19   # kCGEventFlagMaskAlternate 0x080000

_MODIFIER_MAP: dict[str, int] = {
    "cmd": _MOD_COMMAND,
    "command": _MOD_COMMAND,
    "shift": _MOD_SHIFT,
    "ctrl": _MOD_CONTROL,
    "control": _MOD_CONTROL,
    "alt": _MOD_OPTION,
    "opt": _MOD_OPTION,
    "option": _MOD_OPTION,
}

# Mask to isolate just the modifier bits we care about
_MODIFIER_MASK = _MOD_COMMAND | _MOD_SHIFT | _MOD_CONTROL | _MOD_OPTION

# ---------------------------------------------------------------------------
# Virtual key-code table (macOS virtual key codes)
# Covers the printable keys most likely used for shortcuts.
# ---------------------------------------------------------------------------
_KEY_CODES: dict[str, int] = {
    "a": 0x00, "b": 0x0B, "c": 0x08, "d": 0x02,
    "e": 0x0E, "f": 0x03, "g": 0x05, "h": 0x04,
    "i": 0x22, "j": 0x26, "k": 0x28, "l": 0x25,
    "m": 0x2E, "n": 0x2D, "o": 0x1F, "p": 0x23,
    "q": 0x0C, "r": 0x0F, "s": 0x01, "t": 0x11,
    "u": 0x20, "v": 0x09, "w": 0x0D, "x": 0x07,
    "y": 0x10, "z": 0x06,
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14,
    "4": 0x15, "5": 0x17, "6": 0x16, "7": 0x1A,
    "8": 0x1C, "9": 0x19,
    "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76,
    "f5": 0x60, "f6": 0x61, "f7": 0x62, "f8": 0x64,
    "f9": 0x65, "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
    "space": 0x31, "return": 0x24, "enter": 0x24,
    "tab": 0x30, "escape": 0x35, "esc": 0x35,
    "delete": 0x33, "backspace": 0x33,
    "up": 0x7E, "down": 0x7D, "left": 0x7B, "right": 0x7C,
    "minus": 0x1B, "-": 0x1B,
    "equal": 0x18, "=": 0x18, "+": 0x18,
    "[": 0x21, "]": 0x1E,
    ";": 0x29, "'": 0x27,
    ",": 0x2B, ".": 0x2F, "/": 0x2C, "\\": 0x2A,
    "`": 0x32,
}

# Well-known system shortcuts that are likely to conflict.
_SYSTEM_SHORTCUTS: set[tuple[int, int]] = {
    # Cmd+Shift+3 (screenshot), Cmd+Shift+4 (area screenshot)
    (_MOD_COMMAND | _MOD_SHIFT, 0x14),  # 3
    (_MOD_COMMAND | _MOD_SHIFT, 0x15),  # 4
    # Cmd+Shift+5 (screenshot bar)
    (_MOD_COMMAND | _MOD_SHIFT, 0x17),  # 5
    # Cmd+Space (Spotlight)
    (_MOD_COMMAND, 0x31),
    # Cmd+Tab (app switcher)
    (_MOD_COMMAND, 0x30),
}


# ---------------------------------------------------------------------------
# Shortcut parser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedShortcut:
    """Parsed representation of a shortcut string."""

    modifiers: int  # combined CGEventFlags modifier mask
    key_code: int   # macOS virtual key code
    raw: str        # original shortcut string for logging


def parse_shortcut(shortcut: str) -> ParsedShortcut:
    """Parse a human-readable shortcut string into modifiers + key code.

    Accepts strings like ``"cmd+shift+s"``, ``"ctrl+alt+f1"``, etc.
    Parts are separated by ``+`` and are case-insensitive.

    Raises ``ValueError`` on unrecognised tokens or missing key.
    """
    if not shortcut or not shortcut.strip():
        raise ValueError("Empty shortcut string")

    parts = [p.strip().lower() for p in shortcut.split("+")]
    if not parts:
        raise ValueError("Empty shortcut string")

    modifiers = 0
    key_name: str | None = None

    for part in parts:
        if part in _MODIFIER_MAP:
            modifiers |= _MODIFIER_MAP[part]
        elif part in _KEY_CODES:
            if key_name is not None:
                raise ValueError(
                    f"Multiple keys in shortcut '{shortcut}': "
                    f"'{key_name}' and '{part}'"
                )
            key_name = part
        else:
            raise ValueError(f"Unknown key or modifier '{part}' in shortcut '{shortcut}'")

    if key_name is None:
        raise ValueError(f"No key found in shortcut '{shortcut}' (only modifiers)")

    if modifiers == 0:
        raise ValueError(f"No modifiers in shortcut '{shortcut}' — global shortcuts require at least one modifier")

    return ParsedShortcut(modifiers=modifiers, key_code=_KEY_CODES[key_name], raw=shortcut)


def check_conflicts(shortcut: ParsedShortcut) -> bool:
    """Return True and log a warning if *shortcut* conflicts with a known system shortcut."""
    key = (shortcut.modifiers, shortcut.key_code)
    if key in _SYSTEM_SHORTCUTS:
        logger.warning(
            "Shortcut '%s' conflicts with a known macOS system shortcut — it may not work as expected.",
            shortcut.raw,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# HotkeyManager
# ---------------------------------------------------------------------------

class HotkeyManager:
    """Register and manage global keyboard shortcuts via a macOS CGEvent tap.

    Parameters
    ----------
    shortcuts : dict[str, str]
        Mapping of action names to shortcut strings, e.g.
        ``{"toggle_tts": "cmd+shift+s"}``.
    callbacks : dict[str, Callable]
        Mapping of action names to callback functions.
    """

    def __init__(
        self,
        shortcuts: dict[str, str],
        callbacks: dict[str, Callable[[], None]],
    ) -> None:
        self._shortcuts = shortcuts
        self._callbacks = callbacks
        self._parsed: dict[str, ParsedShortcut] = {}
        self._lookup: dict[tuple[int, int], str] = {}  # (mods, keycode) -> action
        self._thread: threading.Thread | None = None
        self._running = False
        self._tap = None
        self._run_loop_source = None

        # Parse all shortcuts eagerly so errors surface early
        for action, shortcut_str in shortcuts.items():
            try:
                parsed = parse_shortcut(shortcut_str)
                check_conflicts(parsed)
                self._parsed[action] = parsed
                self._lookup[(parsed.modifiers, parsed.key_code)] = action
            except ValueError as exc:
                logger.error("Invalid shortcut for '%s': %s", action, exc)

    # ------ public API ------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> bool:
        """Begin listening for global hotkeys in a background thread.

        Returns ``True`` if the event tap was created successfully.
        Returns ``False`` if pyobjc is unavailable, permissions are missing,
        or the manager is already running.
        """
        if self._running:
            logger.warning("HotkeyManager is already running.")
            return False

        if not _QUARTZ_AVAILABLE:
            logger.warning(
                "pyobjc-framework-Quartz is not installed — global hotkeys disabled. "
                "Install with: pip install pyobjc-framework-Quartz"
            )
            return False

        if not self._parsed:
            logger.warning("No valid shortcuts configured — nothing to listen for.")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="claude-speak-hotkeys",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop listening and release the event tap."""
        if not self._running:
            return
        self._running = False

        # Disable the tap so the CFRunLoop exits
        if _QUARTZ_AVAILABLE and self._tap is not None:
            try:
                Quartz.CGEventTapEnable(self._tap, False)
            except Exception:
                pass
            # Stop the run loop from the calling thread
            try:
                if self._run_loop_ref is not None:
                    Quartz.CFRunLoopStop(self._run_loop_ref)
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._tap = None
        self._run_loop_source = None
        self._run_loop_ref = None
        logger.info("HotkeyManager stopped.")

    # ------ internal ------

    def _event_callback(self, proxy, event_type, event, refcon):
        """CGEventTap callback — invoked on every key-down event."""
        # Safety: if the tap is disabled by macOS (e.g. timeout), re-enable it
        if event_type == Quartz.kCGEventTapDisabledByTimeout:
            logger.debug("Event tap timed out, re-enabling.")
            if self._tap is not None:
                Quartz.CGEventTapEnable(self._tap, True)
            return event

        if event_type == Quartz.kCGEventTapDisabledByUserInput:
            return event

        try:
            flags = Quartz.CGEventGetFlags(event)
            key_code = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode,
            )
            # Mask to only the modifier bits we care about
            mods = flags & _MODIFIER_MASK

            action = self._lookup.get((mods, key_code))
            if action is not None:
                cb = self._callbacks.get(action)
                if cb is not None:
                    logger.debug("Hotkey matched: %s (action=%s)", self._parsed[action].raw, action)
                    try:
                        cb()
                    except Exception:
                        logger.exception("Error in hotkey callback for '%s'", action)
                # Swallow the event so it doesn't propagate
                return None
        except Exception:
            logger.exception("Error in hotkey event callback")

        return event

    def _run_event_loop(self) -> None:
        """Create a CGEvent tap and run the CFRunLoop (runs in background thread)."""
        self._run_loop_ref = None
        try:
            # kCGEventKeyDown = 10
            mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)

            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionDefault,  # active tap (can swallow events)
                mask,
                self._event_callback,
                None,
            )

            if tap is None:
                logger.error(
                    "Failed to create CGEvent tap — Accessibility permission is likely not granted. "
                    "Go to System Settings > Privacy & Security > Accessibility and add this app."
                )
                self._running = False
                return

            self._tap = tap
            run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            self._run_loop_source = run_loop_source

            run_loop = Quartz.CFRunLoopGetCurrent()
            self._run_loop_ref = run_loop

            Quartz.CFRunLoopAddSource(run_loop, run_loop_source, Quartz.kCFRunLoopDefaultMode)
            Quartz.CGEventTapEnable(tap, True)

            logger.info(
                "Global hotkeys active (%d shortcut(s) registered).",
                len(self._parsed),
            )

            # CFRunLoopRun blocks until CFRunLoopStop is called (from stop())
            Quartz.CFRunLoopRun()

        except Exception:
            logger.exception("Fatal error in hotkey event loop")
        finally:
            self._running = False
            logger.debug("Hotkey event loop exited.")
