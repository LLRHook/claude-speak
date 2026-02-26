"""
Media key handler -- intercepts hardware media keys to control TTS playback.

Uses pyobjc's Quartz bindings to create a CGEvent tap that captures
NSSystemDefined events (media keys: play/pause, volume up/down).
Only intercepts keys when TTS is actively playing or has queued audio.

Requires macOS Accessibility permissions (System Settings > Privacy &
Security > Accessibility) for CGEventTapCreate to work.

Gracefully degrades if pyobjc-framework-Quartz is not installed or if
Accessibility permissions are not granted.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .config import PLAYING_FILE, QUEUE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Media key constants (from IOKit/hidsystem/ev_keymap.h)
# ---------------------------------------------------------------------------
NX_KEYTYPE_SOUND_UP = 0
NX_KEYTYPE_SOUND_DOWN = 1
NX_KEYTYPE_PLAY = 16

# NSSystemDefined event subtype for media keys
MEDIA_KEY_SUBTYPE = 8

# Key event states encoded in data1
KEY_STATE_DOWN = 0x0A
KEY_STATE_UP = 0x0B


def _is_tts_active() -> bool:
    """Check if TTS is currently playing or has queued audio."""
    if PLAYING_FILE.exists():
        return True
    try:
        # Check if there are items in the queue directory
        if QUEUE_DIR.exists():
            return any(QUEUE_DIR.iterdir())
    except OSError:
        pass
    return False


class MediaKeyHandler:
    """Intercepts hardware media keys to control TTS playback on macOS.

    Uses a CGEvent tap to listen for NSSystemDefined events (media keys).
    Only acts on media keys when TTS is active (playing or queued).

    Parameters
    ----------
    callbacks : dict[str, Callable]
        Dictionary mapping action names to callbacks:
        - ``"toggle_mute"``: Called when play/pause is pressed.
        - ``"volume_up"``: Called when volume up is pressed.
        - ``"volume_down"``: Called when volume down is pressed.

    Attributes
    ----------
    is_running : bool
        Whether the handler is currently listening for media keys.
    """

    def __init__(self, callbacks: dict[str, Callable]) -> None:
        self._callbacks = callbacks
        self._running = False
        self._thread: threading.Thread | None = None
        self._tap = None  # CGEvent tap (Mach port)
        self._run_loop_source = None
        self._run_loop = None
        self._quartz_available = False

        # Pre-check for Quartz availability
        try:
            import Quartz  # noqa: F401
            self._quartz_available = True
        except ImportError:
            logger.warning(
                "pyobjc-framework-Quartz not installed. "
                "Media key support disabled. "
                "Install with: pip install pyobjc-framework-Quartz"
            )

    @property
    def is_running(self) -> bool:
        """Whether the handler is currently listening for media keys."""
        return self._running

    def start(self) -> bool:
        """Begin listening for media keys in a background thread.

        Returns
        -------
        bool
            True if the handler started successfully, False if Quartz is
            unavailable, already running, or the event tap could not be
            created (e.g. missing Accessibility permissions).
        """
        if self._running:
            logger.warning("MediaKeyHandler is already running")
            return False

        if not self._quartz_available:
            logger.warning("Cannot start media key handler: Quartz not available")
            return False

        # Create the event tap on the main thread check (tap creation must
        # happen before we start the run loop thread so we can report errors).
        if not self._create_tap():
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop_thread,
            name="media-key-handler",
            daemon=True,
        )
        self._thread.start()
        logger.info("Media key handler started")
        return True

    def stop(self) -> None:
        """Stop listening for media keys."""
        if not self._running:
            return

        self._running = False

        # Stop the CFRunLoop to unblock the thread
        try:
            import Quartz
            if self._run_loop is not None:
                Quartz.CFRunLoopStop(self._run_loop)
        except Exception as e:
            logger.debug("Error stopping run loop: %s", e)

        # Disable and clean up the tap
        try:
            import Quartz
            if self._tap is not None:
                Quartz.CGEventTapEnable(self._tap, False)
                self._tap = None
        except Exception as e:
            logger.debug("Error disabling event tap: %s", e)

        self._run_loop_source = None
        self._run_loop = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Media key handler stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_tap(self) -> bool:
        """Create a CGEvent tap for NSSystemDefined events.

        Returns True on success, False if the tap could not be created
        (typically because Accessibility permissions are not granted).
        """
        try:
            import Quartz
            from AppKit import NSSystemDefined

            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly,
                Quartz.CGEventMaskBit(NSSystemDefined),
                self._tap_callback,
                None,
            )

            if tap is None:
                logger.warning(
                    "CGEventTapCreate returned None. "
                    "Media key handling requires Accessibility permissions. "
                    "Grant access in: System Settings > Privacy & Security > Accessibility"
                )
                return False

            run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            if run_loop_source is None:
                logger.error("Failed to create run loop source for event tap")
                return False

            self._tap = tap
            self._run_loop_source = run_loop_source
            return True

        except ImportError:
            logger.warning("Quartz framework not available for media key handling")
            return False
        except Exception as e:
            logger.error("Failed to create event tap: %s", e)
            return False

    def _run_loop_thread(self) -> None:
        """Background thread that runs the CFRunLoop for event tap delivery."""
        try:
            import Quartz

            self._run_loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(
                self._run_loop,
                self._run_loop_source,
                Quartz.kCFRunLoopDefaultMode,
            )
            Quartz.CGEventTapEnable(self._tap, True)

            # CFRunLoopRun blocks until CFRunLoopStop is called (from stop())
            Quartz.CFRunLoopRun()

        except Exception as e:
            logger.error("Media key run loop error: %s", e)
            self._running = False

    def _tap_callback(self, proxy, event_type, event, refcon):
        """CGEvent tap callback -- processes NSSystemDefined media key events.

        Parameters are dictated by the CGEventTapCreate callback signature:
        (proxy, type, event, refcon) -> event or None.

        For listen-only taps, the return value is ignored.
        """
        try:
            import Quartz
            from AppKit import NSEvent

            # Re-enable the tap if macOS disabled it (happens under heavy load)
            if event_type == Quartz.kCGEventTapDisabledByTimeout:
                logger.debug("Event tap was disabled by timeout, re-enabling")
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return event

            if event_type == Quartz.kCGEventTapDisabledByUserInput:
                logger.debug("Event tap disabled by user input, re-enabling")
                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, True)
                return event

            # Convert CGEvent to NSEvent for easier parsing
            ns_event = NSEvent.eventWithCGEvent_(event)
            if ns_event is None:
                return event

            # Only handle NSSystemDefined events with media key subtype
            if ns_event.subtype() != MEDIA_KEY_SUBTYPE:
                return event

            # Extract key info from data1:
            # Bits 16-23: key code (NX_KEYTYPE_*)
            # Bits 8-15: key state (0x0A = down, 0x0B = up)
            data1 = ns_event.data1()
            key_code = (data1 & 0xFFFF0000) >> 16
            key_state = (data1 & 0xFF00) >> 8

            # Only respond on key down
            if key_state != KEY_STATE_DOWN:
                return event

            # Only intercept when TTS is active
            if not _is_tts_active():
                return event

            self._dispatch_key(key_code)

        except Exception as e:
            # Never let exceptions escape the callback -- would crash the
            # event tap and potentially the whole event system.
            logger.debug("Error in media key callback: %s", e)

        return event

    def _dispatch_key(self, key_code: int) -> None:
        """Dispatch a media key press to the appropriate callback."""
        if key_code == NX_KEYTYPE_PLAY:
            callback = self._callbacks.get("toggle_mute")
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error("Error in toggle_mute callback: %s", e)

        elif key_code == NX_KEYTYPE_SOUND_UP:
            callback = self._callbacks.get("volume_up")
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error("Error in volume_up callback: %s", e)

        elif key_code == NX_KEYTYPE_SOUND_DOWN:
            callback = self._callbacks.get("volume_down")
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.error("Error in volume_down callback: %s", e)
