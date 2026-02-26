"""
Voice controller — coordinates wake word detection, voice input, and TTS output.

This is the high-level orchestrator that ties together:
  - WakeWordListener: listens for "hey jarvis" (or custom wake word)
  - voice_input: triggers Superwhisper and auto-submits
  - TTS queue: clears the queue on stop phrases

The controller can be started and stopped independently of the TTS daemon.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .config import Config, load_config, MUTE_FILE, PLAYING_FILE
from .voice_input import voice_input_cycle, builtin_voice_input_cycle, SuperwhisperError
from .wakeword import WakeWordListener
from .chimes import play_ack_chime, play_error_chime
from . import queue as Q

logger = logging.getLogger(__name__)


class VoiceController:
    """Orchestrates wake word -> voice input -> auto-submit pipeline.

    Thread safety
    -------------
    This class is used from multiple threads:

      - **Main thread** (daemon): calls start(), stop(), reads is_running.
      - **wakeword-listener thread**: invokes _on_wake_word callback.
      - **voice-input-cycle thread**: spawned by _on_wake_word; calls
        _handle_wake, which acquires _input_lock.

    Shared mutable state and synchronisation:

      _running (bool)
          Written by: main thread (start, stop).
          Read by:    voice-input-cycle thread (_on_wake_word, _handle_wake).
          Protection: plain bool; single-writer pattern.  Reads from other
                      threads see a stale value at worst, causing one extra
                      cycle or a harmless skip.  Acceptable for a shutdown
                      flag.

      _voice_input_active (bool)
          Written by: voice-input-cycle thread (_handle_wake, finally block).
          Read by:    main thread (is_voice_input_active property).
          Protection: plain bool; racy but intentionally so — used only for
                      status reporting, never for control flow decisions.

      _input_lock (threading.Lock)
          Guards the voice input cycle to prevent double-triggering.
          Acquired non-blocking by _handle_wake; if held, the second caller
          exits immediately.

      _wakeword_listener (WakeWordListener | None)
          Written by: main thread (start, stop).
          Read by:    voice-input-cycle thread (_handle_wake finally block,
                      _on_wake_word).
          Protection: main thread writes None in stop() only after the
                      voice input cycle has finished (daemon serialises
                      shutdown).  The voice-input-cycle thread checks for
                      None before calling methods.

      _tts_stop_callback (callable | None)
          Written by: main thread (constructor only).
          Read by:    voice-input-cycle thread (handle_stop, _on_wake_word).
          Protection: effectively immutable after __init__; safe without locks.

    Sentinel files (MUTE_FILE, PLAYING_FILE)
    -----------------------------------------
    File-based signalling between the daemon main loop and the voice
    controller.  On POSIX, create (touch) and unlink are atomic at the
    filesystem level, so concurrent access from multiple threads or
    processes will not corrupt data.  However, the existence check
    followed by create/unlink is *not* atomic, meaning a TOCTOU race is
    theoretically possible.  In practice this only causes a harmless
    double-mute or missed toggle, which self-corrects on the next wake
    word.  All sentinel file operations are wrapped in try/except OSError
    to guard against permission errors and filesystem edge cases.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        tts_stop_callback: Optional[callable] = None,
    ) -> None:
        self._config = config or load_config()
        self._tts_stop_callback = tts_stop_callback

        self._wakeword_listener: Optional[WakeWordListener] = None
        self._running = False
        self._input_lock = threading.Lock()
        self._voice_input_active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        if self._running:
            logger.warning("VoiceController is already running")
            return False

        self._running = True
        started_something = False

        if self._config.wakeword.enabled:
            started_something = self._start_wakeword()

        if started_something:
            logger.info("VoiceController started")

        return True

    def stop(self) -> None:
        self._running = False
        if self._wakeword_listener is not None:
            self._wakeword_listener.stop()
            self._wakeword_listener = None
        logger.info("VoiceController stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_voice_input_active(self) -> bool:
        return self._voice_input_active

    def trigger_voice_input(self) -> bool:
        return self._handle_wake()

    # ------------------------------------------------------------------
    # Stop-phrase handling
    # ------------------------------------------------------------------

    def handle_stop(self, phrase: str = "") -> None:
        """Handle a stop command -- clear queue and stop playback."""
        logger.info("Stop command received: '%s'", phrase)
        Q.clear()
        # Clean up mute state to prevent deadlock.
        # POSIX: unlink is atomic; missing_ok handles the race where
        # another thread already removed the file.
        try:
            MUTE_FILE.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to remove mute sentinel: %s", e)

        if self._tts_stop_callback is not None:
            try:
                self._tts_stop_callback()
            except Exception as e:
                logger.error("Error stopping TTS playback: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_wakeword(self) -> bool:
        self._wakeword_listener = WakeWordListener(
            self._config.wakeword,
            audio_config=self._config.audio,
        )
        self._wakeword_listener.on_wake(self._on_wake_word)
        self._wakeword_listener.on_stop_phrase(self.handle_stop)
        return self._wakeword_listener.start()

    @property
    def bt_workaround_active(self) -> bool:
        """True when the BT mic workaround is engaged for this session.

        When True, the wake word listener is already using the built-in mic
        for the entire session, so callers (e.g. the daemon) can skip the
        per-TTS ``use_builtin_mic()`` / ``use_default_mic()`` calls.
        """
        if self._wakeword_listener is None:
            return False
        return self._wakeword_listener._bt_always_builtin

    def _on_wake_word(self) -> None:
        """Callback invoked when the wake word is detected.

        Context-aware:
          - TTS playing → mute (stop audio, keep position)
          - TTS muted → unmute
          - TTS idle → start voice input

        Serialization: _input_lock in _handle_wake is the real guard against
        double voice input. We don't check _voice_input_active here because
        that flag is set on a different thread and is racy.
        """
        if not self._running:
            return

        # If TTS is currently playing, toggle mute.
        # Sentinel file checks: POSIX guarantees that exists(), touch(),
        # and unlink() each execute atomically at the syscall level.
        # The exists()-then-act pattern has a benign TOCTOU race — at
        # worst we double-toggle, which self-corrects on the next wake.
        try:
            tts_playing = PLAYING_FILE.exists()
        except OSError as e:
            logger.warning("Failed to check playing sentinel: %s", e)
            tts_playing = False

        if tts_playing:
            try:
                if MUTE_FILE.exists():
                    MUTE_FILE.unlink(missing_ok=True)
                    logger.info("TTS unmuted (resume)")
                else:
                    MUTE_FILE.touch()
                    if self._tts_stop_callback is not None:
                        self._tts_stop_callback()
                    logger.info("TTS muted (paused)")
            except OSError as e:
                logger.warning("Failed to toggle mute sentinel: %s", e)
            return

        # TTS is idle → start voice input
        # _handle_wake acquires _input_lock (non-blocking) so concurrent
        # spawns are harmless — the second thread exits immediately.
        thread = threading.Thread(
            target=self._handle_wake,
            name="voice-input-cycle",
            daemon=True,
        )
        thread.start()

    def _choose_voice_input(self) -> str:
        """Decide which voice input flow to use.

        Returns:
            "superwhisper" to use the external Superwhisper app, or
            "builtin" to use the built-in mic -> VAD -> STT pipeline.
        """
        backend = self._config.input.backend
        if backend == "superwhisper":
            return "superwhisper"
        if backend == "auto":
            from .voice_input import _is_superwhisper_running
            if _is_superwhisper_running():
                return "superwhisper"
            return "builtin"
        return "builtin"  # default

    def _handle_wake(self) -> bool:
        """Execute the full voice input cycle.

        Pauses (not stops) the wake word listener during recording to
        release the mic, then resumes it after.

        Chooses between the built-in pipeline and the legacy Superwhisper
        flow based on config and whether Superwhisper is running.
        """
        if not self._input_lock.acquire(blocking=False):
            logger.debug("Voice input lock held, skipping")
            return False

        try:
            self._voice_input_active = True
            logger.info("Starting voice input cycle")

            # Pause wake word listener to release the mic (lightweight, keeps model loaded)
            if self._wakeword_listener and self._wakeword_listener.is_running:
                self._wakeword_listener.pause()

            flow = self._choose_voice_input()
            logger.info("Using %s voice input flow", flow)

            if flow == "superwhisper":
                success = voice_input_cycle(config=self._config.input)
            else:
                success = builtin_voice_input_cycle(config=self._config.input)

            if success:
                logger.info("Voice input cycle completed")
                if self._config.audio.chimes:
                    play_ack_chime(volume=self._config.audio.volume)
            else:
                logger.warning("Voice input cycle did not complete")
                if self._config.audio.chimes:
                    play_error_chime(volume=self._config.audio.volume)

            return success

        except SuperwhisperError as e:
            logger.error("Voice input error: %s", e)
            if self._config.audio.chimes:
                play_error_chime(volume=self._config.audio.volume)
            return False
        except Exception as e:
            logger.error("Unexpected error in voice input cycle: %s", e)
            return False
        finally:
            self._voice_input_active = False
            self._input_lock.release()
            # Resume wake word listener (not restart — model stays loaded)
            if self._wakeword_listener and self._running:
                self._wakeword_listener.resume()
