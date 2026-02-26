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

from .config import Config, InputConfig, WakeWordConfig, load_config, TOGGLE_FILE
from .voice_input import voice_input_cycle, SuperwhisperError
from .wakeword import WakeWordListener
from . import queue as Q

logger = logging.getLogger(__name__)


class VoiceController:
    """Orchestrates wake word -> voice input -> auto-submit pipeline.

    Usage::

        config = load_config()
        controller = VoiceController(config)
        controller.start()
        # ...later...
        controller.stop()

    On wake word detection:
      1. Triggers Superwhisper to start recording.
      2. Waits for transcription to complete.
      3. Auto-submits the transcribed text to Claude Code (Enter key).

    On stop phrase detection:
      1. Clears the TTS queue.
      2. Optionally stops current playback (via tts_stop_callback).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        tts_stop_callback: Optional[callable] = None,
    ) -> None:
        """Initialize the voice controller.

        Args:
            config: Full application config. If None, loads from disk.
            tts_stop_callback: Optional callback to stop current TTS playback.
                Called (with no arguments) when a stop phrase is detected.
                This allows the controller to stop the TTSEngine without
                importing or depending on it directly.
        """
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
        """Start the voice controller.

        Initializes the wake word listener (if enabled) and registers
        callbacks for wake words and stop phrases.

        Returns:
            True if at least one component started successfully.
        """
        if self._running:
            logger.warning("VoiceController is already running")
            return False

        self._running = True
        started_something = False

        # Start wake word listener
        if self._config.wakeword.enabled:
            started_something = self._start_wakeword()
        else:
            logger.info(
                "Wake word detection is disabled. "
                "Enable it in claude-speak.toml [wakeword] enabled = true"
            )

        if started_something:
            logger.info("VoiceController started")
        else:
            logger.info(
                "VoiceController initialized but no active listeners. "
                "Voice input can still be triggered manually via trigger_voice_input()."
            )

        return True

    def stop(self) -> None:
        """Stop the voice controller and all sub-components."""
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
        """True if a voice input cycle is currently in progress."""
        return self._voice_input_active

    def trigger_voice_input(self) -> bool:
        """Manually trigger a voice input cycle (no wake word needed).

        This can be called from the CLI or a keybinding to start voice
        input without the wake word listener.

        Returns:
            True if the cycle completed successfully.
        """
        return self._handle_wake()

    # ------------------------------------------------------------------
    # Stop-phrase handling
    # ------------------------------------------------------------------

    def handle_stop(self, phrase: str = "") -> None:
        """Handle a stop command — clear queue and stop playback.

        Can be called directly or registered as a stop-phrase callback.

        Args:
            phrase: The stop phrase that was detected (for logging).
        """
        logger.info("Stop command received: '%s'", phrase)

        # Clear the TTS queue
        Q.clear()
        logger.info("TTS queue cleared")

        # Stop current playback via callback
        if self._tts_stop_callback is not None:
            try:
                self._tts_stop_callback()
                logger.info("TTS playback stopped")
            except Exception as e:
                logger.error("Error stopping TTS playback: %s", e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_wakeword(self) -> bool:
        """Initialize and start the wake word listener."""
        self._wakeword_listener = WakeWordListener(self._config.wakeword)

        # Register callbacks
        self._wakeword_listener.on_wake(self._on_wake_word)
        self._wakeword_listener.on_stop_phrase(self.handle_stop)

        return self._wakeword_listener.start()

    def _on_wake_word(self) -> None:
        """Callback invoked when the wake word is detected.

        Runs the voice input cycle in a separate thread to avoid blocking
        the wake word listener's audio processing loop.
        """
        if not self._running:
            return

        if self._voice_input_active:
            logger.debug("Voice input already active, ignoring wake word")
            return

        thread = threading.Thread(
            target=self._handle_wake,
            name="voice-input-cycle",
            daemon=True,
        )
        thread.start()

    def _handle_wake(self) -> bool:
        """Execute the full voice input cycle with proper locking.

        Pauses the wake word listener during recording to avoid mic conflicts
        with Superwhisper, then resumes it after.

        Returns:
            True if the cycle completed successfully.
        """
        if not self._input_lock.acquire(blocking=False):
            logger.debug("Voice input lock held, skipping")
            return False

        try:
            self._voice_input_active = True
            logger.info("Starting voice input cycle")

            # Pause wake word listener to release the microphone
            if self._wakeword_listener and self._wakeword_listener.is_running:
                logger.debug("Pausing wake word listener for voice input")
                self._wakeword_listener.stop()

            success = voice_input_cycle(
                config=self._config.input,
            )

            if success:
                logger.info("Voice input cycle completed")
            else:
                logger.warning("Voice input cycle did not complete")

            return success

        except SuperwhisperError as e:
            logger.error("Voice input error: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error in voice input cycle: %s", e)
            return False
        finally:
            self._voice_input_active = False
            self._input_lock.release()
            # Resume wake word listener
            if self._running and self._config.wakeword.enabled:
                logger.debug("Resuming wake word listener")
                self._start_wakeword()
