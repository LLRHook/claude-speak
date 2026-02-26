"""
Wake word detection using openwakeword.

Listens to the microphone for a wake word (default: "hey jarvis") and
triggers a callback when detected. Also supports stop phrases that can
silence/clear the TTS queue.

openwakeword is optional — if not installed, the listener degrades
gracefully with a warning log message.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .config import WakeWordConfig

logger = logging.getLogger(__name__)

# Audio parameters expected by openwakeword
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openwakeword's preferred frame size
_CHANNELS = 1
_DTYPE = "int16"


def _openwakeword_available() -> bool:
    """Check whether openwakeword is importable."""
    try:
        import openwakeword  # noqa: F401
        return True
    except ImportError:
        return False


def _sounddevice_available() -> bool:
    """Check whether sounddevice is importable."""
    try:
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        return False


class WakeWordListener:
    """Continuously listens to the microphone for wake words and stop phrases.

    Usage::

        listener = WakeWordListener(config)
        listener.on_wake(lambda: print("Wake word detected!"))
        listener.on_stop_phrase(lambda phrase: print(f"Stop: {phrase}"))
        listener.start()
        # ...later...
        listener.stop()

    The listener runs in a background thread and does not block.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None) -> None:
        self._config = config or WakeWordConfig()
        self._wake_callbacks: list[Callable[[], None]] = []
        self._stop_phrase_callbacks: list[Callable[[str], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._model = None
        self._stt_for_stop: Optional[object] = None  # reserved for future STT

        # Cooldown to prevent rapid-fire detections
        self._last_wake_time: float = 0.0
        self._cooldown_seconds: float = 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_wake(self, callback: Callable[[], None]) -> None:
        """Register a callback to be invoked when the wake word is detected."""
        self._wake_callbacks.append(callback)

    def on_stop_phrase(self, callback: Callable[[str], None]) -> None:
        """Register a callback for stop-phrase detection.

        The callback receives the matched stop phrase as its argument.
        """
        self._stop_phrase_callbacks.append(callback)

    def start(self) -> bool:
        """Start listening in a background thread.

        Returns:
            True if the listener started successfully, False if dependencies
            are missing or it is already running.
        """
        if self._running:
            logger.warning("WakeWordListener is already running")
            return False

        if not self._config.enabled:
            logger.info("Wake word detection is disabled in config")
            return False

        # Check dependencies
        if not _openwakeword_available():
            logger.warning(
                "openwakeword is not installed — wake word detection disabled. "
                "Install with: pip install openwakeword"
            )
            return False

        if not _sounddevice_available():
            logger.warning(
                "sounddevice is not installed — wake word detection disabled. "
                "Install with: pip install sounddevice"
            )
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="wakeword-listener",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Wake word listener starting (model=%s, sensitivity=%.2f)",
            self._config.model,
            self._config.sensitivity,
        )
        return True

    def stop(self) -> None:
        """Stop the listener and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._model = None
        logger.info("Wake word listener stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Load the openwakeword model. Returns True on success."""
        try:
            from openwakeword.model import Model as OWWModel

            model_name = self._config.model
            logger.info("Loading openwakeword model: %s", model_name)

            # openwakeword can load bundled pre-trained models by name
            # or custom .onnx files by path
            self._model = OWWModel(
                wakeword_models=[model_name],
                inference_framework="onnx",
            )
            logger.info("openwakeword model loaded successfully")
            return True

        except Exception as e:
            logger.error("Failed to load openwakeword model: %s", e)
            return False

    def _listen_loop(self) -> None:
        """Main microphone polling loop (runs in background thread).

        Model loading happens here (not in start()) so the onnxruntime
        session is created in the same thread that uses it — avoids
        segfaults on macOS with daemon threads.
        """
        import numpy as np
        import sounddevice as sd

        logger.debug("Entering listen loop")

        # Load model in this thread
        if not self._load_model():
            self._running = False
            return

        logger.info("Wake word listener ready")

        try:
            with sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype=_DTYPE,
                blocksize=_CHUNK_SAMPLES,
            ) as stream:
                while self._running:
                    audio_data, overflowed = stream.read(_CHUNK_SAMPLES)
                    if overflowed:
                        logger.debug("Audio input overflowed")

                    # audio_data shape: (chunk_samples, channels) as int16
                    # openwakeword expects a 1-D int16 numpy array
                    samples = np.squeeze(audio_data)
                    self._process_audio(samples)

        except sd.PortAudioError as e:
            logger.error("Audio device error: %s", e)
            self._running = False
        except Exception as e:
            logger.error("Unexpected error in listen loop: %s", e)
            self._running = False

        logger.debug("Exited listen loop")

    def _process_audio(self, samples) -> None:
        """Run wake word and stop phrase detection on an audio chunk."""
        if self._model is None:
            return

        # Feed audio to openwakeword
        prediction = self._model.predict(samples)

        # Check each model's score against the sensitivity threshold
        for model_name, score in prediction.items():
            if score >= self._config.sensitivity:
                now = time.monotonic()
                if now - self._last_wake_time < self._cooldown_seconds:
                    logger.debug(
                        "Wake word '%s' detected but in cooldown (%.1fs)",
                        model_name,
                        self._cooldown_seconds,
                    )
                    continue

                self._last_wake_time = now
                logger.info(
                    "Wake word detected: %s (score=%.3f)", model_name, score
                )
                # Reset model scores to prevent repeat triggers
                self._model.reset()
                self._fire_wake_callbacks()

    def _fire_wake_callbacks(self) -> None:
        """Invoke all registered wake-word callbacks."""
        for cb in self._wake_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("Wake callback error: %s", e)

    def _fire_stop_phrase_callbacks(self, phrase: str) -> None:
        """Invoke all registered stop-phrase callbacks."""
        for cb in self._stop_phrase_callbacks:
            try:
                cb(phrase)
            except Exception as e:
                logger.error("Stop phrase callback error: %s", e)

    def check_stop_phrase(self, text: str) -> Optional[str]:
        """Check transcribed text for stop phrases.

        This is intended to be called externally with transcribed text
        (e.g., from Superwhisper output or a separate STT pass) since
        openwakeword handles wake words but not arbitrary phrase detection.

        Args:
            text: The transcribed text to check.

        Returns:
            The matched stop phrase, or None if no match.
        """
        if not text:
            return None

        text_lower = text.strip().lower()
        for phrase in self._config.stop_phrases:
            # Match if the stop phrase is the entire utterance or starts it
            phrase_lower = phrase.lower()
            if text_lower == phrase_lower or text_lower.startswith(phrase_lower):
                logger.info("Stop phrase detected: '%s'", phrase)
                self._fire_stop_phrase_callbacks(phrase)
                return phrase

        return None
