"""
Wake word detection using openwakeword.

Listens to the microphone for a wake word (default: "hey jarvis") and
triggers a callback when detected. Also supports a separate stop model
(stop.onnx) for instant TTS interruption (~80ms latency).

openwakeword is optional — if not installed, the listener degrades
gracefully with a warning log message.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .audio_devices import get_device_manager
from .config import WakeWordConfig

logger = logging.getLogger(__name__)

# Audio parameters expected by openwakeword
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openwakeword's preferred frame size
_CHANNELS = 1
_DTYPE = "int16"


def _openwakeword_available() -> bool:
    try:
        import openwakeword  # noqa: F401
        return True
    except ImportError:
        return False


def _sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        return False


class WakeWordListener:
    """Listens for wake words and stop commands via openwakeword models.

    Supports two models simultaneously:
      - Wake word model (e.g., "hey_jarvis") -> fires wake callbacks
      - Stop model (e.g., "stop.onnx") -> fires stop callbacks

    During TTS playback, swaps to the built-in mic to avoid Bluetooth
    profile switching on wireless headphones.

    Thread safety
    -------------
    This class is used from multiple threads:

      - **Main thread**: calls start(), stop(), pause(), resume(),
        use_builtin_mic(), use_default_mic(), reads is_running.
      - **wakeword-listener thread** (_listen_loop): reads _running_event,
        _paused_event, _swap_mic_event; writes _model, _builtin_mic_id.
      - **Daemon callback threads**: may invoke on_wake / on_stop_phrase
        callbacks, which in turn call pause()/resume()/use_*_mic() from
        the voice-input-cycle thread.

    Shared mutable state and synchronisation:

      _running_event  (threading.Event)
          set()  = listener is running;  clear() = stopped.
          Written by: main thread (start, stop), listener thread (on error).
          Read by:    listener thread (_listen_loop).
          Protection: Event is inherently thread-safe.

      _paused_event   (threading.Event)
          set()  = mic paused (voice input has the mic);  clear() = active.
          Written by: main thread / voice-input thread (pause, resume).
          Read by:    listener thread (_listen_loop).
          Protection: Event is inherently thread-safe.

      _swap_mic_event (threading.Event)
          set()  = use built-in mic;  clear() = use default mic.
          Written by: daemon/voice-controller thread (use_builtin_mic,
                      use_default_mic).
          Read by:    listener thread (_listen_loop).
          Protection: Event is inherently thread-safe.

      _wake_callbacks / _stop_phrase_callbacks (lists)
          Written by: main thread (on_wake, on_stop_phrase) before start().
          Read by:    listener thread (_fire_*_callbacks).
          Protection: safe because callbacks are registered before start()
                      and never mutated afterwards.

      _model (openwakeword Model instance)
          Written by: listener thread (_load_model, stop).
          Read by:    listener thread (_process_audio).
          Protection: single-writer (listener thread); main thread sets to
                      None in stop() after join().

      _last_wake_time / _last_stop_time (floats)
          Written and read only by: listener thread (_process_audio).
          Protection: single-thread access; no lock needed.
    """

    def __init__(self, config: Optional[WakeWordConfig] = None) -> None:
        self._config = config or WakeWordConfig()
        self._wake_callbacks: list[Callable[[], None]] = []
        self._stop_phrase_callbacks: list[Callable[[str], None]] = []
        self._thread: Optional[threading.Thread] = None

        # Threading events — replace plain booleans for cross-thread safety.
        # See class docstring for full thread-safety analysis.
        self._running_event = threading.Event()    # set() = running, clear() = stopped
        self._paused_event = threading.Event()     # set() = paused,  clear() = active
        self._swap_mic_event = threading.Event()   # set() = builtin, clear() = default

        self._model = None
        self._stop_model_name: Optional[str] = None  # key in prediction dict
        self._builtin_mic_id: Optional[int] = None

        # Cooldown to prevent rapid-fire detections (listener thread only)
        self._last_wake_time: float = 0.0
        self._last_stop_time: float = 0.0
        self._cooldown_seconds: float = 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_wake(self, callback: Callable[[], None]) -> None:
        self._wake_callbacks.append(callback)

    def on_stop_phrase(self, callback: Callable[[str], None]) -> None:
        self._stop_phrase_callbacks.append(callback)

    def start(self) -> bool:
        if self._running_event.is_set():
            logger.warning("WakeWordListener is already running")
            return False

        if not self._config.enabled:
            return False

        if not _openwakeword_available() or not _sounddevice_available():
            logger.warning("openwakeword or sounddevice not installed")
            return False

        self._running_event.set()
        self._paused_event.clear()  # start in active (non-paused) state
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="wakeword-listener",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Wake word listener starting (model=%s, stop_model=%s, sensitivity=%.2f)",
            self._config.model,
            self._config.stop_model or "none",
            self._config.sensitivity,
        )
        return True

    def stop(self) -> None:
        self._running_event.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._model = None
        logger.info("Wake word listener stopped")

    def pause(self) -> None:
        """Pause mic input entirely (during voice input -- Superwhisper needs the mic)."""
        self._paused_event.set()
        logger.debug("Wake word listener paused")

    def resume(self) -> None:
        """Resume mic input after pause."""
        self._paused_event.clear()
        logger.debug("Wake word listener resumed")

    def use_builtin_mic(self) -> None:
        """Swap to built-in mic (during TTS to avoid BT profile switch)."""
        self._swap_mic_event.set()

    def use_default_mic(self) -> None:
        """Swap back to default mic (after TTS finishes)."""
        self._swap_mic_event.clear()

    @property
    def is_running(self) -> bool:
        return self._running_event.is_set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        try:
            from openwakeword.model import Model as OWWModel

            models = [self._config.model]

            # Add stop model if configured
            if self._config.stop_model:
                import os
                stop_path = self._config.stop_model
                # Resolve relative paths from project root
                if not os.path.isabs(stop_path):
                    from .config import PROJECT_DIR
                    stop_path = str(PROJECT_DIR / stop_path)
                if os.path.exists(stop_path):
                    models.append(stop_path)
                    # openwakeword uses the filename stem as the model key
                    self._stop_model_name = os.path.splitext(os.path.basename(stop_path))[0]
                    logger.info("Stop model loaded: %s (key=%s)", stop_path, self._stop_model_name)
                else:
                    logger.warning("Stop model not found: %s", stop_path)

            logger.info("Loading openwakeword models: %s", models)
            self._model = OWWModel(
                wakeword_models=models,
                inference_framework="onnx",
            )
            logger.info("Models loaded: %s", list(self._model.models.keys()))
            return True

        except Exception as e:
            logger.error("Failed to load openwakeword model: %s", e)
            return False

    def _listen_loop(self) -> None:
        import numpy as np
        import sounddevice as sd

        if not self._load_model():
            self._running_event.clear()
            return

        self._builtin_mic_id = get_device_manager().find_builtin_mic()
        if self._builtin_mic_id is not None:
            logger.info("Built-in mic: device %d", self._builtin_mic_id)

        logger.info("Wake word listener ready")

        try:
            while self._running_event.is_set():
                # Paused = mic fully released (voice input has the mic).
                # Use wait() with timeout so we wake promptly on resume
                # instead of busy-polling with sleep().
                if self._paused_event.is_set():
                    time.sleep(0.1)
                    continue

                # Pick device: use built-in mic during TTS to avoid BT
                # profile switch; otherwise use the system default.
                device = None
                if self._swap_mic_event.is_set() and self._builtin_mic_id is not None:
                    device = self._builtin_mic_id

                try:
                    with sd.InputStream(
                        samplerate=_SAMPLE_RATE,
                        channels=_CHANNELS,
                        dtype=_DTYPE,
                        blocksize=_CHUNK_SAMPLES,
                        device=device,
                    ) as stream:
                        # Snapshot the current mic-swap state so we can
                        # detect when it changes and re-open the stream
                        # on the new device.
                        current_swap = self._swap_mic_event.is_set()
                        while (self._running_event.is_set()
                               and not self._paused_event.is_set()
                               and self._swap_mic_event.is_set() == current_swap):
                            audio_data, overflowed = stream.read(_CHUNK_SAMPLES)
                            if overflowed:
                                logger.debug("Audio input overflowed")
                            samples = np.squeeze(audio_data)
                            self._process_audio(samples)
                except sd.PortAudioError as e:
                    logger.error("Audio device error: %s (retrying in 1s)", e)
                    time.sleep(1)

        except Exception as e:
            logger.error("Unexpected error in listen loop: %s", e)
            self._running_event.clear()

    def _process_audio(self, samples) -> None:
        if self._model is None:
            return

        prediction = self._model.predict(samples)
        now = time.monotonic()

        # Check stop model FIRST (priority over wake word)
        if self._stop_model_name and self._stop_model_name in prediction:
            score = prediction[self._stop_model_name]
            if score >= self._config.stop_sensitivity:
                if now - self._last_stop_time >= self._cooldown_seconds:
                    self._last_stop_time = now
                    self._last_wake_time = now  # suppress wake for this frame
                    logger.info("Stop detected: %s (score=%.3f)", self._stop_model_name, score)
                    self._model.reset()
                    self._fire_stop_phrase_callbacks("stop")
                    return  # stop wins — skip wake word check

        # Check wake word models
        for model_name, score in prediction.items():
            if model_name == self._stop_model_name:
                continue  # already handled above
            if score >= self._config.sensitivity:
                if now - self._last_wake_time < self._cooldown_seconds:
                    continue
                self._last_wake_time = now
                logger.info("Wake word detected: %s (score=%.3f)", model_name, score)
                self._model.reset()
                self._fire_wake_callbacks()
                return  # only one wake per frame

    def _fire_wake_callbacks(self) -> None:
        for cb in self._wake_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("Wake callback error: %s", e)

    def _fire_stop_phrase_callbacks(self, phrase: str) -> None:
        for cb in self._stop_phrase_callbacks:
            try:
                cb(phrase)
            except Exception as e:
                logger.error("Stop phrase callback error: %s", e)
