"""
TTS engine — backend-agnostic audio synthesis with streaming playback.

TTSEngine handles audio device routing, stream management, and gapless playback.
Actual speech synthesis is delegated to a TTSBackend (default: KokoroBackend).
"""

import logging
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from .audio_devices import get_device_manager
from .config import Config
from .ssml import generate_silence, parse_ssml
from .tts_base import TTSBackend

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:
    logger.critical(
        "sounddevice is not installed. "
        "Install it with: pip install sounddevice "
        "(PortAudio must also be installed: brew install portaudio)"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Kokoro backend
# ---------------------------------------------------------------------------

def _get_onnx_providers() -> list[str]:
    """Return ONNX Runtime execution providers, preferring GPU acceleration.

    - macOS Apple Silicon: CoreMLExecutionProvider
    - Windows with DirectX 12 GPU: DmlExecutionProvider
    - Fallback: CPUExecutionProvider
    """
    import platform
    providers = []
    if platform.machine() == "arm64" and platform.system() == "Darwin":
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
        except ImportError:
            pass
    elif platform.system() == "Windows":
        try:
            import onnxruntime
            available = onnxruntime.get_available_providers()
            if "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")
        except ImportError:
            pass
    providers.append("CPUExecutionProvider")
    return providers


class KokoroBackend(TTSBackend):
    """Kokoro-ONNX TTS backend.

    Handles model loading, voice resolution (including blended voices),
    and streaming audio generation via kokoro_onnx.
    """

    def __init__(self, config: Config):
        self.config = config
        self._kokoro = None
        self._voice_style = None  # resolved in load() — string or numpy blend array

    def load(self) -> None:
        """Load the Kokoro ONNX model and resolve voice style.

        If model files are missing, auto-downloads them via ensure_models().
        On download failure, logs actionable manual-download instructions and re-raises.
        """
        from kokoro_onnx import Kokoro

        model_path = self.config.tts.model_path
        voices_path = self.config.tts.voices_path

        # Auto-download models if either file is missing
        if not Path(model_path).exists() or not Path(voices_path).exists():
            logger.info("Models not found locally, downloading...")
            try:
                from .models import MODELS_DIR, ensure_models
                paths = ensure_models()
                # Point config paths at the freshly downloaded files
                self.config.tts.model_path = str(paths["kokoro-v1.0.onnx"])
                self.config.tts.voices_path = str(paths["voices-v1.0.bin"])
                model_path = self.config.tts.model_path
                voices_path = self.config.tts.voices_path
            except Exception as exc:
                logger.critical(
                    "Model download failed: %s\n"
                    "       Download manually with:\n"
                    "         curl -L -o %s/kokoro-v1.0.onnx "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx\n"
                    "         curl -L -o %s/voices-v1.0.bin "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin\n"
                    "       Or run: claude-speak setup",
                    exc, MODELS_DIR, MODELS_DIR,
                )
                raise

        providers = _get_onnx_providers()
        logger.info("ONNX providers: %s", providers)
        logger.info("Loading model from %s...", Path(model_path).name)
        self._kokoro = Kokoro(model_path, voices_path)
        self._voice_style = self._resolve_voice()

    async def generate(
        self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us",
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Generate audio segments from *text* using the Kokoro streaming API.

        The *voice* parameter is accepted for interface compliance but the
        backend uses its internally resolved ``_voice_style`` (which may be a
        blended numpy array) so that voice blending works transparently.
        """
        stream = self._kokoro.create_stream(
            text,
            voice=self._voice_style,
            speed=speed,
            lang=lang,
        )
        async for samples, sample_rate in stream:
            yield samples, sample_rate

    def list_voices(self) -> list[str]:
        """Return available Kokoro voice names."""
        return list(self._kokoro.get_voices())

    def is_loaded(self) -> bool:
        """Check if the Kokoro model is loaded and ready."""
        return self._kokoro is not None

    @property
    def name(self) -> str:
        """Human-readable engine name."""
        return "kokoro"

    # -- internal helpers ---------------------------------------------------

    @property
    def kokoro(self):
        """Direct access to the underlying Kokoro instance (for legacy compat)."""
        return self._kokoro

    @kokoro.setter
    def kokoro(self, value):
        self._kokoro = value

    @property
    def voice_style(self):
        """Resolved voice style (string or numpy blend array)."""
        return self._voice_style

    @voice_style.setter
    def voice_style(self, value):
        self._voice_style = value

    def _resolve_voice(self):
        """Parse voice config — single name or blend like 'bm_george:60+bm_fable:40'.

        Returns either a string (single voice) or a numpy array (blended style).
        """
        voice_str = self.config.tts.voice
        if "+" not in voice_str:
            return voice_str

        # Parse blend: "bm_george:60+bm_fable:40"
        blend = None
        parts = []
        for part in voice_str.split("+"):
            part = part.strip()
            if ":" in part:
                name, weight = part.rsplit(":", 1)
                weight = float(weight) / 100.0
            else:
                name = part
                weight = None  # will be set to equal share later
            parts.append((name.strip(), weight))

        # Fill in equal weights for parts without explicit weight
        unweighted = [p for p in parts if p[1] is None]
        if unweighted:
            remaining = 1.0 - sum(w for _, w in parts if w is not None)
            equal_share = remaining / len(unweighted)
            parts = [(n, w if w is not None else equal_share) for n, w in parts]

        for name, weight in parts:
            style = self._kokoro.get_voice_style(name)
            blend = style * weight if blend is None else np.add(blend, style * weight)

        names = " + ".join(f"{n} ({w*100:.0f}%)" for n, w in parts)
        logger.info("Voice blend: %s", names)
        return blend


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

AVAILABLE_ENGINES = ["kokoro", "piper", "elevenlabs"]


def create_backend(engine_name: str, config: Config) -> TTSBackend:
    """Create a TTS backend by name.

    Supported engines:
      - "kokoro"     → KokoroBackend (built-in)
      - "piper"      → PiperBackend  (from tts_piper module)
      - "elevenlabs" → ElevenLabsBackend (from tts_elevenlabs module)

    Raises ValueError for unknown engine names.
    """
    name = engine_name.lower().strip()
    if name == "kokoro":
        return KokoroBackend(config)
    elif name == "piper":
        try:
            from .tts_piper import PiperBackend
        except ImportError as exc:
            raise ImportError(
                "Piper backend requires the tts_piper module. "
                "Ensure claude_speak/tts_piper.py is installed."
            ) from exc
        return PiperBackend(config)
    elif name == "elevenlabs":
        try:
            from .tts_elevenlabs import ElevenLabsBackend
        except ImportError as exc:
            raise ImportError(
                "ElevenLabs backend requires the tts_elevenlabs module. "
                "Ensure claude_speak/tts_elevenlabs.py is installed."
            ) from exc
        return ElevenLabsBackend(config)
    else:
        raise ValueError(
            f"Unknown TTS engine: {engine_name!r}. "
            f"Available engines: {', '.join(AVAILABLE_ENGINES)}"
        )


# ---------------------------------------------------------------------------
# TTSEngine — playback & device management (backend-agnostic)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Loads a TTS backend, streams audio to the correct device with no gaps."""

    def __init__(self, config: Config, backend: TTSBackend | None = None):
        self.config = config
        self._backend = backend if backend is not None else KokoroBackend(config)
        self._output_device = None
        self._stream = None
        self._sample_rate = None
        self._stream_lock = threading.Lock()  # protects _stream creation/destruction
        self._stopped = threading.Event()  # signals _write_samples to bail immediately

    # -- Legacy compatibility properties ------------------------------------
    # Existing tests and code reference engine.kokoro directly. These
    # properties proxy through to the KokoroBackend when applicable.

    @property
    def kokoro(self):
        """Legacy access to the underlying Kokoro model instance."""
        if isinstance(self._backend, KokoroBackend):
            return self._backend.kokoro
        return None

    @kokoro.setter
    def kokoro(self, value):
        if isinstance(self._backend, KokoroBackend):
            self._backend.kokoro = value

    @property
    def _voice_style(self):
        """Legacy access to resolved voice style."""
        if isinstance(self._backend, KokoroBackend):
            return self._backend.voice_style
        return None

    @_voice_style.setter
    def _voice_style(self, value):
        if isinstance(self._backend, KokoroBackend):
            self._backend.voice_style = value

    # -- Public API ---------------------------------------------------------

    def load(self):
        """Load the TTS backend and resolve audio device."""
        self._backend.load()
        dm = get_device_manager()
        self._output_device = dm.resolve_output(self.config.tts.device)
        logger.info(
            "Ready. Voice: %s, Lang: %s, Speed: %s, Device: %s",
            self.config.tts.voice, self.config.tts.lang,
            self.config.tts.speed, dm.get_device_name(self._output_device),
        )

    def swap_backend(self, backend: TTSBackend):
        """Swap to a new TTS backend. Stops current playback first.

        The old backend is replaced immediately. Current playback is stopped
        and the audio stream is closed so the next speak() call starts fresh
        with the new backend.
        """
        old_name = self._backend.name if self._backend else "none"
        # Stop playback and close the audio stream
        self.stop()
        self._backend = backend
        logger.info(
            "Backend swapped: %s → %s",
            old_name, backend.name,
        )

    def _resolve_voice(self):
        """Delegate voice resolution to the backend (legacy compat)."""
        if isinstance(self._backend, KokoroBackend):
            return self._backend._resolve_voice()
        return self.config.tts.voice

    def _ensure_stream(self, sample_rate: int):
        """Create or reuse a persistent output stream for gapless playback.

        Retries up to 3 times with exponential backoff on PortAudio errors.
        Falls back to the default output device if the configured device fails.
        """
        with self._stream_lock:
            # Before reusing an existing stream, verify the output device is still available
            if (self._stream is not None
                    and self._stream.active
                    and self._sample_rate == sample_rate):
                dm = get_device_manager()
                if self._output_device is not None and not dm.is_device_available(self._output_device):
                    logger.warning("Output device disconnected, falling back to default")
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                    dm.invalidate_cache()
                    self._output_device = dm.get_default_output()
                else:
                    self._stopped.clear()
                    return

            # Close old stream
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass

            self._stopped.clear()
            self._sample_rate = sample_rate

            # Retry with exponential backoff on PortAudio errors.
            # Re-resolve device on each retry so we pick up newly connected or
            # default devices after a disconnection (e.g. AirPods closed).
            delays = [0.1, 0.5, 1.0]
            last_err = None
            dm = get_device_manager()
            for attempt in range(len(delays) + 1):
                if attempt > 0:
                    self._output_device = dm.resolve_output(self.config.tts.device)
                device = self._output_device
                # On final retry, fall back to default device
                if attempt == len(delays):
                    device = dm.get_default_output()
                    logger.warning("Falling back to default output device (device %s)", device)
                try:
                    self._stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=device,
                        blocksize=0,  # let sounddevice pick optimal size
                    )
                    self._stream.start()
                    if attempt > 0:
                        logger.info("Audio stream recovered on device: %s", dm.get_device_name(self._output_device))
                    return
                except sd.PortAudioError as e:
                    last_err = e
                    if attempt < len(delays):
                        logger.warning(
                            "PortAudio error (attempt %d/3): %s", attempt + 1, e,
                        )
                        time.sleep(delays[attempt])
                    else:
                        # All retries exhausted (including default device fallback)
                        logger.error(
                            "PortAudio error: stream creation failed after "
                            "3 retries + default device fallback: %s", last_err,
                        )
                        self._stream = None

    def _write_samples(self, samples: np.ndarray):
        """Write audio samples to the persistent stream.

        Lock is held only briefly to grab the stream ref. The actual
        blocking write() happens outside the lock so stop() can abort
        the stream instantly without waiting for the write to finish.
        """
        vol = self.config.tts.volume
        if vol < 1.0:
            samples = samples * vol
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        chunk_size = 4096
        offset = 0
        while offset < len(samples):
            if self._stopped.is_set():
                break
            # Grab stream ref under lock (fast)
            with self._stream_lock:
                stream = self._stream
            if stream is None:
                break
            end = min(offset + chunk_size, len(samples))
            try:
                stream.write(samples[offset:end])
            except sd.PortAudioError as e:
                logger.warning("Audio write failed, device may have disconnected: %s", e)
                dm = get_device_manager()
                if self._output_device is not None and not dm.is_device_available(self._output_device):
                    # Device is gone — invalidate cache so next _ensure_stream re-resolves
                    dm.invalidate_cache()
                    self._output_device = None
                with self._stream_lock:
                    if self._stream is stream:
                        try:
                            self._stream.close()
                        except Exception:
                            pass
                        self._stream = None
                break
            except Exception:
                # Stream was aborted by stop() — clean up
                with self._stream_lock:
                    if self._stream is stream:
                        self._stream = None
                break
            offset = end

    def _maybe_resolve_device(self):
        """Re-resolve device only every N seconds (not per chunk)."""
        dm = get_device_manager()
        self._output_device = dm.maybe_resolve_output(self.config.tts.device)

    async def generate_audio(self, text: str) -> list[tuple[np.ndarray, int]]:
        """Generate audio samples without playing. Returns list of (samples, sample_rate)."""
        if not self._backend.is_loaded():
            self.load()

        result = []
        async for samples, sample_rate in self._backend.generate(
            text,
            voice=self.config.tts.voice,
            speed=self.config.tts.speed,
            lang=self.config.tts.lang,
        ):
            result.append((samples, sample_rate))
        return result

    def play_audio(self, segments: list[tuple[np.ndarray, int]]):
        """Play pre-generated audio segments."""
        self._maybe_resolve_device()
        for samples, sample_rate in segments:
            self._ensure_stream(sample_rate)
            self._write_samples(samples)

    async def speak(self, text: str):
        """Stream text through the backend and play audio seamlessly.

        If *text* contains SSML-like markup (``<pause>``, ``<slow>``, ``<fast>``,
        ``<spell>``), it is parsed into :class:`SpeechSegment` objects and each
        segment is synthesised with the appropriate speed modifier / silence.
        Plain text without any tags is handled identically to the previous
        behaviour (single pass through the backend).
        """
        if not self._backend.is_loaded():
            self.load()

        self._maybe_resolve_device()

        segments = parse_ssml(text)

        t0 = time.monotonic()
        first = True

        for seg in segments:
            if self._stopped.is_set():
                break

            # Insert silence for pause segments.
            if seg.pause_ms > 0:
                sample_rate = self._sample_rate or 24000
                silence = generate_silence(seg.pause_ms, sample_rate)
                self._ensure_stream(sample_rate)
                self._write_samples(silence)
                # A pause-only segment has no text to synthesise.
                if not seg.text:
                    continue

            # Nothing to synthesise for empty text (e.g. bare pause already handled).
            if not seg.text:
                continue

            # Compute effective speed: base config speed * segment modifier.
            effective_speed = self.config.tts.speed * seg.speed_modifier

            async for samples, sample_rate in self._backend.generate(
                seg.text,
                voice=self.config.tts.voice,
                speed=effective_speed,
                lang=self.config.tts.lang,
            ):
                if first:
                    logger.debug("tts first-segment: %.0fms", (time.monotonic() - t0) * 1000)
                    first = False
                self._ensure_stream(sample_rate)
                self._write_samples(samples)

    def stop(self):
        """Stop playback immediately (thread-safe).

        Sets _stopped event first (instant, no lock needed) so _write_samples
        bails between chunks. Then acquires lock to abort the PortAudio stream,
        which also interrupts any in-progress write() call.
        """
        self._stopped.set()
        with self._stream_lock:
            if self._stream is not None:
                try:
                    self._stream.abort()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def list_voices(self) -> list[str]:
        if not self._backend.is_loaded():
            self.load()
        return self._backend.list_voices()
