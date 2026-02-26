"""
TTS engine — Kokoro model management, streaming audio, device routing.
Uses a persistent output stream for gapless playback between chunks.
Pre-generates upcoming chunks to eliminate dead air between sentences.
"""

import logging
import sys
import threading
import time
import numpy as np
from pathlib import Path

from .config import Config

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

_DEVICE_RESOLVE_INTERVAL = 30  # seconds between audio device re-checks


def _get_onnx_providers() -> list[str]:
    """Return ONNX Runtime execution providers, preferring CoreML on Apple Silicon."""
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
    providers.append("CPUExecutionProvider")
    return providers


class TTSEngine:
    """Loads Kokoro once, streams audio to the correct device with no gaps."""

    def __init__(self, config: Config):
        self.config = config
        self.kokoro = None
        self._output_device = None
        self._stream = None
        self._sample_rate = None
        self._last_device_resolve = 0.0
        self._stream_lock = threading.Lock()  # protects _stream creation/destruction
        self._stopped = threading.Event()  # signals _write_samples to bail immediately
        self._voice_style = None  # resolved in load() — string or numpy blend array

    def load(self):
        """Load model, resolve voice (with optional blending), and audio device.

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
                from .models import ensure_models, MODELS_DIR
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
        # kokoro-onnx's Kokoro constructor does not expose a `providers` parameter,
        # so provider selection is logged for diagnostics but cannot be passed directly.
        self.kokoro = Kokoro(model_path, voices_path)
        self._voice_style = self._resolve_voice()
        self._resolve_device()
        logger.info(
            "Ready. Voice: %s, Lang: %s, Speed: %s, Device: %s",
            self.config.tts.voice, self.config.tts.lang,
            self.config.tts.speed, self._device_name(),
        )

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
            style = self.kokoro.get_voice_style(name)
            if blend is None:
                blend = style * weight
            else:
                blend = np.add(blend, style * weight)

        names = " + ".join(f"{n} ({w*100:.0f}%)" for n, w in parts)
        logger.info("Voice blend: %s", names)
        return blend

    def _resolve_device(self):
        """Pick the output audio device."""
        device_pref = self.config.tts.device

        if device_pref and device_pref != "auto":
            try:
                self._output_device = int(device_pref)
                return
            except ValueError:
                pass
            for i, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] > 0 and device_pref.lower() in d["name"].lower():
                    self._output_device = i
                    return

        self._output_device = sd.default.device[1]

    def _device_name(self) -> str:
        try:
            info = sd.query_devices(self._output_device)
            return info["name"]
        except Exception:
            return f"device {self._output_device}"

    def _ensure_stream(self, sample_rate: int):
        """Create or reuse a persistent output stream for gapless playback.

        Retries up to 3 times with exponential backoff on PortAudio errors.
        Falls back to the default output device if the configured device fails.
        """
        with self._stream_lock:
            # Reuse if device and sample rate haven't changed
            if (self._stream is not None
                    and self._stream.active
                    and self._sample_rate == sample_rate):
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

            # Retry with exponential backoff on PortAudio errors
            delays = [0.1, 0.5, 1.0]
            last_err = None
            for attempt in range(len(delays) + 1):
                device = self._output_device
                # On final retry, fall back to default device
                if attempt == len(delays) and device != sd.default.device[1]:
                    device = sd.default.device[1]
                    logger.warning("Falling back to default output device")
                try:
                    self._stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="float32",
                        device=device,
                        blocksize=0,  # let sounddevice pick optimal size
                    )
                    self._stream.start()
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
            except Exception:
                # Stream was aborted by stop() or device error — clean up
                with self._stream_lock:
                    if self._stream is stream:  # only if nobody else replaced it
                        self._stream = None
                break
            offset = end

    def _maybe_resolve_device(self):
        """Re-resolve device only every N seconds (not per chunk)."""
        now = time.monotonic()
        if now - self._last_device_resolve >= _DEVICE_RESOLVE_INTERVAL:
            self._resolve_device()
            self._last_device_resolve = now

    async def generate_audio(self, text: str) -> list[tuple[np.ndarray, int]]:
        """Generate audio samples without playing. Returns list of (samples, sample_rate)."""
        if not self.kokoro:
            self.load()

        result = []
        stream = self.kokoro.create_stream(
            text,
            voice=self._voice_style,
            speed=self.config.tts.speed,
            lang=self.config.tts.lang,
        )
        async for samples, sample_rate in stream:
            result.append((samples, sample_rate))
        return result

    def play_audio(self, segments: list[tuple[np.ndarray, int]]):
        """Play pre-generated audio segments."""
        self._maybe_resolve_device()
        for samples, sample_rate in segments:
            self._ensure_stream(sample_rate)
            self._write_samples(samples)

    async def speak(self, text: str):
        """Stream text through Kokoro and play audio seamlessly."""
        if not self.kokoro:
            self.load()

        self._maybe_resolve_device()

        t0 = time.monotonic()
        stream = self.kokoro.create_stream(
            text,
            voice=self._voice_style,
            speed=self.config.tts.speed,
            lang=self.config.tts.lang,
        )

        first = True
        async for samples, sample_rate in stream:
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
        if not self.kokoro:
            self.load()
        return list(self.kokoro.get_voices())
