"""
TTS engine — Kokoro model management, streaming audio, device routing.
Uses a persistent output stream for gapless playback between chunks.
Pre-generates upcoming chunks to eliminate dead air between sentences.
"""

import os
import time
import numpy as np
from pathlib import Path

from .config import Config

PERF_LOG = os.environ.get("CLAUDE_SPEAK_PERF", "").lower() in ("1", "true", "yes")

_DEVICE_RESOLVE_INTERVAL = 30  # seconds between audio device re-checks


class TTSEngine:
    """Loads Kokoro once, streams audio to the correct device with no gaps."""

    def __init__(self, config: Config):
        self.config = config
        self.kokoro = None
        self._output_device = None
        self._stream = None
        self._sample_rate = None
        self._last_device_resolve = 0.0

    def load(self):
        """Load model and resolve audio device."""
        from kokoro_onnx import Kokoro

        model_path = self.config.tts.model_path
        voices_path = self.config.tts.voices_path

        print(f"[tts] Loading model from {Path(model_path).name}...", flush=True)
        self.kokoro = Kokoro(model_path, voices_path)
        self._resolve_device()
        print(
            f"[tts] Ready. Voice: {self.config.tts.voice}, "
            f"Speed: {self.config.tts.speed}, "
            f"Device: {self._device_name()}",
            flush=True,
        )

    def _resolve_device(self):
        """Pick the output audio device."""
        import sounddevice as sd

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
        import sounddevice as sd
        try:
            info = sd.query_devices(self._output_device)
            return info["name"]
        except Exception:
            return f"device {self._output_device}"

    def _ensure_stream(self, sample_rate: int):
        """Create or reuse a persistent output stream for gapless playback."""
        import sounddevice as sd

        # Reuse if device and sample rate haven't changed
        if (self._stream is not None
                and self._stream.active
                and self._sample_rate == sample_rate):
            return

        # Close old stream
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass

        self._sample_rate = sample_rate
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=self._output_device,
            blocksize=0,  # let sounddevice pick optimal size
        )
        self._stream.start()

    def _write_samples(self, samples: np.ndarray):
        """Write audio samples to the persistent stream. Blocks until done."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        # Write in chunks to keep the stream fed smoothly
        chunk_size = 4096
        offset = 0
        while offset < len(samples):
            end = min(offset + chunk_size, len(samples))
            self._stream.write(samples[offset:end])
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
            voice=self.config.tts.voice,
            speed=self.config.tts.speed,
            lang="en-us",
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
            voice=self.config.tts.voice,
            speed=self.config.tts.speed,
            lang="en-us",
        )

        first = True
        async for samples, sample_rate in stream:
            if first:
                if PERF_LOG:
                    print(f"[perf] tts first-segment: {(time.monotonic() - t0)*1000:.0f}ms", flush=True)
                first = False
            self._ensure_stream(sample_rate)
            self._write_samples(samples)

    def drain(self):
        """Wait for all buffered audio to finish playing."""
        if self._stream is not None and self._stream.active:
            import time
            # Write a small silence buffer to flush the stream
            silence = np.zeros((self._sample_rate or 24000, 1), dtype="float32")
            try:
                self._stream.write(silence[:1024])
            except Exception:
                pass

    def stop(self):
        """Stop playback immediately."""
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
