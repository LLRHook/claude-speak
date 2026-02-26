"""
Piper TTS backend for claude-speak.

Uses the piper-tts Python package to synthesize speech from ONNX voice models.
Piper is a fast, local neural text-to-speech engine built on VITS.

Voice models are downloaded from HuggingFace (rhasspy/piper-voices) and cached
in ~/.claude-speak/models/piper/.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .config import Config
from .models import (
    PIPER_MODELS_DIR,
    PIPER_VOICES,
    download_piper_voice,
    list_downloaded_piper_voices,
)
from .tts_base import TTSBackend

logger = logging.getLogger(__name__)


class PiperBackend(TTSBackend):
    """Piper TTS backend using piper-tts ONNX models.

    Handles model downloading, voice resolution, and audio generation.
    Piper generates complete audio for each sentence, so each yield from
    generate() corresponds to one sentence's worth of audio.
    """

    def __init__(self, config: Config):
        self.config = config
        self._voice = None  # PiperVoice instance
        self._model_path: Path | None = None
        self._sample_rate: int = 22050  # Piper default; updated from config on load

    def load(self) -> None:
        """Load the Piper voice model.

        Downloads the model if not already cached locally. The voice name
        is read from ``config.tts.voice`` (e.g. "en_US-lessac-medium").
        """
        try:
            from piper import PiperVoice
        except ImportError:
            raise ImportError(
                "piper-tts is not installed. "
                "Install it with: pip install 'claude-speak[piper]'"
            )

        voice_name = self.config.tts.voice

        # Try to resolve the model file path
        model_path = self._resolve_model_path(voice_name)

        if model_path is None or not model_path.exists():
            # Auto-download if it's a known voice
            if voice_name in PIPER_VOICES:
                logger.info("Downloading Piper voice model: %s", voice_name)
                model_path = download_piper_voice(voice_name)
            else:
                raise FileNotFoundError(
                    f"Piper model not found for voice {voice_name!r}. "
                    f"Known voices: {list(PIPER_VOICES)}. "
                    f"Or place a custom .onnx + .onnx.json in {PIPER_MODELS_DIR}"
                )

        logger.info("Loading Piper model from %s", model_path)
        self._voice = PiperVoice.load(str(model_path))
        self._model_path = model_path

        # Read sample rate from the loaded voice config
        if hasattr(self._voice, "config") and hasattr(self._voice.config, "sample_rate"):
            self._sample_rate = self._voice.config.sample_rate
        logger.info("Piper voice loaded: %s (sample_rate=%d)", voice_name, self._sample_rate)

    async def generate(
        self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us"
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Generate audio segments from text.

        Piper synthesizes complete audio per sentence. The *speed* parameter
        is converted to Piper's ``length_scale`` (inverse relationship:
        speed 2.0 -> length_scale 0.5, i.e. faster).

        Yields (samples_float32, sample_rate) tuples.
        """
        if self._voice is None:
            raise RuntimeError("PiperBackend.load() must be called before generate()")

        # Convert speed to length_scale (inverse: higher speed = lower length_scale)
        length_scale = 1.0 / speed if speed > 0 else 1.0

        # Synthesize using piper's synthesize method, writing to an in-memory WAV
        # then converting to float32 numpy array.
        # We use synthesize_stream_raw for streaming raw audio bytes per sentence.
        try:
            for audio_bytes in self._voice.synthesize_stream_raw(
                text,
                length_scale=length_scale,
                sentence_silence=0.2,
            ):
                # audio_bytes is 16-bit signed PCM, mono
                samples_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                samples_float32 = samples_int16.astype(np.float32) / 32768.0
                yield samples_float32, self._sample_rate
        except Exception as exc:
            logger.error("Piper synthesis failed: %s", exc)
            raise

    def list_voices(self) -> list[str]:
        """Return available voice names.

        Includes both downloaded models and known (downloadable) voices.
        """
        downloaded = list_downloaded_piper_voices()
        known = list(PIPER_VOICES.keys())
        # Merge: downloaded first, then known voices not yet downloaded
        all_voices = list(downloaded)
        for v in known:
            if v not in all_voices:
                all_voices.append(v)
        return all_voices

    def is_loaded(self) -> bool:
        """Check if the Piper voice model is loaded and ready."""
        return self._voice is not None

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable engine name."""
        return "piper"

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _resolve_model_path(voice_name: str) -> Path | None:
        """Resolve a voice name to its local .onnx model file path.

        Checks:
        1. If voice_name is already an absolute path to an existing .onnx file.
        2. If the voice exists in the Piper models cache directory.
        """
        # Direct path?
        candidate = Path(voice_name)
        if candidate.is_absolute() and candidate.exists() and candidate.suffix == ".onnx":
            return candidate

        # Check the cache directory
        cached = PIPER_MODELS_DIR / f"{voice_name}.onnx"
        if cached.exists():
            return cached

        return None

    @staticmethod
    def speed_to_length_scale(speed: float) -> float:
        """Convert a speed multiplier to Piper's length_scale parameter.

        Piper uses length_scale where lower values = faster speech.
        speed=1.0 -> length_scale=1.0 (normal)
        speed=2.0 -> length_scale=0.5 (2x faster)
        speed=0.5 -> length_scale=2.0 (2x slower)
        """
        if speed <= 0:
            return 1.0
        return 1.0 / speed
