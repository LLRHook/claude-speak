"""ElevenLabs cloud TTS backend.

Streams text-to-speech via the ElevenLabs API, yielding numpy audio chunks
compatible with the TTSBackend interface.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from .config import Config
from .tts_base import TTSBackend

logger = logging.getLogger(__name__)


def _mask_api_key(key: str) -> str:
    """Mask an API key for safe logging, showing only the last 4 characters."""
    if not key or len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


def _sanitize_exception(exc: Exception, api_key: str) -> str:
    """Return a string representation of *exc* with the API key redacted."""
    msg = str(exc)
    if api_key and api_key in msg:
        msg = msg.replace(api_key, _mask_api_key(api_key))
    return msg


# PCM format at 24 kHz — raw signed 16-bit little-endian samples.
# This avoids MP3 decoding overhead and gives us clean numpy conversion.
_OUTPUT_FORMAT = "pcm_24000"
_SAMPLE_RATE = 24000

# Default ElevenLabs model — fast, multilingual, good quality.
_DEFAULT_MODEL = "eleven_flash_v2_5"


def _resolve_api_key(config: Config) -> str:
    """Resolve the ElevenLabs API key from multiple sources.

    Priority order:
        1. ELEVENLABS_API_KEY environment variable
        2. config.tts.elevenlabs_api_key (from claude-speak.toml [tts] section)
        3. ~/.claude-speak/config.toml [elevenlabs] api_key field

    Returns the API key string, or empty string if none found.
    """
    # 1. Environment variable (highest priority)
    env_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if env_key:
        return env_key

    # 2. Config field (from claude-speak.toml)
    if config.tts.elevenlabs_api_key:
        return config.tts.elevenlabs_api_key

    # 3. User-level config file
    user_config = Path.home() / ".claude-speak" / "config.toml"
    if user_config.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]

            with open(user_config, "rb") as f:
                data = tomllib.load(f)
            file_key = data.get("elevenlabs", {}).get("api_key", "")
            if file_key:
                return file_key
        except Exception as exc:
            logger.debug("Failed to read %s: %s", user_config, exc)

    return ""


def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw PCM signed-16-bit little-endian bytes to float32 numpy array."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    # Normalize int16 range [-32768, 32767] to float32 range [-1.0, 1.0]
    samples /= 32768.0
    return samples


class ElevenLabsBackend(TTSBackend):
    """ElevenLabs cloud TTS backend.

    Uses the ``elevenlabs`` Python SDK to stream speech synthesis from
    the ElevenLabs API.  Audio is returned as PCM float32 numpy arrays
    at 24 kHz, matching the interface expected by TTSEngine.
    """

    def __init__(self, config: Config):
        self._config = config
        self._client = None  # elevenlabs.ElevenLabs sync client
        self._async_client = None  # elevenlabs.client.AsyncElevenLabs
        self._api_key: str = ""
        self._voices_cache: list[str] | None = None

    # -- TTSBackend interface -------------------------------------------------

    def load(self) -> None:
        """Initialize the ElevenLabs client and validate the API key.

        Raises RuntimeError if no API key is found.
        """
        self._api_key = _resolve_api_key(self._config)
        if not self._api_key:
            raise RuntimeError(
                "ElevenLabs API key not found. Provide it via:\n"
                "  1. ELEVENLABS_API_KEY environment variable\n"
                "  2. tts.elevenlabs_api_key in claude-speak.toml\n"
                "  3. [elevenlabs] api_key in ~/.claude-speak/config.toml"
            )

        try:
            from elevenlabs import ElevenLabs
            from elevenlabs.client import AsyncElevenLabs
        except ImportError:
            raise RuntimeError(
                "elevenlabs package is not installed. "
                "Install it with: pip install elevenlabs"
            )

        self._client = ElevenLabs(api_key=self._api_key)
        self._async_client = AsyncElevenLabs(api_key=self._api_key)
        logger.info("ElevenLabs client initialized (key=%s)", _mask_api_key(self._api_key))

    async def generate(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Stream audio from the ElevenLabs API.

        Yields (samples, sample_rate) tuples where *samples* is a float32
        numpy array and *sample_rate* is 24000 Hz (PCM output format).

        Uses the sync client's streaming endpoint, which returns an iterator
        of audio byte chunks.  Each chunk is converted from PCM int16 to
        float32 on the fly.
        """
        if self._client is None:
            raise RuntimeError("ElevenLabs backend not loaded. Call load() first.")

        from elevenlabs import VoiceSettings

        # Build voice settings — map speed to the SDK's speed parameter
        voice_settings = VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            speed=speed,
        )

        try:
            # Use the sync streaming API (returns an iterator of bytes chunks).
            # We wrap this in an async generator so it fits the TTSBackend interface.
            audio_stream = self._client.text_to_speech.stream(
                voice_id=voice,
                text=text,
                model_id=_DEFAULT_MODEL,
                output_format=_OUTPUT_FORMAT,
                voice_settings=voice_settings,
            )

            pcm_buffer = b""
            # Each chunk from the stream is raw bytes; accumulate into
            # reasonable-sized audio segments before yielding.
            min_chunk_bytes = _SAMPLE_RATE * 2 * 1  # ~1 second of 16-bit PCM at 24kHz
            for chunk in audio_stream:
                if isinstance(chunk, bytes) and len(chunk) > 0:
                    pcm_buffer += chunk
                    if len(pcm_buffer) >= min_chunk_bytes:
                        # Ensure even number of bytes (int16 = 2 bytes per sample)
                        usable = (len(pcm_buffer) // 2) * 2
                        samples = _pcm_bytes_to_float32(pcm_buffer[:usable])
                        pcm_buffer = pcm_buffer[usable:]
                        yield (samples, _SAMPLE_RATE)

            # Flush remaining buffer
            if len(pcm_buffer) >= 2:
                usable = (len(pcm_buffer) // 2) * 2
                samples = _pcm_bytes_to_float32(pcm_buffer[:usable])
                yield (samples, _SAMPLE_RATE)

        except Exception as exc:
            logger.warning("ElevenLabs API error: %s", _sanitize_exception(exc, self._api_key))
            raise

    def list_voices(self) -> list[str]:
        """Fetch available voices from the ElevenLabs API.

        Returns a list of ``"name (voice_id)"`` strings.  Results are cached
        after the first successful fetch.
        """
        if self._voices_cache is not None:
            return list(self._voices_cache)

        if self._client is None:
            raise RuntimeError("ElevenLabs backend not loaded. Call load() first.")

        try:
            response = self._client.voices.get_all()
            self._voices_cache = [
                f"{v.name} ({v.voice_id})" for v in response.voices
            ]
            return list(self._voices_cache)
        except Exception as exc:
            logger.warning("Failed to fetch ElevenLabs voices: %s", _sanitize_exception(exc, self._api_key))
            raise

    def is_loaded(self) -> bool:
        """Check if the ElevenLabs client is initialized."""
        return self._client is not None

    @property
    def name(self) -> str:  # noqa: D401
        """Human-readable engine name."""
        return "elevenlabs"
