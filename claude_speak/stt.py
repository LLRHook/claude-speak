"""
Speech-to-text interface and backend implementations.

Provides an abstract SpeechRecognizer base class and concrete backends.
The default backend is MLX Whisper (Apple Silicon native via Metal GPU).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SpeechRecognizer(ABC):
    """Abstract base class for speech-to-text backends."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio samples to text.

        Args:
            audio: float32 numpy array of audio samples
            sample_rate: sample rate in Hz (default 16000)

        Returns:
            Transcribed text string
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (deps installed, model accessible)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...


class MLXWhisperRecognizer(SpeechRecognizer):
    """MLX Whisper backend — Apple Silicon native transcription.

    Uses the mlx-whisper package which runs Whisper models on Metal GPU
    via Apple's MLX framework. Models auto-download from Hugging Face
    on first use.
    """

    # Model mapping: size name -> Hugging Face model ID
    MODEL_MAP = {
        "tiny": "mlx-community/whisper-tiny",
        "base": "mlx-community/whisper-base",
        "small": "mlx-community/whisper-small",
        "medium": "mlx-community/whisper-medium",
    }

    def __init__(self, model: str = "base"):
        self._model_size = model
        self._model_id = self.MODEL_MAP.get(model, model)

    @property
    def name(self) -> str:
        return f"MLX Whisper ({self._model_size})"

    def is_available(self) -> bool:
        try:
            import mlx_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        import mlx_whisper

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed (mlx_whisper expects 16kHz)
        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # mlx_whisper.transcribe accepts numpy arrays directly
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_id,
            language="en",
        )

        text = result.get("text", "").strip()
        logger.debug("Transcribed (%s): '%s'", self.name, text)
        return text


def get_recognizer(backend: str = "auto", model: str = "base") -> SpeechRecognizer:
    """Factory function to get a speech recognizer.

    Args:
        backend: "mlx", "whisper_cpp", or "auto" (tries mlx first)
        model: model size - "tiny", "base", "small", "medium"

    Returns:
        A SpeechRecognizer instance.

    Raises:
        RuntimeError: If the requested backend is not available.
    """
    if backend in ("auto", "mlx"):
        try:
            recognizer = MLXWhisperRecognizer(model=model)
            if recognizer.is_available():
                return recognizer
        except Exception as e:
            logger.warning("MLX Whisper not available: %s", e)
        if backend == "mlx":
            raise RuntimeError("MLX Whisper requested but not available")

    raise RuntimeError(f"No STT backend available (tried: {backend})")
