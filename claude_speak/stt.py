"""
Speech-to-text interface and backend implementations.

Provides an abstract SpeechRecognizer base class and concrete backends.
Available backends:
  - MLX Whisper (Apple Silicon native via Metal GPU)
  - faster-whisper (CTranslate2, cross-platform, CUDA optional)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

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
    MODEL_MAP: ClassVar[dict[str, str]] = {
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


class FasterWhisperRecognizer(SpeechRecognizer):
    """faster-whisper backend — CTranslate2-based transcription.

    Works on Windows, Linux, and macOS. Uses CTranslate2 for efficient
    inference with int8 quantization. Supports CUDA GPU acceleration
    when available, otherwise falls back to CPU.
    """

    MODEL_MAP: ClassVar[dict[str, str]] = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v3",
    }

    def __init__(self, model: str = "base"):
        self._model_size = model
        self._model_id = self.MODEL_MAP.get(model, model)
        self._model = None  # lazy load

    @property
    def name(self) -> str:
        return f"faster-whisper ({self._model_size})"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        # Detect CUDA availability
        device = "cpu"
        compute_type = "int8"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
        except ImportError:
            pass

        logger.info("Loading faster-whisper model %s on %s (%s)", self._model_id, device, compute_type)
        self._model = WhisperModel(self._model_id, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        self._load_model()

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample if needed (faster-whisper expects 16kHz)
        if sample_rate != 16000:
            duration = len(audio) / sample_rate
            target_len = int(duration * 16000)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        segments, _info = self._model.transcribe(audio, language="en", beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        logger.debug("Transcribed (%s): '%s'", self.name, text)
        return text


def get_recognizer(backend: str = "auto", model: str = "base") -> SpeechRecognizer:
    """Factory function to get a speech recognizer.

    Args:
        backend: "mlx", "faster_whisper", or "auto" (tries mlx first, then faster-whisper)
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

    if backend in ("auto", "faster_whisper"):
        try:
            recognizer = FasterWhisperRecognizer(model=model)
            if recognizer.is_available():
                return recognizer
        except Exception as e:
            logger.warning("faster-whisper not available: %s", e)
        if backend == "faster_whisper":
            raise RuntimeError("faster-whisper requested but not available")

    raise RuntimeError(f"No STT backend available (tried: {backend})")
