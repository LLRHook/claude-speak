"""Abstract TTS backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import numpy as np


class TTSBackend(ABC):
    """Abstract base class for TTS engines."""

    @abstractmethod
    def load(self) -> None:
        """Load model files and initialize the engine."""
        ...

    @abstractmethod
    async def generate(
        self, text: str, voice: str, speed: float = 1.0, lang: str = "en-us",
    ) -> AsyncIterator[tuple[np.ndarray, int]]:
        """Generate audio segments from text. Yields (samples, sample_rate) tuples."""
        ...

    @abstractmethod
    def list_voices(self) -> list[str]:
        """Return available voice names."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the engine is loaded and ready."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...
