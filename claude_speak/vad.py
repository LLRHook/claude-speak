"""
Voice Activity Detection using Silero VAD (ONNX).

Uses the Silero VAD v5 ONNX model to detect speech in audio chunks.
The model is auto-downloaded on first use to ~/.claude-speak/models/.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort

from .models import MODEL_REGISTRY, MODELS_DIR, download_model

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

# ---------------------------------------------------------------------------
# VAD class
# ---------------------------------------------------------------------------


class SileroVAD:
    """Voice Activity Detection using Silero VAD (ONNX).

    The Silero VAD model processes 512-sample chunks at 16 kHz (32 ms each)
    and returns a speech probability between 0 and 1. Internal hidden states
    (h and c tensors) are maintained across calls for context, and should be
    reset between separate utterances.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate

        # Internal ONNX state tensors: shape [2, 1, 64], float32
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

        # Load ONNX model
        self._session = self._load_model()

    def _load_model(self):
        """Load the Silero VAD ONNX model, downloading if needed."""
        model_info = MODEL_REGISTRY.get("silero_vad.onnx")
        if model_info is None:
            raise RuntimeError(
                "silero_vad.onnx not found in MODEL_REGISTRY. "
                "Ensure the model entry has been added to claude_speak/models.py."
            )

        model_path = MODELS_DIR / model_info.filename

        # Auto-download if missing
        if not model_path.exists():
            log.info("Silero VAD model not found, downloading...")
            download_model(model_info, MODELS_DIR)

        log.info("Loading Silero VAD model from %s", model_path)
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        return session

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Check if audio chunk contains speech.

        Args:
            audio_chunk: Audio samples as float32 ndarray. Should be 512 samples
                at 16 kHz (32 ms). Values should be in [-1.0, 1.0] range.

        Returns:
            True if the speech probability exceeds the threshold.
        """
        # Ensure correct shape: flatten to 1D then reshape to [1, N]
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()
        audio_chunk = audio_chunk.astype(np.float32)
        input_audio = audio_chunk.reshape(1, -1)

        # Sample rate as int64 tensor
        sr = np.array(self.sample_rate, dtype=np.int64)

        # Run inference
        ort_inputs = {
            "input": input_audio,
            "sr": sr,
            "h": self._h,
            "c": self._c,
        }

        ort_outputs = self._session.run(None, ort_inputs)

        # Outputs: [probability, h_out, c_out]
        probability = ort_outputs[0].item()
        self._h = ort_outputs[1]
        self._c = ort_outputs[2]

        return probability > self.threshold

    def reset(self) -> None:
        """Reset internal state between utterances.

        Clears the hidden state tensors so the next call to is_speech()
        starts fresh without context from previous audio.
        """
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_vad_instance: SileroVAD | None = None


def get_vad(threshold: float = 0.5) -> SileroVAD:
    """Get or create a singleton VAD instance.

    Args:
        threshold: Speech probability threshold (0.0 to 1.0).
            Only used when creating a new instance.

    Returns:
        The singleton SileroVAD instance.
    """
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = SileroVAD(threshold=threshold)
    return _vad_instance
