"""
Model management for claude-speak.
Downloads and caches Kokoro TTS model files to ~/.claude-speak/models/.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a downloadable model file."""

    name: str
    filename: str
    url: str
    sha256: str  # hex digest; empty string means "skip verification"
    size_bytes: int


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelInfo] = {
    "kokoro-v1.0.onnx": ModelInfo(
        name="kokoro-v1.0.onnx",
        filename="kokoro-v1.0.onnx",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx",
        sha256="",
        size_bytes=341_858_009,  # ~326 MB
    ),
    "voices-v1.0.bin": ModelInfo(
        name="voices-v1.0.bin",
        filename="voices-v1.0.bin",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin",
        sha256="",
        size_bytes=341_583_872,  # ~326 MB
    ),
    "silero_vad.onnx": ModelInfo(
        name="silero_vad.onnx",
        filename="silero_vad.onnx",
        url="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        sha256="",  # skip verification
        size_bytes=0,  # skip size check — small model (~2 MB), size may change between versions
    ),
}

# ---------------------------------------------------------------------------
# STT (Whisper) model registry
# ---------------------------------------------------------------------------

STT_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "tiny": {"hf_repo": "mlx-community/whisper-tiny", "size_hint": "~39 MB"},
    "base": {"hf_repo": "mlx-community/whisper-base", "size_hint": "~74 MB"},
    "small": {"hf_repo": "mlx-community/whisper-small", "size_hint": "~244 MB"},
    "medium": {"hf_repo": "mlx-community/whisper-medium", "size_hint": "~769 MB"},
}

# ---------------------------------------------------------------------------
# Default cache directory
# ---------------------------------------------------------------------------

MODELS_DIR: Path = Path.home() / ".claude-speak" / "models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Simple terminal progress callback for :func:`urllib.request.urlretrieve`."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_down = downloaded / (1 << 20)
        mb_total = total_size / (1 << 20)
        bar = f"\r  downloaded {mb_down:6.1f} / {mb_total:.1f} MB  ({pct:5.1f}%)"
    else:
        mb_down = downloaded / (1 << 20)
        bar = f"\r  downloaded {mb_down:6.1f} MB"
    sys.stderr.write(bar)
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_model(model_info: ModelInfo, dest_dir: Path) -> Path:
    """Download a single model file to *dest_dir* and return the local path.

    * Skips the download when the file already exists with the expected size.
    * Verifies the SHA-256 checksum after downloading (unless *sha256* is empty).
    * Shows a simple progress bar on *stderr*.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / model_info.filename

    # Skip if already present with correct size
    if dest_path.exists() and dest_path.stat().st_size == model_info.size_bytes:
        log.info("Model %s already present, skipping download.", model_info.name)
        return dest_path

    log.info("Downloading %s from %s ...", model_info.name, model_info.url)

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    try:
        urllib.request.urlretrieve(model_info.url, str(tmp_path), reporthook=_progress_hook)
        sys.stderr.write("\n")
        sys.stderr.flush()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    # Verify SHA-256 if a checksum is provided
    if model_info.sha256:
        actual = _sha256_file(tmp_path)
        if actual != model_info.sha256:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {model_info.name}: "
                f"expected {model_info.sha256}, got {actual}"
            )
        log.info("SHA-256 verified for %s.", model_info.name)

    tmp_path.rename(dest_path)
    log.info("Saved %s (%d bytes).", model_info.name, dest_path.stat().st_size)
    return dest_path


def ensure_models(dest_dir: Path | None = None) -> dict[str, Path]:
    """Download every model in :data:`MODEL_REGISTRY` and return a name-to-path map."""
    if dest_dir is None:
        dest_dir = MODELS_DIR
    paths: dict[str, Path] = {}
    for name, info in MODEL_REGISTRY.items():
        paths[name] = download_model(info, dest_dir)
    return paths


def get_model_path(name: str, dest_dir: Path | None = None) -> Path:
    """Return the expected local path for *name* (does **not** trigger a download)."""
    if dest_dir is None:
        dest_dir = MODELS_DIR
    if name in MODEL_REGISTRY:
        return dest_dir / MODEL_REGISTRY[name].filename
    return dest_dir / name


def list_stt_models() -> dict[str, dict[str, str]]:
    """Return the STT model registry."""
    return STT_MODEL_REGISTRY


def ensure_stt_model(model_size: str = "base") -> str:
    """Return the HF repo ID for *model_size*, optionally pre-downloading it.

    mlx-whisper handles the actual download from Hugging Face on first use.
    This function optionally triggers that download early by running a silent
    transcription if mlx_whisper is available.

    Args:
        model_size: One of the keys in :data:`STT_MODEL_REGISTRY` (e.g. "base").

    Returns:
        The Hugging Face repo ID string (e.g. "mlx-community/whisper-base").
    """
    if model_size not in STT_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown STT model size {model_size!r}. "
            f"Choose from: {list(STT_MODEL_REGISTRY)}"
        )

    hf_repo = STT_MODEL_REGISTRY[model_size]["hf_repo"]

    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        log.warning(
            "mlx_whisper is not installed; skipping STT model pre-download. "
            "Install it with: pip install mlx-whisper"
        )
        return hf_repo

    log.info("Pre-downloading STT model %s from %s ...", model_size, hf_repo)
    try:
        import numpy as np

        # A 0.1-second silent clip is enough to trigger the model download
        # without doing any real work.
        silent_audio = np.zeros(int(0.1 * 16000), dtype=np.float32)
        mlx_whisper.transcribe(silent_audio, path_or_hf_repo=hf_repo, language="en")
        log.info("STT model %s ready.", model_size)
    except Exception as exc:
        log.warning("STT model pre-download encountered an error: %s", exc)

    return hf_repo
