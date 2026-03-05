"""
Model management for claude-speak.
Downloads and caches TTS model files to ~/.claude-speak/models/.
Supports Kokoro and Piper voice models.
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
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        sha256="",
        size_bytes=325_532_387,  # ~310 MB
    ),
    "voices-v1.0.bin": ModelInfo(
        name="voices-v1.0.bin",
        filename="voices-v1.0.bin",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        sha256="",
        size_bytes=28_214_398,  # ~27 MB
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
# Piper voice model registry
# ---------------------------------------------------------------------------

_PIPER_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


@dataclass(frozen=True)
class PiperVoiceInfo:
    """Metadata for a downloadable Piper voice model.

    Each Piper voice consists of an ONNX model file and a companion JSON config.
    """

    name: str  # e.g. "en_US-lessac-medium"
    description: str
    onnx_url: str
    config_url: str  # .onnx.json companion file


def _piper_urls(voice_name: str) -> tuple[str, str]:
    """Derive HuggingFace download URLs from a Piper voice name.

    Voice names follow the pattern ``{lang}_{REGION}-{speaker}-{quality}``.
    The HF path is ``{lang_lower}/{lang_REGION}/{speaker}/{quality}/{voice_name}.onnx``.
    """
    # e.g. "en_US-lessac-medium" -> lang_region="en_US", speaker="lessac", quality="medium"
    lang_region, speaker, quality = voice_name.rsplit("-", 2)
    lang_lower = lang_region.split("_")[0]  # "en"
    base = f"{_PIPER_HF_BASE}/{lang_lower}/{lang_region}/{speaker}/{quality}"
    onnx_url = f"{base}/{voice_name}.onnx"
    config_url = f"{base}/{voice_name}.onnx.json"
    return onnx_url, config_url


PIPER_VOICES: dict[str, PiperVoiceInfo] = {}

_piper_voice_defs = [
    ("en_US-lessac-medium", "US English, Lessac (female, general purpose)"),
    ("en_US-ryan-medium", "US English, Ryan (male)"),
    ("en_GB-alan-medium", "British English, Alan (male)"),
]

for _name, _desc in _piper_voice_defs:
    _onnx, _cfg = _piper_urls(_name)
    PIPER_VOICES[_name] = PiperVoiceInfo(
        name=_name,
        description=_desc,
        onnx_url=_onnx,
        config_url=_cfg,
    )

PIPER_MODELS_DIR: Path = Path.home() / ".claude-speak" / "models" / "piper"

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
        import mlx_whisper
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


# ---------------------------------------------------------------------------
# Piper model helpers
# ---------------------------------------------------------------------------


def download_piper_voice(voice_name: str, dest_dir: Path | None = None) -> Path:
    """Download a Piper voice model (.onnx + .onnx.json) and return the .onnx path.

    Skips download if the .onnx file already exists.
    """
    if dest_dir is None:
        dest_dir = PIPER_MODELS_DIR

    if voice_name not in PIPER_VOICES:
        raise ValueError(
            f"Unknown Piper voice {voice_name!r}. "
            f"Available: {list(PIPER_VOICES)}"
        )

    info = PIPER_VOICES[voice_name]
    dest_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = dest_dir / f"{voice_name}.onnx"
    config_path = dest_dir / f"{voice_name}.onnx.json"

    for url, path in [(info.onnx_url, onnx_path), (info.config_url, config_path)]:
        if path.exists():
            log.info("Piper file %s already present, skipping.", path.name)
            continue
        log.info("Downloading %s ...", url)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, str(tmp), reporthook=_progress_hook)
            sys.stderr.write("\n")
            sys.stderr.flush()
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.rename(path)
        log.info("Saved %s.", path.name)

    return onnx_path


def list_downloaded_piper_voices(dest_dir: Path | None = None) -> list[str]:
    """Return names of Piper voices that have been downloaded (both .onnx and .onnx.json present)."""
    if dest_dir is None:
        dest_dir = PIPER_MODELS_DIR
    if not dest_dir.exists():
        return []
    voices = []
    for onnx_file in sorted(dest_dir.glob("*.onnx")):
        config_file = onnx_file.with_suffix(".onnx.json")
        if config_file.exists():
            # Strip ".onnx" to get the voice name
            voices.append(onnx_file.stem)
    return voices
