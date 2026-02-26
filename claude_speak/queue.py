"""
Queue manager — file-based FIFO queue for TTS messages.
Producer (hook script) writes text files. Consumer (daemon) reads and speaks them.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from .config import QUEUE_DIR


def ensure_queue_dir() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    # Restrict queue directory to owner-only access
    os.chmod(QUEUE_DIR, 0o700)


def enqueue(text: str) -> Path:
    """Write text to a new queue file. Returns the file path."""
    ensure_queue_dir()
    # Timestamp with microseconds for uniqueness and ordering
    timestamp = f"{time.time():.6f}"
    path = QUEUE_DIR / f"{timestamp}.txt"
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def enqueue_chunks(chunks: list[str]) -> list[Path]:
    """Write multiple chunks as sequential queue files."""
    ensure_queue_dir()
    paths: list[Path] = []
    base_time = time.time()
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        # Increment by small amount to preserve order
        timestamp = f"{base_time + i * 0.000001:.6f}"
        path = QUEUE_DIR / f"{timestamp}.txt"
        path.write_text(chunk, encoding="utf-8")
        os.chmod(path, 0o600)
        paths.append(path)
    return paths


def peek() -> list[Path]:
    """List queued files in order."""
    ensure_queue_dir()
    return sorted(QUEUE_DIR.glob("*.txt"))


def dequeue() -> tuple[Path, str] | None:
    """Pop the next file from the queue. Returns (path, text) or None."""
    files = peek()
    if not files:
        return None
    path = files[0]
    try:
        text = path.read_text(encoding="utf-8").strip()
        path.unlink()
        return (path, text)
    except (FileNotFoundError, PermissionError):
        return None


def clear() -> None:
    """Remove all queued files."""
    for f in peek():
        with contextlib.suppress(OSError):
            f.unlink(missing_ok=True)


def depth() -> int:
    """Number of items in the queue."""
    return len(peek())
