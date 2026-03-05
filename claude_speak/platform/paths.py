"""Centralized runtime paths for claude-speak."""
from __future__ import annotations

import os
from pathlib import Path

from . import is_windows


def _runtime_dir() -> Path:
    """Base directory for runtime state files."""
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "claude-speak"
    return Path("/tmp")


def _prefix() -> str:
    """Filename prefix -- on macOS files live directly in /tmp with prefix."""
    return "" if is_windows() else "claude-speak-"


def queue_dir() -> Path:
    if is_windows():
        return _runtime_dir() / "queue"
    return Path("/tmp/claude-speak-queue")


def pid_file() -> Path:
    if is_windows():
        return _runtime_dir() / "daemon.pid"
    return Path("/tmp/claude-speak-daemon.pid")


def mute_file() -> Path:
    if is_windows():
        return _runtime_dir() / "muted"
    return Path("/tmp/claude-speak-muted")


def playing_file() -> Path:
    if is_windows():
        return _runtime_dir() / "playing"
    return Path("/tmp/claude-speak-playing")


def lock_file() -> Path:
    if is_windows():
        return _runtime_dir() / "daemon.lock"
    return Path("/tmp/claude-speak-daemon.lock")


def start_ts_file() -> Path:
    if is_windows():
        return _runtime_dir() / "daemon.start_ts"
    return Path("/tmp/claude-speak-daemon.start_ts")


def socket_path() -> Path:
    if is_windows():
        return _runtime_dir() / "ipc.sock"  # Not used on Windows (TCP instead)
    return Path("/tmp/claude-speak.sock")


def ipc_port() -> int:
    """TCP port for IPC on Windows."""
    return 52483


def pos_file() -> Path:
    if is_windows():
        return _runtime_dir() / "pos"
    return Path("/tmp/claude-speak-pos")


def hook_lock() -> Path:
    if is_windows():
        return _runtime_dir() / "hook.lock"
    return Path("/tmp/claude-speak-hook.lock")


def perf_log() -> Path:
    if is_windows():
        return _runtime_dir() / "perf.log"
    return Path("/tmp/claude-speak-perf.log")


def ensure_runtime_dir() -> None:
    """Create the runtime directory if it doesn't exist (Windows only, /tmp always exists)."""
    if is_windows():
        _runtime_dir().mkdir(parents=True, exist_ok=True)
