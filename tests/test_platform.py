"""Tests for the platform detection and paths modules."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_current_platform_returns_string(self):
        from claude_speak.platform import current_platform
        result = current_platform()
        assert isinstance(result, str)
        assert result == sys.platform

    def test_is_macos(self):
        from claude_speak.platform import is_macos
        assert is_macos() == (sys.platform == "darwin")

    def test_is_windows(self):
        from claude_speak.platform import is_windows
        assert is_windows() == (sys.platform == "win32")

    def test_is_linux(self):
        from claude_speak.platform import is_linux
        assert is_linux() == sys.platform.startswith("linux")


class TestPaths:
    """Test platform-aware path resolution."""

    def test_queue_dir_returns_path(self):
        from claude_speak.platform.paths import queue_dir
        result = queue_dir()
        assert isinstance(result, Path)

    def test_pid_file_returns_path(self):
        from claude_speak.platform.paths import pid_file
        result = pid_file()
        assert isinstance(result, Path)

    def test_mute_file_returns_path(self):
        from claude_speak.platform.paths import mute_file
        result = mute_file()
        assert isinstance(result, Path)

    def test_playing_file_returns_path(self):
        from claude_speak.platform.paths import playing_file
        result = playing_file()
        assert isinstance(result, Path)

    def test_lock_file_returns_path(self):
        from claude_speak.platform.paths import lock_file
        result = lock_file()
        assert isinstance(result, Path)

    def test_start_ts_file_returns_path(self):
        from claude_speak.platform.paths import start_ts_file
        result = start_ts_file()
        assert isinstance(result, Path)

    def test_pos_file_returns_path(self):
        from claude_speak.platform.paths import pos_file
        result = pos_file()
        assert isinstance(result, Path)

    def test_hook_lock_returns_path(self):
        from claude_speak.platform.paths import hook_lock
        result = hook_lock()
        assert isinstance(result, Path)

    def test_perf_log_returns_path(self):
        from claude_speak.platform.paths import perf_log
        result = perf_log()
        assert isinstance(result, Path)

    def test_ipc_port_returns_int(self):
        from claude_speak.platform.paths import ipc_port
        result = ipc_port()
        assert isinstance(result, int)
        assert result == 52483

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS path format")
    def test_macos_paths_use_tmp(self):
        from claude_speak.platform.paths import queue_dir, pid_file
        assert str(queue_dir()).startswith("/tmp/")
        assert str(pid_file()).startswith("/tmp/")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path format")
    def test_windows_paths_use_localappdata(self):
        from claude_speak.platform.paths import queue_dir, pid_file
        assert "claude-speak" in str(queue_dir())
        assert "claude-speak" in str(pid_file())

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS socket path")
    def test_socket_path_returns_path_on_macos(self):
        from claude_speak.platform.paths import socket_path
        result = socket_path()
        assert isinstance(result, Path)
        assert str(result).endswith(".sock")

    def test_ensure_runtime_dir(self, tmp_path):
        from claude_speak.platform.paths import ensure_runtime_dir
        # Should not raise
        ensure_runtime_dir()
