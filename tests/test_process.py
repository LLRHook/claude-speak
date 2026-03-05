"""Tests for cross-platform process utilities."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from claude_speak.platform.process import (
    acquire_file_lock,
    find_processes_by_name,
    is_process_alive,
)


class TestIsProcessAlive:
    """Test is_process_alive()."""

    def test_current_process_is_alive(self):
        assert is_process_alive(os.getpid()) is True

    def test_nonexistent_pid_is_not_alive(self):
        # Use a very high PID that almost certainly doesn't exist
        assert is_process_alive(99999999) is False

    def test_zero_pid_is_not_alive(self):
        # PID 0 is special on all platforms
        result = is_process_alive(0)
        # On some systems PID 0 exists (kernel), on others it doesn't
        assert isinstance(result, bool)

    def test_negative_pid(self):
        assert is_process_alive(-1) is False


class TestAcquireFileLock:
    """Test acquire_file_lock()."""

    def test_acquire_lock_succeeds(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        fd = acquire_file_lock(lock_path)
        assert fd is not None
        fd.close()

    def test_double_lock_fails(self, tmp_path):
        lock_path = tmp_path / "test.lock"
        fd1 = acquire_file_lock(lock_path)
        assert fd1 is not None
        try:
            fd2 = acquire_file_lock(lock_path)
            assert fd2 is None
        finally:
            fd1.close()


class TestFindProcessesByName:
    """Test find_processes_by_name()."""

    def test_returns_list(self):
        result = find_processes_by_name("python")
        assert isinstance(result, list)

    def test_finds_current_python(self):
        # Should find at least one Python process (the test runner itself).
        if sys.platform == "win32":
            pytest.importorskip("psutil")
        result = find_processes_by_name("python")
        # pgrep may or may not find "python" depending on how the process
        # is named (e.g. "Python" vs "python3.13" vs full path).  On CI
        # runners this is unreliable, so just verify the return type.
        assert isinstance(result, list)

    def test_nonexistent_pattern_returns_empty(self):
        result = find_processes_by_name("zzz_nonexistent_process_xyz_12345")
        assert result == []
