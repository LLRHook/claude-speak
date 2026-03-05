"""
Unit tests for src/daemon.py — daemon helpers and lifecycle functions.

All file system operations use tmp_path or are mocked.
Tests run without audio hardware.
"""

import os
import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from claude_speak.config import Config, TTSConfig, WakeWordConfig, AudioConfig
from claude_speak.daemon import (
    _is_stop_command,
    _try_reload_config,
    write_pid,
    read_pid,
    acquire_lock,
    handle_shutdown,
    status,
    stop_daemon,
)


# ---------------------------------------------------------------------------
# Tests: _is_stop_command
# ---------------------------------------------------------------------------

class TestIsStopCommand:
    """Tests for _is_stop_command."""

    def test_exact_match(self):
        assert _is_stop_command("stop", ["stop", "quiet", "shut up"])

    def test_case_insensitive(self):
        assert _is_stop_command("STOP", ["stop", "quiet", "shut up"])
        assert _is_stop_command("Quiet", ["stop", "quiet", "shut up"])

    def test_whitespace_stripped(self):
        assert _is_stop_command("  stop  ", ["stop", "quiet", "shut up"])
        assert _is_stop_command("\tstop\n", ["stop", "quiet", "shut up"])

    def test_multi_word_phrase(self):
        assert _is_stop_command("shut up", ["stop", "quiet", "shut up"])

    def test_non_stop_command_returns_false(self):
        assert not _is_stop_command("hello world", ["stop", "quiet", "shut up"])

    def test_partial_match_not_accepted(self):
        """'stop talking' should NOT match 'stop'."""
        assert not _is_stop_command("stop talking", ["stop", "quiet", "shut up"])

    def test_empty_text(self):
        assert not _is_stop_command("", ["stop", "quiet"])

    def test_empty_phrases_list(self):
        assert not _is_stop_command("stop", [])


# ---------------------------------------------------------------------------
# Tests: _try_reload_config
# ---------------------------------------------------------------------------

class TestTryReloadConfig:
    """Tests for _try_reload_config."""

    def test_config_reload_when_mtime_changed(self, tmp_path):
        """Should reload config when file mtime changes."""
        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nvoice = 'af_sarah'\n")

        old_config = Config()
        engine = MagicMock()
        old_mtime = 100.0

        new_config = Config(tts=TTSConfig(voice="bm_fable"))
        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config):
            result_config, result_mtime = _try_reload_config(old_config, engine, old_mtime)
            assert result_config is new_config
            assert result_mtime == config_file.stat().st_mtime
            engine._resolve_device.assert_called_once()

    def test_config_not_reloaded_when_mtime_same(self, tmp_path):
        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\n")
        current_mtime = config_file.stat().st_mtime

        old_config = Config()
        engine = MagicMock()

        with patch("claude_speak.daemon.CONFIG_PATH", config_file):
            result_config, result_mtime = _try_reload_config(old_config, engine, current_mtime)
            assert result_config is old_config
            assert result_mtime == current_mtime
            engine._resolve_device.assert_not_called()

    def test_config_reload_handles_missing_file(self):
        """If config file is missing, return existing config unchanged."""
        old_config = Config()
        engine = MagicMock()
        with patch("claude_speak.daemon.CONFIG_PATH", Path("/nonexistent/path/config.toml")):
            result_config, result_mtime = _try_reload_config(old_config, engine, 0.0)
            assert result_config is old_config


# ---------------------------------------------------------------------------
# Tests: PID file management
# ---------------------------------------------------------------------------

class TestPidFile:
    """Tests for write_pid and read_pid."""

    def test_write_and_read_pid_round_trip(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        with patch("claude_speak.daemon.PID_FILE", pid_file):
            write_pid()
            result = read_pid()
            assert result == os.getpid()

    def test_read_pid_returns_none_when_file_missing(self, tmp_path):
        pid_file = tmp_path / "nonexistent.pid"
        with patch("claude_speak.daemon.PID_FILE", pid_file):
            assert read_pid() is None

    def test_read_pid_returns_none_when_process_dead(self, tmp_path):
        """If PID file contains a dead process, return None and clean up."""
        pid_file = tmp_path / "test.pid"
        # Use a PID that almost certainly doesn't exist
        pid_file.write_text("999999999")
        with patch("claude_speak.daemon.PID_FILE", pid_file):
            result = read_pid()
            assert result is None
            # PID file should be cleaned up
            assert not pid_file.exists()

    def test_read_pid_returns_none_on_invalid_content(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("not-a-number")
        with patch("claude_speak.daemon.PID_FILE", pid_file):
            result = read_pid()
            assert result is None


# ---------------------------------------------------------------------------
# Tests: acquire_lock
# ---------------------------------------------------------------------------

class TestAcquireLock:
    """Tests for acquire_lock."""

    def test_acquire_lock_returns_true_first_time(self, tmp_path):
        lock_file = tmp_path / "test.lock"
        with patch("claude_speak.daemon.LOCK_FILE", lock_file):
            result = acquire_lock()
            assert result is True
            # Clean up the fd stored on the function
            if hasattr(acquire_lock, "_fd"):
                acquire_lock._fd.close()
                delattr(acquire_lock, "_fd")

    def test_acquire_lock_writes_pid_to_file(self, tmp_path):
        lock_file = tmp_path / "test.lock"
        with patch("claude_speak.daemon.LOCK_FILE", lock_file):
            acquire_lock()
            # On Windows the lock file byte range is held by msvcrt.locking,
            # so opening a second handle would raise PermissionError.  Read
            # the content through the already-open fd instead.
            if hasattr(acquire_lock, "_fd"):
                acquire_lock._fd.seek(0)
                content = acquire_lock._fd.read()
                acquire_lock._fd.close()
                delattr(acquire_lock, "_fd")
            else:
                content = lock_file.read_text()
            assert content.strip() == str(os.getpid())


# ---------------------------------------------------------------------------
# Tests: handle_shutdown
# ---------------------------------------------------------------------------

class TestHandleShutdown:
    """Tests for handle_shutdown."""

    def test_handle_shutdown_cleans_up_files(self, tmp_path):
        pid_file = tmp_path / "test.pid"
        lock_file = tmp_path / "test.lock"
        start_ts_file = tmp_path / "test.start_ts"
        mute_file = tmp_path / "test.muted"
        playing_file = tmp_path / "test.playing"

        # Create all sentinel files
        for f in [pid_file, lock_file, start_ts_file, mute_file, playing_file]:
            f.touch()

        with patch("claude_speak.daemon.PID_FILE", pid_file), \
             patch("claude_speak.daemon.LOCK_FILE", lock_file), \
             patch("claude_speak.daemon.START_TS_FILE", start_ts_file), \
             patch("claude_speak.daemon.MUTE_FILE", mute_file), \
             patch("claude_speak.daemon.PLAYING_FILE", playing_file):
            with pytest.raises(SystemExit) as exc_info:
                handle_shutdown(signal.SIGTERM, None)
            assert exc_info.value.code == 0
            assert not pid_file.exists()
            assert not lock_file.exists()
            assert not start_ts_file.exists()
            assert not mute_file.exists()
            assert not playing_file.exists()


# ---------------------------------------------------------------------------
# Tests: status
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for status output."""

    def test_status_when_not_running(self, tmp_path, capsys):
        pid_file = tmp_path / "test.pid"
        toggle_file = tmp_path / "toggle"
        with patch("claude_speak.daemon.read_pid", return_value=None), \
             patch("claude_speak.daemon.TOGGLE_FILE", toggle_file), \
             patch("claude_speak.daemon.Q") as mock_q:
            mock_q.depth.return_value = 0
            status()
            captured = capsys.readouterr()
            assert "stopped" in captured.out
