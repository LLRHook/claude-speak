"""
Security audit tests for claude-speak (Phase 9.2.1 / 9.2.2).

Audit Summary
=============
The following files were audited and the issues below were found and fixed:

1. **daemon.py** (file permissions)
   - PID file created without restrictive permissions -> Fixed: os.chmod(0o600)
   - Lock file created without restrictive permissions -> Fixed: os.chmod(0o600)
   - START_TS_FILE created without restrictive permissions -> Fixed: os.chmod(0o600)
   - Log file created via RotatingFileHandler with default perms -> Fixed: os.chmod(0o600)

2. **queue.py** (file permissions)
   - Queue directory created with default permissions -> Fixed: os.chmod(0o700)
   - Queue files created with default permissions -> Fixed: os.chmod(0o600)

3. **ipc.py** (socket permissions, input validation)
   - Unix domain socket created without restrictive permissions -> Fixed: os.chmod(0o600)
   - IPC message 'type' field not validated as non-empty string -> Fixed: validation added
   - IPC message 'text' field not validated as string -> Fixed: validation added
   - _MAX_MSG_SIZE (256KB) is enforced in _handle_client -> Verified: OK

4. **tts_elevenlabs.py** (API key exposure)
   - API key could leak in exception messages logged at WARNING level
     -> Fixed: _sanitize_exception() redacts the key from exception strings
   - Added _mask_api_key() utility for safe logging of key references

5. **config.py** (API key file permissions warning)
   - Config files containing API keys had no permission check
     -> Fixed: warns at load time if config is group/other-readable

6. **hooks/speak-response.sh** (shell injection)
   - `eval "$JQ_OUTPUT"` was a shell injection vector: jq output from crafted
     transcript JSON could contain arbitrary shell commands
     -> Fixed: replaced eval with separate jq -r invocations per field
   - Queue directory and files created without restrictive permissions
     -> Fixed: chmod 700/600 after creation

7. **hooks/speak_response.py** (file permissions)
   - Queue directory and files created without restrictive permissions
     -> Fixed: os.chmod(0o700) on dir, os.chmod(0o600) on files

8. **cli.py** (API key display)
   - `cmd_config` printed API keys in plain text
     -> Fixed: sensitive keys are now masked (showing only last 4 chars)

9. **voice_input.py** (osascript calls)
   - Reviewed: All osascript calls use hardcoded AppleScript strings or
     integer keycode/modifier parameters. No user-controlled text is
     interpolated into AppleScript. No injection risk found.

10. **normalizer.py** (ReDoS)
    - Reviewed: Regex patterns are anchored or use bounded quantifiers.
      No practical ReDoS risk identified.

11. **hotkeys.py** / **media_keys.py** (privilege issues)
    - Reviewed: Properly document Accessibility permission requirements.
      No unsafe event handling found.

12. **tts.py** (file handling)
    - Reviewed: No direct file I/O on untrusted paths. Audio is streamed
      through numpy arrays and sounddevice. No issues found.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Skip markers for platform-specific tests
_skip_windows_perms = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX file permissions (chmod 0o600/0o700) not enforced on Windows",
)
_skip_windows_unix_socket = pytest.mark.skipif(
    sys.platform == "win32",
    reason="AF_UNIX sockets not reliably available on Windows",
)


# =========================================================================
# 1. File permission tests
# =========================================================================


@_skip_windows_perms
class TestDaemonFilePermissions:
    """Verify that daemon runtime files are created with restrictive permissions."""

    def test_write_pid_sets_0o600(self, tmp_path: Path):
        """PID file should be owner-read/write only."""
        pid_file = tmp_path / "test.pid"
        with patch("claude_speak.daemon.PID_FILE", pid_file):
            from claude_speak.daemon import write_pid
            write_pid()
        assert pid_file.exists()
        mode = stat.S_IMODE(pid_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_acquire_lock_sets_0o600(self, tmp_path: Path):
        """Lock file should be owner-read/write only."""
        lock_file = tmp_path / "test.lock"
        with patch("claude_speak.daemon.LOCK_FILE", lock_file):
            from claude_speak.daemon import acquire_lock
            result = acquire_lock()
        assert result is True
        assert lock_file.exists()
        mode = stat.S_IMODE(lock_file.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


@_skip_windows_perms
class TestQueueFilePermissions:
    """Verify that queue directory and files have restrictive permissions."""

    def test_ensure_queue_dir_sets_0o700(self, tmp_path: Path):
        """Queue directory should be owner-only."""
        queue_dir = tmp_path / "queue"
        with patch("claude_speak.queue.QUEUE_DIR", queue_dir):
            from claude_speak.queue import ensure_queue_dir
            ensure_queue_dir()
        assert queue_dir.exists()
        mode = stat.S_IMODE(queue_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_enqueue_file_sets_0o600(self, tmp_path: Path):
        """Queue files should be owner-read/write only."""
        queue_dir = tmp_path / "queue"
        with patch("claude_speak.queue.QUEUE_DIR", queue_dir):
            from claude_speak.queue import enqueue
            path = enqueue("test text")
        assert path.exists()
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

    def test_enqueue_chunks_files_set_0o600(self, tmp_path: Path):
        """All chunk files should be owner-read/write only."""
        queue_dir = tmp_path / "queue"
        with patch("claude_speak.queue.QUEUE_DIR", queue_dir):
            from claude_speak.queue import enqueue_chunks
            paths = enqueue_chunks(["chunk 1", "chunk 2"])
        for p in paths:
            mode = stat.S_IMODE(p.stat().st_mode)
            assert mode == 0o600, f"Expected 0o600 for {p}, got {oct(mode)}"


@_skip_windows_perms
class TestHookQueuePermissions:
    """Verify that the Python hook sets restrictive permissions on queue files."""

    def test_enqueue_text_sets_permissions(self, tmp_path: Path):
        """enqueue_text should create dir with 0o700 and file with 0o600."""
        from claude_speak.hooks.speak_response import enqueue_text

        queue_dir = tmp_path / "hook-queue"
        path = enqueue_text("hello", queue_dir=queue_dir)

        assert path is not None
        dir_mode = stat.S_IMODE(queue_dir.stat().st_mode)
        assert dir_mode == 0o700, f"Expected dir 0o700, got {oct(dir_mode)}"
        file_mode = stat.S_IMODE(path.stat().st_mode)
        assert file_mode == 0o600, f"Expected file 0o600, got {oct(file_mode)}"


# =========================================================================
# 2. IPC socket permission test
# =========================================================================


@_skip_windows_unix_socket
class TestIPCSocketPermissions:
    """Verify that the IPC socket has restrictive permissions."""

    def test_socket_has_0o600_permissions(self):
        """After binding, the socket file should be owner-only."""
        import uuid
        from claude_speak.ipc import IPCServer

        sock_path = Path(f"/tmp/cs-sec-test-{uuid.uuid4().hex[:8]}.sock")
        try:
            server = IPCServer(socket_path=sock_path)
            server.start()
            # Wait for socket to appear
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if sock_path.exists():
                    break
                time.sleep(0.05)
            assert sock_path.exists(), "Socket file was not created"
            mode = stat.S_IMODE(sock_path.stat().st_mode)
            assert mode == 0o600, f"Expected socket 0o600, got {oct(mode)}"
        finally:
            server.stop()
            sock_path.unlink(missing_ok=True)


# =========================================================================
# 3. IPC message validation tests
# =========================================================================


@_skip_windows_unix_socket
class TestIPCMessageValidation:
    """Verify that the IPC server validates message fields."""

    @pytest.fixture
    def sock_path(self) -> Path:
        import uuid
        path = Path(f"/tmp/cs-sec-val-{uuid.uuid4().hex[:8]}.sock")
        yield path
        path.unlink(missing_ok=True)

    def _wait_for_socket(self, path: Path, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    s.connect(str(path))
                    s.close()
                    return
                except OSError:
                    pass
            time.sleep(0.05)
        raise TimeoutError(f"Socket {path} not ready after {timeout}s")

    def _send_raw(self, sock_path: Path, msg: dict) -> dict | None:
        """Send a raw JSON message and return the response."""
        from claude_speak.ipc import send_message
        return send_message(msg, socket_path=sock_path, timeout=2.0)

    def test_rejects_missing_type(self, sock_path: Path):
        """Messages without 'type' should be rejected."""
        from claude_speak.ipc import IPCServer

        server = IPCServer(socket_path=sock_path)
        server.register_handler("echo", lambda m: {"ok": True})
        server.start()
        self._wait_for_socket(sock_path)
        try:
            resp = self._send_raw(sock_path, {"text": "hello"})
            assert resp is not None
            assert resp["ok"] is False
            assert "type" in resp["error"].lower()
        finally:
            server.stop()

    def test_rejects_non_string_type(self, sock_path: Path):
        """Messages with non-string 'type' should be rejected."""
        from claude_speak.ipc import IPCServer

        server = IPCServer(socket_path=sock_path)
        server.register_handler("echo", lambda m: {"ok": True})
        server.start()
        self._wait_for_socket(sock_path)
        try:
            resp = self._send_raw(sock_path, {"type": 123})
            assert resp is not None
            assert resp["ok"] is False
            assert "type" in resp["error"].lower()
        finally:
            server.stop()

    def test_rejects_non_string_text(self, sock_path: Path):
        """Messages with non-string 'text' should be rejected."""
        from claude_speak.ipc import IPCServer

        server = IPCServer(socket_path=sock_path)
        server.register_handler("speak", lambda m: {"ok": True})
        server.start()
        self._wait_for_socket(sock_path)
        try:
            resp = self._send_raw(sock_path, {"type": "speak", "text": 12345})
            assert resp is not None
            assert resp["ok"] is False
            assert "text" in resp["error"].lower()
        finally:
            server.stop()

    def test_accepts_valid_message(self, sock_path: Path):
        """Valid messages with string type and text should be accepted."""
        from claude_speak.ipc import IPCServer

        server = IPCServer(socket_path=sock_path)
        server.register_handler("speak", lambda m: {"ok": True, "echo": m.get("text")})
        server.start()
        self._wait_for_socket(sock_path)
        try:
            resp = self._send_raw(sock_path, {"type": "speak", "text": "hello"})
            assert resp is not None
            assert resp["ok"] is True
            assert resp["echo"] == "hello"
        finally:
            server.stop()


# =========================================================================
# 4. API key masking tests
# =========================================================================


class TestAPIKeyMasking:
    """Verify that API keys are masked in logs and string representations."""

    def test_mask_api_key_standard(self):
        """Standard-length key should show only last 4 chars."""
        from claude_speak.tts_elevenlabs import _mask_api_key
        key = "sk-1234567890abcdef"
        masked = _mask_api_key(key)
        assert masked.endswith("cdef")
        assert "1234567890" not in masked
        assert masked == "*" * (len(key) - 4) + "cdef"

    def test_mask_api_key_short(self):
        """Very short key should be fully masked."""
        from claude_speak.tts_elevenlabs import _mask_api_key
        assert _mask_api_key("ab") == "****"
        assert _mask_api_key("") == "****"

    def test_sanitize_exception_redacts_key(self):
        """Exception messages containing the API key should have it redacted."""
        from claude_speak.tts_elevenlabs import _sanitize_exception, _mask_api_key
        key = "sk-secret-api-key-12345678"
        exc = RuntimeError(f"Authentication failed for key {key}")
        sanitized = _sanitize_exception(exc, key)
        assert key not in sanitized
        assert _mask_api_key(key) in sanitized

    def test_sanitize_exception_no_key_in_message(self):
        """When key is not in the exception, the message is returned as-is."""
        from claude_speak.tts_elevenlabs import _sanitize_exception
        exc = RuntimeError("Network timeout")
        result = _sanitize_exception(exc, "sk-12345678")
        assert result == "Network timeout"


# =========================================================================
# 5. CLI config masking test
# =========================================================================


class TestCLIConfigMasking:
    """Verify that `cmd_config` masks sensitive fields."""

    def test_api_key_masked_in_config_output(self, capsys):
        """The elevenlabs_api_key should be masked when printing config."""
        from claude_speak.config import Config

        config = Config()
        config.tts.elevenlabs_api_key = "sk-secret-key-1234"

        with patch("claude_speak.cli.load_config", return_value=config):
            from claude_speak.cli import cmd_config
            cmd_config()

        output = capsys.readouterr().out
        assert "sk-secret-key-1234" not in output
        assert "1234" in output  # last 4 chars should be visible
        assert "****" in output or "**" in output  # should contain masking

    def test_empty_api_key_not_masked(self, capsys):
        """An empty API key should not trigger masking."""
        from claude_speak.config import Config

        config = Config()
        config.tts.elevenlabs_api_key = ""

        with patch("claude_speak.cli.load_config", return_value=config):
            from claude_speak.cli import cmd_config
            cmd_config()

        output = capsys.readouterr().out
        assert "elevenlabs_api_key = ''" in output


# =========================================================================
# 6. Config file permission warning test
# =========================================================================


@_skip_windows_perms
class TestConfigPermissionWarning:
    """Verify that load_config warns about too-open config file permissions."""

    def test_warns_when_config_world_readable(self, tmp_path: Path):
        """If config file with API key is world-readable, a warning is logged."""
        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text(
            '[tts]\nelevenlabs_api_key = "sk-test-key"\n'
        )
        # Set world-readable
        os.chmod(config_file, 0o644)

        with patch("claude_speak.config.CONFIG_PATH", config_file), \
             patch("claude_speak.config._config_logger") as mock_logger:
            from claude_speak.config import load_config
            load_config()
            # Should have warned about permissions
            assert mock_logger.warning.called
            warning_msg = mock_logger.warning.call_args_list[0][0][0]
            assert "readable by group/others" in warning_msg


# =========================================================================
# 7. Shell injection prevention test (speak-response.sh)
# =========================================================================


class TestShellScriptSecurity:
    """Verify that the shell hook does not use eval on untrusted input."""

    def test_no_eval_of_jq_output(self):
        """The shell script should not eval jq output (shell injection vector)."""
        script_path = Path(__file__).resolve().parent.parent / "hooks" / "speak-response.sh"
        content = script_path.read_text()
        # The old vulnerable pattern was: eval "$JQ_OUTPUT"
        # After the fix, we use separate jq -r calls per field
        assert 'eval "$JQ_OUTPUT"' not in content, (
            "speak-response.sh still uses eval on jq output -- this is a shell injection vector"
        )
