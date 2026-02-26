"""
Comprehensive tests for claude_speak/ipc.py — Unix domain socket IPC.

Covers: server lifecycle, message handling, acknowledgment format,
send_message, is_daemon_running, connection timeout, socket cleanup,
fallback behavior, and concurrent connections.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_speak.ipc import (
    IPCServer,
    SOCKET_PATH,
    send_message,
    is_daemon_running,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sock_path() -> Path:
    """Return a unique temporary socket path for test isolation.

    Uses /tmp directly because Unix domain socket paths are limited to
    ~104 bytes on macOS, and pytest's tmp_path is typically too long.
    """
    import uuid
    path = Path(f"/tmp/cs-test-{uuid.uuid4().hex[:8]}.sock")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def server(sock_path: Path) -> IPCServer:
    """Create an IPCServer bound to a temporary socket path.

    Automatically stops the server after the test.
    """
    srv = IPCServer(socket_path=sock_path)
    yield srv
    # Ensure cleanup even if the test didn't call stop()
    if srv.is_running:
        srv.stop()
    sock_path.unlink(missing_ok=True)


@pytest.fixture
def echo_server(server: IPCServer, sock_path: Path) -> IPCServer:
    """An IPCServer with an 'echo' handler that returns the message text."""
    def _echo(msg: dict) -> dict:
        return {"ok": True, "echo": msg.get("text", "")}

    server.register_handler("echo", _echo)
    server.start()
    # Give the listener thread a moment to bind
    _wait_for_socket(sock_path)
    return server


def _wait_for_socket(path: Path, timeout: float = 2.0) -> None:
    """Block until the socket file exists and accepts connections."""
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


# =========================================================================
# Tests: server start/stop lifecycle
# =========================================================================

class TestServerLifecycle:
    """Test IPCServer start, stop, and is_running."""

    def test_is_running_false_before_start(self, server: IPCServer):
        assert server.is_running is False

    def test_start_creates_socket_file(self, server: IPCServer, sock_path: Path):
        server.start()
        _wait_for_socket(sock_path)
        assert sock_path.exists()
        assert server.is_running is True

    def test_stop_removes_socket_file(self, server: IPCServer, sock_path: Path):
        server.start()
        _wait_for_socket(sock_path)
        server.stop()
        assert not sock_path.exists()
        assert server.is_running is False

    def test_double_start_is_safe(self, server: IPCServer, sock_path: Path):
        """Calling start() twice should not crash."""
        server.start()
        _wait_for_socket(sock_path)
        server.start()  # should be a no-op warning
        assert server.is_running is True

    def test_double_stop_is_safe(self, server: IPCServer, sock_path: Path):
        """Calling stop() twice should not crash."""
        server.start()
        _wait_for_socket(sock_path)
        server.stop()
        server.stop()  # should be a no-op
        assert server.is_running is False

    def test_stop_before_start_is_safe(self, server: IPCServer):
        """Calling stop() without start() should not crash."""
        server.stop()  # no-op
        assert server.is_running is False

    def test_socket_cleanup_on_shutdown(self, sock_path: Path):
        """Socket file must be removed after stop()."""
        srv = IPCServer(socket_path=sock_path)
        srv.start()
        _wait_for_socket(sock_path)
        assert sock_path.exists()
        srv.stop()
        assert not sock_path.exists()


# =========================================================================
# Tests: send_message with mock server
# =========================================================================

class TestSendMessage:
    """Test the client-side send_message function."""

    def test_send_and_receive_echo(self, echo_server: IPCServer, sock_path: Path):
        response = send_message(
            {"type": "echo", "text": "hello"},
            socket_path=sock_path,
        )
        assert response is not None
        assert response["ok"] is True
        assert response["echo"] == "hello"

    def test_send_returns_none_when_no_server(self, sock_path: Path):
        """When no server is listening, send_message returns None."""
        result = send_message(
            {"type": "echo", "text": "hello"},
            timeout=0.5,
            socket_path=sock_path,
        )
        assert result is None

    def test_send_returns_error_for_unknown_type(self, echo_server: IPCServer, sock_path: Path):
        response = send_message(
            {"type": "nonexistent"},
            socket_path=sock_path,
        )
        assert response is not None
        assert response["ok"] is False
        assert "unknown message type" in response["error"]


# =========================================================================
# Tests: speak message handling
# =========================================================================

class TestSpeakHandler:
    """Test the 'speak' message type end-to-end via IPC."""

    def test_speak_enqueues_text(self, server: IPCServer, sock_path: Path):
        enqueued: list[str] = []

        def _speak_handler(msg: dict) -> dict:
            text = msg.get("text", "")
            if not text:
                return {"ok": False, "error": "empty text"}
            enqueued.append(text)
            return {"ok": True}

        server.register_handler("speak", _speak_handler)
        server.start()
        _wait_for_socket(sock_path)

        response = send_message(
            {"type": "speak", "text": "Hello world"},
            socket_path=sock_path,
        )
        assert response == {"ok": True}
        assert enqueued == ["Hello world"]

    def test_speak_empty_text_returns_error(self, server: IPCServer, sock_path: Path):
        def _speak_handler(msg: dict) -> dict:
            text = msg.get("text", "")
            if not text:
                return {"ok": False, "error": "empty text"}
            return {"ok": True}

        server.register_handler("speak", _speak_handler)
        server.start()
        _wait_for_socket(sock_path)

        response = send_message(
            {"type": "speak", "text": ""},
            socket_path=sock_path,
        )
        assert response is not None
        assert response["ok"] is False
        assert "empty text" in response["error"]


# =========================================================================
# Tests: stop message handling
# =========================================================================

class TestStopHandler:
    """Test the 'stop' message type."""

    def test_stop_calls_callbacks(self, server: IPCServer, sock_path: Path):
        stopped = []

        def _stop_handler(msg: dict) -> dict:
            stopped.append(True)
            return {"ok": True}

        server.register_handler("stop", _stop_handler)
        server.start()
        _wait_for_socket(sock_path)

        response = send_message({"type": "stop"}, socket_path=sock_path)
        assert response == {"ok": True}
        assert stopped == [True]


# =========================================================================
# Tests: status response
# =========================================================================

class TestStatusHandler:
    """Test the 'status' message type."""

    def test_status_returns_expected_fields(self, server: IPCServer, sock_path: Path):
        def _status_handler(msg: dict) -> dict:
            return {
                "ok": True,
                "queue_depth": 3,
                "enabled": True,
                "uptime": 42.5,
            }

        server.register_handler("status", _status_handler)
        server.start()
        _wait_for_socket(sock_path)

        response = send_message({"type": "status"}, socket_path=sock_path)
        assert response is not None
        assert response["ok"] is True
        assert response["queue_depth"] == 3
        assert response["enabled"] is True
        assert response["uptime"] == 42.5


# =========================================================================
# Tests: acknowledgment format
# =========================================================================

class TestAcknowledgmentFormat:
    """Verify the JSON acknowledgment format."""

    def test_ok_response_format(self, echo_server: IPCServer, sock_path: Path):
        response = send_message(
            {"type": "echo", "text": "test"},
            socket_path=sock_path,
        )
        assert isinstance(response, dict)
        assert "ok" in response
        assert response["ok"] is True

    def test_error_response_has_error_field(self, echo_server: IPCServer, sock_path: Path):
        response = send_message(
            {"type": "bad_type"},
            socket_path=sock_path,
        )
        assert isinstance(response, dict)
        assert response["ok"] is False
        assert "error" in response
        assert isinstance(response["error"], str)

    def test_handler_exception_returns_error(self, server: IPCServer, sock_path: Path):
        def _failing_handler(msg: dict) -> dict:
            raise ValueError("kaboom")

        server.register_handler("boom", _failing_handler)
        server.start()
        _wait_for_socket(sock_path)

        response = send_message({"type": "boom"}, socket_path=sock_path)
        assert response is not None
        assert response["ok"] is False
        assert "kaboom" in response["error"]


# =========================================================================
# Tests: connection timeout
# =========================================================================

class TestConnectionTimeout:
    """Test that send_message respects its timeout parameter."""

    def test_timeout_returns_none(self, sock_path: Path):
        """When no server is running, send_message times out and returns None."""
        t0 = time.monotonic()
        result = send_message(
            {"type": "echo", "text": "hi"},
            timeout=0.3,
            socket_path=sock_path,
        )
        elapsed = time.monotonic() - t0
        assert result is None
        # Should have completed reasonably fast (not hung)
        assert elapsed < 2.0


# =========================================================================
# Tests: socket cleanup on shutdown
# =========================================================================

class TestSocketCleanup:
    """Verify socket file is cleaned up properly."""

    def test_cleanup_on_stop(self, sock_path: Path):
        srv = IPCServer(socket_path=sock_path)
        srv.start()
        _wait_for_socket(sock_path)
        assert sock_path.exists()
        srv.stop()
        assert not sock_path.exists()

    def test_stale_socket_cleaned_on_start(self, sock_path: Path):
        """If a stale socket file exists, start() should replace it."""
        # Create a stale socket file (not a real socket)
        sock_path.touch()
        assert sock_path.exists()

        srv = IPCServer(socket_path=sock_path)
        srv.start()
        _wait_for_socket(sock_path)
        assert srv.is_running
        srv.stop()


# =========================================================================
# Tests: fallback behavior (socket unavailable)
# =========================================================================

class TestFallbackBehavior:
    """Test that the hook falls back to file queue when IPC is unavailable."""

    def test_send_to_daemon_returns_false_when_no_server(self):
        """send_to_daemon() should return False when no IPC server is running."""
        from claude_speak.hooks.speak_response import send_to_daemon

        # Use a socket path that definitely doesn't exist
        with patch("claude_speak.ipc.SOCKET_PATH", Path("/tmp/nonexistent-test.sock")):
            result = send_to_daemon("hello")
        assert result is False

    def test_send_to_daemon_returns_true_on_success(self, server: IPCServer, sock_path: Path):
        """send_to_daemon() should return True when the IPC server accepts."""
        from claude_speak.hooks.speak_response import send_to_daemon

        def _speak_handler(msg: dict) -> dict:
            return {"ok": True}

        server.register_handler("speak", _speak_handler)
        server.start()
        _wait_for_socket(sock_path)

        # Patch send_message at the point of import inside send_to_daemon.
        # The default socket_path parameter is bound at definition time, so
        # we wrap send_message to inject our test socket path.
        original_send = send_message
        def _patched_send(msg, timeout=2.0, socket_path=sock_path):
            return original_send(msg, timeout=timeout, socket_path=socket_path)

        with patch("claude_speak.ipc.send_message", _patched_send):
            result = send_to_daemon("hello world")
        assert result is True

    def test_hook_falls_back_to_file_queue(self, tmp_path: Path):
        """When IPC fails, the hook should write to the file queue."""
        from claude_speak.hooks.speak_response import send_to_daemon, enqueue_text

        # send_to_daemon fails
        with patch("claude_speak.ipc.SOCKET_PATH", Path("/tmp/nonexistent-test.sock")):
            assert send_to_daemon("hello") is False

        # File queue still works
        queue_dir = tmp_path / "queue"
        queue_dir.mkdir()
        path = enqueue_text("hello", queue_dir=queue_dir)
        assert path is not None
        assert path.read_text(encoding="utf-8") == "hello"


# =========================================================================
# Tests: is_daemon_running
# =========================================================================

class TestIsDaemonRunning:
    """Test the is_daemon_running() probe."""

    def test_returns_false_when_no_socket(self, sock_path: Path):
        assert is_daemon_running(socket_path=sock_path) is False

    def test_returns_false_when_socket_file_but_no_server(self, sock_path: Path):
        """A stale socket file (no listener) should return False."""
        sock_path.touch()  # not a real socket
        assert is_daemon_running(socket_path=sock_path) is False

    def test_returns_true_when_server_listening(self, echo_server: IPCServer, sock_path: Path):
        assert is_daemon_running(socket_path=sock_path) is True

    def test_returns_false_after_server_stops(self, server: IPCServer, sock_path: Path):
        server.register_handler("echo", lambda m: {"ok": True})
        server.start()
        _wait_for_socket(sock_path)
        assert is_daemon_running(socket_path=sock_path) is True

        server.stop()
        assert is_daemon_running(socket_path=sock_path) is False


# =========================================================================
# Tests: concurrent connections
# =========================================================================

class TestConcurrentConnections:
    """Verify the server handles multiple simultaneous clients."""

    def test_multiple_sequential_messages(self, echo_server: IPCServer, sock_path: Path):
        """Multiple messages sent sequentially should all succeed."""
        for i in range(10):
            response = send_message(
                {"type": "echo", "text": f"msg-{i}"},
                socket_path=sock_path,
            )
            assert response is not None
            assert response["ok"] is True
            assert response["echo"] == f"msg-{i}"

    def test_concurrent_messages(self, echo_server: IPCServer, sock_path: Path):
        """Multiple threads sending simultaneously should all get responses."""
        results: list[dict | None] = [None] * 10
        errors: list[Exception | None] = [None] * 10

        def _send(idx: int) -> None:
            try:
                results[idx] = send_message(
                    {"type": "echo", "text": f"concurrent-{idx}"},
                    socket_path=sock_path,
                    timeout=5.0,
                )
            except Exception as exc:
                errors[idx] = exc

        threads = [threading.Thread(target=_send, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # All threads should have completed without error
        for i, err in enumerate(errors):
            assert err is None, f"Thread {i} raised: {err}"

        # All responses should be valid
        for i, resp in enumerate(results):
            assert resp is not None, f"Thread {i} got None response"
            assert resp["ok"] is True
            assert resp["echo"] == f"concurrent-{i}"


# =========================================================================
# Tests: daemon IPC integration (handlers wired in daemon._create_ipc_server)
# =========================================================================

class TestDaemonIPCIntegration:
    """Test the IPC handlers as wired by the daemon's _create_ipc_server."""

    def test_speak_handler_enqueues_via_Q(self, sock_path: Path):
        """The daemon speak handler should call Q.enqueue."""
        import asyncio
        from claude_speak.daemon import _create_ipc_server

        loop = asyncio.new_event_loop()
        queue_ready = asyncio.Event()
        engine = MagicMock()

        with patch("claude_speak.daemon.Q") as mock_q, \
             patch("claude_speak.daemon.TOGGLE_FILE") as mock_toggle, \
             patch("claude_speak.daemon.START_TS_FILE") as mock_start_ts:
            config = MagicMock()
            server = _create_ipc_server(engine, config, queue_ready, loop)
            server._socket_path = sock_path
            server.start()
            _wait_for_socket(sock_path)

            try:
                resp = send_message(
                    {"type": "speak", "text": "test text"},
                    socket_path=sock_path,
                )
                assert resp is not None
                assert resp["ok"] is True
                mock_q.enqueue.assert_called_once_with("test text")
            finally:
                server.stop()
                loop.close()

    def test_stop_handler_calls_engine_stop_and_queue_clear(self, sock_path: Path):
        """The daemon stop handler should call engine.stop() and Q.clear()."""
        import asyncio
        from claude_speak.daemon import _create_ipc_server

        loop = asyncio.new_event_loop()
        queue_ready = asyncio.Event()
        engine = MagicMock()

        with patch("claude_speak.daemon.Q") as mock_q, \
             patch("claude_speak.daemon.TOGGLE_FILE"), \
             patch("claude_speak.daemon.START_TS_FILE"):
            config = MagicMock()
            server = _create_ipc_server(engine, config, queue_ready, loop)
            server._socket_path = sock_path
            server.start()
            _wait_for_socket(sock_path)

            try:
                resp = send_message(
                    {"type": "stop"},
                    socket_path=sock_path,
                )
                assert resp == {"ok": True}
                engine.stop.assert_called_once()
                mock_q.clear.assert_called_once()
            finally:
                server.stop()
                loop.close()

    def test_status_handler_returns_expected_fields(self, sock_path: Path, tmp_path: Path):
        """The daemon status handler should return queue_depth, enabled, uptime."""
        import asyncio
        from claude_speak.daemon import _create_ipc_server

        loop = asyncio.new_event_loop()
        queue_ready = asyncio.Event()
        engine = MagicMock()

        toggle = tmp_path / "toggle"
        toggle.touch()
        start_ts = tmp_path / "start_ts"
        start_ts.write_text(str(time.time() - 60))  # 60s ago

        with patch("claude_speak.daemon.Q") as mock_q, \
             patch("claude_speak.daemon.TOGGLE_FILE", toggle), \
             patch("claude_speak.daemon.START_TS_FILE", start_ts):
            mock_q.depth.return_value = 5
            config = MagicMock()
            server = _create_ipc_server(engine, config, queue_ready, loop)
            server._socket_path = sock_path
            server.start()
            _wait_for_socket(sock_path)

            try:
                resp = send_message(
                    {"type": "status"},
                    socket_path=sock_path,
                )
                assert resp is not None
                assert resp["ok"] is True
                assert resp["queue_depth"] == 5
                assert resp["enabled"] is True
                assert resp["uptime"] >= 59.0  # at least 59s
            finally:
                server.stop()
                loop.close()

    def test_speak_empty_text_rejected(self, sock_path: Path):
        """Speak handler should reject empty text."""
        import asyncio
        from claude_speak.daemon import _create_ipc_server

        loop = asyncio.new_event_loop()
        queue_ready = asyncio.Event()
        engine = MagicMock()

        with patch("claude_speak.daemon.Q") as mock_q, \
             patch("claude_speak.daemon.TOGGLE_FILE"), \
             patch("claude_speak.daemon.START_TS_FILE"):
            config = MagicMock()
            server = _create_ipc_server(engine, config, queue_ready, loop)
            server._socket_path = sock_path
            server.start()
            _wait_for_socket(sock_path)

            try:
                resp = send_message(
                    {"type": "speak", "text": ""},
                    socket_path=sock_path,
                )
                assert resp is not None
                assert resp["ok"] is False
                mock_q.enqueue.assert_not_called()
            finally:
                server.stop()
                loop.close()
