"""
Unix domain socket IPC for claude-speak.

Replaces SIGUSR1 signal + file-based queue with a proper IPC protocol.
Messages are newline-delimited JSON over a Unix domain socket.

**Server side** (daemon):
    IPCServer listens on SOCKET_PATH in a background thread, accepts
    connections, dispatches messages to registered handlers, and sends
    JSON acknowledgments back.

**Client side** (hooks / CLI):
    send_message() connects, sends a JSON message, reads the response,
    and closes.  is_daemon_running() probes the socket.
"""

from __future__ import annotations

import json
import logging
import os
import selectors
import socket
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

SOCKET_PATH = Path("/tmp/claude-speak.sock")

# Maximum message size (256 KB should be more than enough)
_MAX_MSG_SIZE = 256 * 1024


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def send_message(msg: dict, timeout: float = 2.0, socket_path: Path = SOCKET_PATH) -> dict | None:
    """Send a JSON message to the daemon and return the response.

    Connects to the Unix domain socket, sends *msg* as JSON + newline,
    reads the response JSON, and closes.  Returns the parsed response
    dict, or None on any failure (connection refused, timeout, etc.).
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        payload = json.dumps(msg) + "\n"
        sock.sendall(payload.encode("utf-8"))

        # Read response until newline or EOF
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        if not data:
            return None

        # Take only the first line (in case of trailing data)
        line = data.split(b"\n", 1)[0]
        return json.loads(line.decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, ConnectionError) as exc:
        logger.debug("send_message failed: %s", exc)
        return None
    finally:
        sock.close()


def is_daemon_running(socket_path: Path = SOCKET_PATH) -> bool:
    """Check whether the daemon is reachable via the IPC socket.

    Returns True if the socket file exists AND a connection can be
    established (the daemon is actually listening).
    """
    if not socket_path.exists():
        return False

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect(str(socket_path))
        return True
    except (OSError, ConnectionError):
        return False
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

# Type alias for message handlers:
#   handler(msg_dict) -> response_dict
MessageHandler = Callable[[dict], dict]


class IPCServer:
    """Non-blocking Unix domain socket server for the daemon.

    Usage::

        server = IPCServer()
        server.register_handler("speak", my_speak_handler)
        server.start()
        ...
        server.stop()

    The server runs in a background daemon thread so it never blocks
    the asyncio main loop.
    """

    def __init__(self, socket_path: Path = SOCKET_PATH):
        self._socket_path = socket_path
        self._handlers: dict[str, MessageHandler] = {}
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- public API --

    @property
    def is_running(self) -> bool:
        """True if the server thread is alive and accepting connections."""
        return self._thread is not None and self._thread.is_alive()

    def register_handler(self, msg_type: str, handler: MessageHandler) -> None:
        """Register a handler for a given message type."""
        self._handlers[msg_type] = handler

    def start(self) -> None:
        """Bind the socket and start the listener thread."""
        if self.is_running:
            logger.warning("IPCServer.start() called but already running")
            return

        self._stop_event.clear()
        self._cleanup_socket()

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(str(self._socket_path))
        # Restrict socket to owner-only access
        os.chmod(str(self._socket_path), 0o600)
        self._server_sock.listen(16)
        self._server_sock.setblocking(False)

        self._thread = threading.Thread(
            target=self._accept_loop,
            name="ipc-server",
            daemon=True,
        )
        self._thread.start()
        logger.info("IPC server started on %s", self._socket_path)

    def stop(self) -> None:
        """Signal the server thread to stop and clean up."""
        self._stop_event.set()

        # Close the server socket to unblock select()
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        self._cleanup_socket()
        logger.info("IPC server stopped")

    # -- internals --

    def _cleanup_socket(self) -> None:
        """Remove the socket file if it exists."""
        try:
            self._socket_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _accept_loop(self) -> None:
        """Main loop: accept connections and dispatch in the same thread.

        Uses a selector so we can wake up periodically to check the
        stop event without burning CPU.
        """
        sel = selectors.DefaultSelector()
        try:
            sel.register(self._server_sock, selectors.EVENT_READ)
        except (ValueError, OSError):
            # Socket already closed (stop() called before loop started)
            return

        try:
            while not self._stop_event.is_set():
                try:
                    events = sel.select(timeout=0.25)
                except (OSError, ValueError):
                    # Socket closed from another thread
                    break

                for _key, _ in events:
                    srv = self._server_sock
                    if srv is None:
                        break
                    try:
                        conn, _ = srv.accept()
                    except (OSError, BlockingIOError, AttributeError):
                        continue
                    # Handle client in the same thread (messages are tiny + fast)
                    self._handle_client(conn)
        finally:
            sel.close()

    def _handle_client(self, conn: socket.socket) -> None:
        """Read one message from *conn*, dispatch, respond, and close."""
        conn.settimeout(2.0)
        try:
            data = b""
            while len(data) < _MAX_MSG_SIZE:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break

            if not data:
                return

            line = data.split(b"\n", 1)[0]
            msg = json.loads(line.decode("utf-8"))

            if not isinstance(msg, dict):
                response = {"ok": False, "error": "expected JSON object"}
            elif not isinstance(msg.get("type"), str) or not msg.get("type"):
                response = {"ok": False, "error": "'type' must be a non-empty string"}
            else:
                # Validate known field types for safety
                if "text" in msg and not isinstance(msg["text"], str):
                    response = {"ok": False, "error": "'text' must be a string"}
                    reply = json.dumps(response) + "\n"
                    conn.sendall(reply.encode("utf-8"))
                    return
                msg_type = msg["type"]
                handler = self._handlers.get(msg_type)
                if handler is None:
                    response = {"ok": False, "error": f"unknown message type: {msg_type!r}"}
                else:
                    try:
                        response = handler(msg)
                    except Exception as exc:
                        logger.error("Handler for %r raised: %s", msg_type, exc)
                        response = {"ok": False, "error": str(exc)}

            reply = json.dumps(response) + "\n"
            conn.sendall(reply.encode("utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.debug("IPC client error: %s", exc)
        finally:
            conn.close()
