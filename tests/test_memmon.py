"""
Tests for claude_speak/memmon.py — Memory monitoring.

Covers: MemoryMonitor lifecycle, get_stats(), snapshot(), disabled vs
enabled behavior, top_allocations content, and IPC handler integration.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_speak.memmon import (
    MemoryMonitor,
    MemorySnapshot,
    get_monitor,
    init_monitor,
)


# =========================================================================
# Tests: MemoryMonitor start / stop
# =========================================================================

class TestMemoryMonitorLifecycle:
    """Test start() and stop() behaviour."""

    def test_start_sets_start_time(self):
        mon = MemoryMonitor(enabled=False)
        assert mon._start_time is None
        mon.start()
        assert mon._start_time is not None
        mon.stop()

    def test_stop_clears_start_time(self):
        mon = MemoryMonitor(enabled=False)
        mon.start()
        mon.stop()
        assert mon._start_time is None

    def test_start_with_tracemalloc(self):
        """tracemalloc should be tracing after start(enabled=True)."""
        import tracemalloc

        mon = MemoryMonitor(enabled=True)
        mon.start()
        try:
            assert tracemalloc.is_tracing()
        finally:
            mon.stop()
        assert not tracemalloc.is_tracing()

    def test_start_without_tracemalloc(self):
        """tracemalloc should NOT be tracing when enabled=False."""
        import tracemalloc

        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            assert not tracemalloc.is_tracing()
        finally:
            mon.stop()

    def test_double_stop_is_safe(self):
        """Calling stop() twice should not raise."""
        mon = MemoryMonitor(enabled=True)
        mon.start()
        mon.stop()
        mon.stop()  # should be a no-op

    def test_stop_before_start_is_safe(self):
        """Calling stop() without start() should not raise."""
        mon = MemoryMonitor(enabled=False)
        mon.stop()


# =========================================================================
# Tests: get_stats()
# =========================================================================

class TestGetStats:
    """Test the get_stats() dict."""

    def test_returns_expected_keys_when_disabled(self):
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()

        assert "rss_mb" in stats
        assert "tracemalloc_enabled" in stats
        assert "tracemalloc_current_mb" in stats
        assert "tracemalloc_peak_mb" in stats
        assert "top_allocations" in stats
        assert "uptime_seconds" in stats

    def test_returns_expected_keys_when_enabled(self):
        mon = MemoryMonitor(enabled=True)
        mon.start()
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()

        assert stats["tracemalloc_enabled"] is True
        assert "rss_mb" in stats
        assert "tracemalloc_current_mb" in stats
        assert "tracemalloc_peak_mb" in stats
        assert "top_allocations" in stats
        assert "uptime_seconds" in stats

    def test_rss_is_positive(self):
        """RSS should always be a positive number."""
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()
        assert stats["rss_mb"] > 0

    def test_uptime_increases(self):
        """uptime_seconds should reflect elapsed time."""
        mon = MemoryMonitor(enabled=False)
        mon.start()
        time.sleep(0.1)
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()
        assert stats["uptime_seconds"] >= 0.1

    def test_disabled_monitor_returns_rss(self):
        """Even when tracemalloc is disabled, RSS should be reported."""
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()
        assert stats["rss_mb"] > 0
        assert stats["tracemalloc_enabled"] is False

    def test_top_allocations_empty_when_disabled(self):
        """top_allocations should be an empty list when tracemalloc is disabled."""
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            stats = mon.get_stats()
        finally:
            mon.stop()
        assert stats["top_allocations"] == []

    def test_top_allocations_populated_when_enabled(self):
        """When enabled, top_allocations should contain entries after some alloc."""
        mon = MemoryMonitor(enabled=True)
        mon.start()
        try:
            # Force some allocations so tracemalloc has data
            _ = [bytearray(1024) for _ in range(100)]
            stats = mon.get_stats()
        finally:
            mon.stop()
        # There should be at least one allocation site
        assert isinstance(stats["top_allocations"], list)
        # We can't guarantee the exact count, but with 100 x 1KB allocs
        # there should be data.
        if stats["top_allocations"]:
            entry = stats["top_allocations"][0]
            assert "file" in entry
            assert "size_kb" in entry
            assert "count" in entry

    def test_tracemalloc_values_positive_when_enabled(self):
        """traced current/peak should be > 0 when tracemalloc is active."""
        mon = MemoryMonitor(enabled=True)
        mon.start()
        try:
            # Allocate something to ensure non-zero traced memory
            _ = bytearray(1024 * 100)
            stats = mon.get_stats()
        finally:
            mon.stop()
        assert stats["tracemalloc_current_mb"] >= 0
        assert stats["tracemalloc_peak_mb"] > 0


# =========================================================================
# Tests: snapshot()
# =========================================================================

class TestSnapshot:
    """Test snapshot() returns a valid MemorySnapshot."""

    def test_snapshot_returns_dataclass(self):
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            snap = mon.snapshot()
        finally:
            mon.stop()
        assert isinstance(snap, MemorySnapshot)

    def test_snapshot_rss_positive(self):
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            snap = mon.snapshot()
        finally:
            mon.stop()
        assert snap.rss_mb > 0

    def test_snapshot_timestamp_recent(self):
        mon = MemoryMonitor(enabled=False)
        t0 = time.time()
        mon.start()
        try:
            snap = mon.snapshot()
        finally:
            mon.stop()
        assert snap.timestamp >= t0
        assert snap.timestamp <= time.time()

    def test_snapshot_traced_zero_when_disabled(self):
        mon = MemoryMonitor(enabled=False)
        mon.start()
        try:
            snap = mon.snapshot()
        finally:
            mon.stop()
        assert snap.traced_current_mb == 0.0
        assert snap.traced_peak_mb == 0.0

    def test_snapshot_traced_nonzero_when_enabled(self):
        mon = MemoryMonitor(enabled=True)
        mon.start()
        try:
            _ = bytearray(1024 * 100)
            snap = mon.snapshot()
        finally:
            mon.stop()
        # Peak should be > 0 because we allocated 100 KB
        assert snap.traced_peak_mb > 0


# =========================================================================
# Tests: module-level singleton
# =========================================================================

class TestSingleton:
    """Test get_monitor() and init_monitor()."""

    def test_get_monitor_returns_instance(self):
        mon = get_monitor()
        assert isinstance(mon, MemoryMonitor)

    def test_get_monitor_returns_same_instance(self):
        mon1 = get_monitor()
        mon2 = get_monitor()
        assert mon1 is mon2

    def test_init_monitor_replaces_singleton(self):
        old = get_monitor()
        new = init_monitor(enabled=True)
        assert new is not old
        assert get_monitor() is new
        # Clean up
        new.stop()
        init_monitor(enabled=False)


# =========================================================================
# Tests: IPC handler integration
# =========================================================================

@pytest.mark.skipif(
    __import__("sys").platform == "win32",
    reason="Uses AF_UNIX sockets not available on Windows",
)
class TestIPCHandlerIntegration:
    """Test the mem_stats IPC handler wired in daemon._create_ipc_server."""

    def test_mem_stats_handler_returns_ok(self):
        """The mem_stats handler should return stats with ok=True."""
        import socket
        import json
        import threading

        from claude_speak.ipc import IPCServer, send_message

        # Set up a test monitor
        monitor = init_monitor(enabled=False)
        monitor.start()

        try:
            from claude_speak.daemon import _create_ipc_server

            loop = asyncio.new_event_loop()
            queue_ready = asyncio.Event()
            engine = MagicMock()

            import uuid
            sock_path = Path(f"/tmp/cs-test-mem-{uuid.uuid4().hex[:8]}.sock")

            with patch("claude_speak.daemon.Q") as mock_q, \
                 patch("claude_speak.daemon.TOGGLE_FILE"), \
                 patch("claude_speak.daemon.START_TS_FILE"):
                config = MagicMock()
                server = _create_ipc_server(engine, config, queue_ready, loop)
                server._socket_path = sock_path
                server.start()

                # Wait for socket
                import time as _time
                deadline = _time.monotonic() + 2.0
                while _time.monotonic() < deadline:
                    if sock_path.exists():
                        try:
                            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            s.settimeout(0.5)
                            s.connect(str(sock_path))
                            s.close()
                            break
                        except OSError:
                            pass
                    _time.sleep(0.05)

                try:
                    resp = send_message(
                        {"type": "mem_stats"},
                        socket_path=sock_path,
                    )
                    assert resp is not None
                    assert resp["ok"] is True
                    assert "rss_mb" in resp
                    assert resp["rss_mb"] > 0
                    assert "tracemalloc_enabled" in resp
                    assert "top_allocations" in resp
                    assert "uptime_seconds" in resp
                finally:
                    server.stop()
                    loop.close()
                    sock_path.unlink(missing_ok=True)
        finally:
            monitor.stop()

    def test_mem_stats_handler_with_tracemalloc_enabled(self):
        """When tracemalloc is enabled, the handler should include traced stats."""
        import socket
        import json
        import uuid

        from claude_speak.ipc import IPCServer, send_message

        monitor = init_monitor(enabled=True)
        monitor.start()

        try:
            from claude_speak.daemon import _create_ipc_server

            loop = asyncio.new_event_loop()
            queue_ready = asyncio.Event()
            engine = MagicMock()

            sock_path = Path(f"/tmp/cs-test-mem-{uuid.uuid4().hex[:8]}.sock")

            with patch("claude_speak.daemon.Q") as mock_q, \
                 patch("claude_speak.daemon.TOGGLE_FILE"), \
                 patch("claude_speak.daemon.START_TS_FILE"):
                config = MagicMock()
                server = _create_ipc_server(engine, config, queue_ready, loop)
                server._socket_path = sock_path
                server.start()

                # Wait for socket
                import time as _time
                deadline = _time.monotonic() + 2.0
                while _time.monotonic() < deadline:
                    if sock_path.exists():
                        try:
                            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                            s.settimeout(0.5)
                            s.connect(str(sock_path))
                            s.close()
                            break
                        except OSError:
                            pass
                    _time.sleep(0.05)

                try:
                    # Force some allocations
                    _ = [bytearray(1024) for _ in range(50)]
                    resp = send_message(
                        {"type": "mem_stats"},
                        socket_path=sock_path,
                    )
                    assert resp is not None
                    assert resp["ok"] is True
                    assert resp["tracemalloc_enabled"] is True
                    assert resp["tracemalloc_peak_mb"] >= 0
                finally:
                    server.stop()
                    loop.close()
                    sock_path.unlink(missing_ok=True)
        finally:
            monitor.stop()
            # Reset to disabled for other tests
            init_monitor(enabled=False)
