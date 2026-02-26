"""
Memory monitoring for claude-speak daemon.

Provides tracemalloc-based memory tracking and RSS reporting.
tracemalloc is only activated when explicitly enabled (via env var),
keeping overhead at zero for normal operation.

Usage:
    from claude_speak.memmon import get_monitor

    monitor = get_monitor()
    monitor.start()
    ...
    stats = monitor.get_stats()
    snapshot = monitor.snapshot()
    ...
    monitor.stop()
"""

from __future__ import annotations

import resource
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""

    rss_mb: float
    traced_current_mb: float
    traced_peak_mb: float
    timestamp: float


class MemoryMonitor:
    """On-demand memory monitor with optional tracemalloc support.

    Args:
        enabled: If True, tracemalloc will be started for detailed
                 allocation tracking.  If False, only RSS is reported.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled
        self._start_time: float | None = None

    # -- lifecycle --

    def start(self) -> None:
        """Begin monitoring.  Starts tracemalloc if enabled."""
        self._start_time = time.monotonic()
        if self._enabled:
            tracemalloc.start(10)  # 10 frames deep

    def stop(self) -> None:
        """Stop monitoring and release tracemalloc resources."""
        if self._enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._start_time = None

    # -- queries --

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _get_rss_mb() -> float:
        """Return current RSS in megabytes.

        On macOS, ru_maxrss is in bytes.
        """
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 * 1024)

    def snapshot(self) -> MemorySnapshot:
        """Capture a point-in-time memory snapshot."""
        rss = self._get_rss_mb()
        traced_current = 0.0
        traced_peak = 0.0
        if self._enabled and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            traced_current = current / (1024 * 1024)
            traced_peak = peak / (1024 * 1024)
        return MemorySnapshot(
            rss_mb=round(rss, 2),
            traced_current_mb=round(traced_current, 2),
            traced_peak_mb=round(traced_peak, 2),
            timestamp=time.time(),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return a dict of memory statistics suitable for IPC / display.

        Always includes ``rss_mb`` and ``uptime_seconds``.
        tracemalloc fields are only populated when enabled.
        """
        snap = self.snapshot()
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.monotonic() - self._start_time

        stats: dict[str, Any] = {
            "rss_mb": snap.rss_mb,
            "tracemalloc_enabled": self._enabled,
            "tracemalloc_current_mb": snap.traced_current_mb,
            "tracemalloc_peak_mb": snap.traced_peak_mb,
            "top_allocations": [],
            "uptime_seconds": round(uptime, 1),
        }

        if self._enabled and tracemalloc.is_tracing():
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")[:5]
                stats["top_allocations"] = [
                    {
                        "file": str(stat.traceback),
                        "size_kb": round(stat.size / 1024, 1),
                        "count": stat.count,
                    }
                    for stat in top_stats
                ]
            except Exception:
                # tracemalloc can occasionally raise if state is inconsistent
                pass

        return stats


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_monitor: MemoryMonitor | None = None


def get_monitor() -> MemoryMonitor:
    """Return the module-level MemoryMonitor singleton.

    Creates one (disabled) on first access.  The daemon's startup code
    should call ``init_monitor(enabled=...)`` before this.
    """
    global _monitor
    if _monitor is None:
        _monitor = MemoryMonitor(enabled=False)
    return _monitor


def init_monitor(enabled: bool = False) -> MemoryMonitor:
    """Create (or replace) the module-level MemoryMonitor singleton.

    Args:
        enabled: Whether to activate tracemalloc.

    Returns:
        The newly created MemoryMonitor instance.
    """
    global _monitor
    _monitor = MemoryMonitor(enabled=enabled)
    return _monitor
