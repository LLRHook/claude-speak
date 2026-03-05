#!/usr/bin/env python3
"""
daemon_stop -- SessionEnd hook for Claude Code.

Stops the claude-speak daemon when a Claude Code session ends.
Replaces hooks/daemon-stop.sh with cross-platform Python.
"""
from __future__ import annotations

import sys


def run() -> int:
    """Stop the daemon. Always returns 0."""
    try:
        from claude_speak.ipc import send_message
        resp = send_message({"type": "stop"}, timeout=2.0)
        if resp and resp.get("ok"):
            return 0
    except Exception:
        pass

    # Fallback: PID-based stop
    try:
        from claude_speak.daemon import stop_daemon
        stop_daemon()
    except Exception:
        pass

    return 0


def main() -> None:
    try:
        sys.exit(run())
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
