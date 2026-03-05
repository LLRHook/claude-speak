#!/usr/bin/env python3
"""
daemon_restart -- UserPromptSubmit hook for Claude Code.

Detects "restart daemon" phrases and restarts the daemon automatically.
Exits with code 2 to block the prompt from being processed by Claude.
Replaces hooks/daemon-restart.sh with cross-platform Python.
"""
from __future__ import annotations

import json
import re
import sys
import time

_RESTART_PATTERN = re.compile(
    r'^\s*(restart\s+(the\s+)?d[ae]+mon|d[ae]+mon\s+restart)\s*[.!?]*\s*$',
    re.IGNORECASE,
)


def run(stdin_data: str | None = None) -> int:
    """Check if prompt is a restart request. Returns 2 to block, 0 otherwise."""
    if stdin_data is None:
        try:
            stdin_data = sys.stdin.read()
        except Exception:
            return 0

    try:
        data = json.loads(stdin_data)
    except (json.JSONDecodeError, TypeError):
        return 0

    prompt = data.get("prompt", "")
    if not prompt or not _RESTART_PATTERN.match(prompt):
        return 0

    # Stop daemon
    try:
        from claude_speak.ipc import send_message
        send_message({"type": "stop"}, timeout=2.0)
    except Exception:
        pass
    try:
        from claude_speak.daemon import stop_daemon
        stop_daemon()
    except Exception:
        pass

    time.sleep(1)

    # Restart daemon
    try:
        from claude_speak.hooks.daemon_start import run as start_daemon
        start_daemon()
    except Exception:
        pass

    time.sleep(1)
    print("Daemon restarted.", file=sys.stderr)
    return 2  # Block prompt from being processed


def main() -> None:
    try:
        sys.exit(run())
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
