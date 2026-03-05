#!/usr/bin/env python3
"""
daemon_start -- SessionStart hook for Claude Code.

Starts the claude-speak daemon when a Claude Code session begins.
Replaces hooks/daemon-start.sh with cross-platform Python.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def run() -> int:
    """Start the daemon if enabled and not already running. Always returns 0."""
    toggle_file = Path.home() / ".claude-speak-enabled"
    if not toggle_file.exists():
        return 0

    # Import platform paths
    from claude_speak.platform.paths import pid_file

    pid_path = pid_file()

    # Check if already running
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            # Check if process is alive
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return 0  # Already running
            else:
                os.kill(pid, 0)
                return 0  # Already running
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass  # Stale PID file

    # Launch daemon as detached subprocess
    import subprocess

    project_root = str(Path(__file__).resolve().parent.parent.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "")
    if project_root not in env["PYTHONPATH"]:
        env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env["PYTHONPATH"] else "")

    devnull = os.open(os.devnull, os.O_RDWR)
    try:
        kwargs: dict = {
            "stdin": devnull,
            "stdout": devnull,
            "stderr": devnull,
            "env": env,
        }
        if sys.platform == "win32":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen(
            [sys.executable, "-m", "claude_speak.daemon"],
            **kwargs,
        )
    finally:
        os.close(devnull)

    return 0


def main() -> None:
    try:
        sys.exit(run())
    except Exception:
        sys.exit(0)  # Never propagate errors to Claude Code


if __name__ == "__main__":
    main()
