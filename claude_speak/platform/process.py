"""Cross-platform process management utilities for claude-speak."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


def acquire_file_lock(path: Path) -> IO | None:
    """Acquire an exclusive, non-blocking file lock.

    Uses fcntl.flock() on Unix and msvcrt.locking() on Windows.
    Returns the open file descriptor (keep it open to hold the lock),
    or None if the lock could not be acquired.
    """
    if sys.platform == "win32":
        import msvcrt

        # On Windows, use a two-step approach:
        # 1. Ensure the file exists with at least one byte (msvcrt.locking
        #    requires the byte range to exist).
        # 2. Open in "r+" mode (no truncation) to avoid conflicts with
        #    existing locks, then lock byte 0.
        # If the file doesn't exist yet, create it with a sentinel byte.
        if not path.exists():
            path.write_text(" ")
        fd = open(path, "r+")
        try:
            fd.seek(0)
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
            return fd
        except OSError:
            try:
                fd.close()
            except OSError:
                pass
            return None
    else:
        import fcntl

        fd = open(path, "w+")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError:
            fd.close()
            return None


# ---------------------------------------------------------------------------
# Process status
# ---------------------------------------------------------------------------


def is_process_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running.

    Uses os.kill(pid, 0) on Unix and kernel32.OpenProcess on Windows
    (with a psutil fallback).
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _is_process_alive_win32(pid)
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we don't have permission to signal it
            return True


def _is_process_alive_win32(pid: int) -> bool:
    """Windows implementation of is_process_alive using kernel32."""
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except (OSError, AttributeError):
        # Fallback to psutil if ctypes approach fails
        try:
            import psutil

            return psutil.pid_exists(pid)
        except ImportError:
            logger.warning("Cannot check process status: ctypes failed and psutil not installed")
            return False


# ---------------------------------------------------------------------------
# Process control
# ---------------------------------------------------------------------------


def terminate_process(pid: int) -> None:
    """Send a graceful termination signal to a process.

    SIGTERM on Unix, psutil.Process.terminate() on Windows.
    Raises ProcessLookupError if the process does not exist.
    """
    if sys.platform == "win32":
        import psutil

        proc = psutil.Process(pid)
        proc.terminate()
    else:
        import signal

        os.kill(pid, signal.SIGTERM)


def force_kill_process(pid: int) -> None:
    """Forcefully kill a process.

    SIGKILL on Unix, psutil.Process.kill() on Windows.
    Raises ProcessLookupError if the process does not exist.
    """
    if sys.platform == "win32":
        import psutil

        proc = psutil.Process(pid)
        proc.kill()
    else:
        import signal

        os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Process discovery
# ---------------------------------------------------------------------------


def find_processes_by_name(pattern: str) -> list[int]:
    """Find PIDs of processes whose command line matches *pattern*.

    Uses ``pgrep -f`` on Unix and ``psutil.process_iter()`` on Windows.
    Returns a (possibly empty) list of integer PIDs.
    """
    if sys.platform == "win32":
        return _find_processes_win32(pattern)
    else:
        return _find_processes_unix(pattern)


def _find_processes_unix(pattern: str) -> list[int]:
    """Unix implementation using pgrep."""
    pids: list[int] = []
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line:
                pids.append(int(line))
    except Exception:
        pass
    return pids


def _find_processes_win32(pattern: str) -> list[int]:
    """Windows implementation using psutil."""
    pids: list[int] = []
    try:
        import psutil

        pattern_lower = pattern.lower()
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                info = proc.info
                # Check process name
                name = info.get("name", "") or ""
                if pattern_lower in name.lower():
                    pids.append(info["pid"])
                    continue
                # Check command line
                cmdline = info.get("cmdline") or []
                cmdline_str = " ".join(cmdline).lower()
                if pattern_lower in cmdline_str:
                    pids.append(info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except ImportError:
        logger.warning("psutil not installed; cannot find processes by name on Windows")
    return pids


# ---------------------------------------------------------------------------
# Detached process launch
# ---------------------------------------------------------------------------


def launch_detached(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Launch a fully detached subprocess.

    On Unix, uses ``start_new_session=True``.
    On Windows, uses ``CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS``
    and redirects stdin/stdout/stderr to ``os.devnull``.
    """
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        if sys.platform == "win32":
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            subprocess.Popen(
                cmd,
                stdin=devnull_fd,
                stdout=devnull_fd,
                stderr=devnull_fd,
                creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                env=env,
            )
        else:
            subprocess.Popen(
                cmd,
                stdin=devnull_fd,
                stdout=devnull_fd,
                stderr=devnull_fd,
                start_new_session=True,
                env=env,
            )
    finally:
        os.close(devnull_fd)
