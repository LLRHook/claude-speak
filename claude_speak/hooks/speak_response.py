#!/usr/bin/env python3
"""
speak_response — Python hook for Claude Code PostToolUse / Stop events.

Replaces the shell-based speak-response.sh with proper error handling,
type safety, and testability.  Reads hook input JSON from stdin, extracts
new assistant text from the JSONL transcript, strips markdown, writes to
the daemon queue directory, and signals the daemon via SIGUSR1.

Error handling: every failure path exits 0 so errors never propagate to
Claude Code.  Set CLAUDE_SPEAK_DEBUG=1 to log diagnostics to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (mirror the shell script defaults)
# ---------------------------------------------------------------------------

TOGGLE_FILE = Path.home() / ".claude-speak-enabled"
QUEUE_DIR = Path("/tmp/claude-speak-queue")
POS_FILE = Path("/tmp/claude-speak-pos")
PID_FILE = Path("/tmp/claude-speak-daemon.pid")
PERF_LOG = Path("/tmp/claude-speak-perf.log")
HOOK_LOCK = Path("/tmp/claude-speak-hook.lock")

DEBUG = bool(os.environ.get("CLAUDE_SPEAK_DEBUG", ""))
PERF_ENABLED = bool(os.environ.get("CLAUDE_SPEAK_PERF", ""))

logger = logging.getLogger("claude-speak-hook")


# ---------------------------------------------------------------------------
# Debug / perf helpers
# ---------------------------------------------------------------------------

def _debug(msg: str) -> None:
    """Write a debug message to stderr when CLAUDE_SPEAK_DEBUG is set."""
    if DEBUG:
        print(f"[claude-speak-hook] {msg}", file=sys.stderr, flush=True)


def _perf_now() -> float:
    """High-resolution timestamp for performance logging."""
    return time.monotonic()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class HookInput:
    """Parsed Claude Code hook input."""
    transcript_path: str
    session_id: str
    hook_event_name: str


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_hook_input(raw: str) -> HookInput | None:
    """Parse the JSON blob that Claude Code pipes to the hook on stdin.

    Returns None (rather than raising) on any parse failure — the hook must
    never block Claude Code.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        _debug(f"JSON parse error: {exc}")
        return None

    if not isinstance(data, dict):
        _debug(f"Expected JSON object, got {type(data).__name__}")
        return None

    transcript_path = data.get("transcript_path", "")
    session_id = data.get("session_id", "")
    hook_event_name = data.get("hook_event_name", "")

    if not transcript_path:
        _debug("transcript_path missing or empty")
        return None

    return HookInput(
        transcript_path=str(transcript_path),
        session_id=str(session_id),
        hook_event_name=str(hook_event_name),
    )


# ---------------------------------------------------------------------------
# Position tracking (atomic via mkdir lock — same strategy as shell version)
# ---------------------------------------------------------------------------

def _acquire_lock(lock_path: Path, timeout: float = 3.0, poll: float = 0.1) -> bool:
    """Acquire a directory-based lock.  Returns True on success.

    Uses ``mkdir`` atomicity (POSIX guarantee) — works on macOS which lacks
    ``flock`` on some filesystems.  Stale locks older than 10 s are removed.
    """
    # Clean stale lock
    if lock_path.is_dir():
        try:
            mtime = lock_path.stat().st_mtime
            if time.time() - mtime > 10:
                lock_path.rmdir()
        except OSError:
            pass

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            lock_path.mkdir()
            return True
        except FileExistsError:
            time.sleep(poll)
        except OSError as exc:
            _debug(f"Lock mkdir error: {exc}")
            return False
    _debug(f"Could not acquire lock after {timeout}s, giving up")
    return False


def _release_lock(lock_path: Path) -> None:
    """Release the directory-based lock."""
    try:
        lock_path.rmdir()
    except OSError:
        pass


def read_position(pos_file: Path, session_id: str) -> int:
    """Read the last-processed line position for *session_id*.

    Returns 0 if the file is missing, corrupt, or belongs to a different
    session.
    """
    try:
        text = pos_file.read_text(encoding="utf-8")
        lines = text.strip().split("\n")
        if len(lines) >= 2 and lines[0] == session_id:
            return int(lines[1])
    except (OSError, ValueError):
        pass
    return 0


def write_position(pos_file: Path, session_id: str, position: int) -> None:
    """Persist *position* for *session_id* to disk."""
    try:
        pos_file.write_text(f"{session_id}\n{position}\n", encoding="utf-8")
    except OSError as exc:
        _debug(f"Failed to write position file: {exc}")


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------

def extract_assistant_text(transcript_path: str, skip_lines: int) -> tuple[str, int]:
    """Read new assistant text from the JSONL transcript file.

    Args:
        transcript_path: Absolute path to the Claude Code JSONL transcript.
        skip_lines: Number of lines already processed.

    Returns:
        (extracted_text, total_line_count) — extracted_text may be empty if
        there is no new assistant content.
    """
    try:
        with open(transcript_path, encoding="utf-8") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        _debug(f"Could not read transcript: {exc}")
        return "", skip_lines

    total = len(all_lines)
    if total <= skip_lines:
        return "", total

    new_lines = all_lines[skip_lines:]
    texts: list[str] = []

    for line in new_lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "assistant":
            continue
        for block in obj.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)

    return "\n".join(texts), total


# ---------------------------------------------------------------------------
# Markdown stripping (light pass — mirrors the sed/perl logic in the shell)
#
# The full normalizer handles code blocks, tables, and lists downstream; here
# we only strip bold/italic markers, links, images, HRs, and HTML tags, then
# collapse blank lines and trim whitespace.
# ---------------------------------------------------------------------------

# Pre-compile regexes for performance
_RE_BOLD_ITALIC = re.compile(r"\*{1,3}([^*]*)\*{1,3}")
_RE_UNDERSCORE_EMPH = re.compile(r"_{1,3}([^_]*)_{1,3}")
_RE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_RE_HR = re.compile(r"^[-*_]{3,}$", re.MULTILINE)
_RE_HTML_TAG = re.compile(r"<[^>]*>")
_RE_BLANK_LINES = re.compile(r"\n\s*\n")
_RE_LEADING_WS = re.compile(r"^[ \t]+", re.MULTILINE)
_RE_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting from *text*.

    This is a lightweight pass matching what the shell script's ``sed``
    commands do.  The full normalizer pipeline in the daemon handles deeper
    transformations (code blocks, tables, lists, etc.).
    """
    text = _RE_BOLD_ITALIC.sub(r"\1", text)
    text = _RE_UNDERSCORE_EMPH.sub(r"\1", text)
    text = _RE_IMAGE.sub(r"\1", text)      # images before links (![...])
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_HR.sub("", text)
    text = _RE_HTML_TAG.sub("", text)

    # Collapse blank lines and trim each line
    text = _RE_LEADING_WS.sub("", text)
    text = _RE_TRAILING_WS.sub("", text)

    # Remove fully blank lines
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Queue + daemon signalling
# ---------------------------------------------------------------------------

def enqueue_text(text: str, queue_dir: Path = QUEUE_DIR) -> Path | None:
    """Write *text* into the daemon queue directory.

    Returns the written file path, or None on failure.
    """
    try:
        queue_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(queue_dir, 0o700)
    except OSError as exc:
        _debug(f"Failed to create queue dir: {exc}")
        return None

    timestamp = f"{time.time():.6f}"
    path = queue_dir / f"{timestamp}.txt"
    try:
        path.write_text(text, encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        _debug(f"Failed to write queue file: {exc}")
        return None
    return path


def signal_daemon(pid_file: Path = PID_FILE) -> bool:
    """Send SIGUSR1 to the daemon so it processes the queue immediately.

    Returns True if the signal was sent successfully.
    """
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGUSR1)
        return True
    except (OSError, ValueError):
        return False


def send_to_daemon(text: str) -> bool:
    """Try to send *text* to the daemon via Unix domain socket (fast path).

    Returns True if the daemon acknowledged the message, False on any
    failure (daemon not running, socket error, etc.).  Callers should
    fall back to the file queue when this returns False.
    """
    try:
        from claude_speak.ipc import send_message

        response = send_message({"type": "speak", "text": text}, timeout=2.0)
        if response is not None and response.get("ok"):
            _debug("Sent via IPC socket (fast path)")
            return True
    except Exception as exc:
        _debug(f"IPC send failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(stdin_data: str | None = None) -> int:
    """Execute the hook logic.  Always returns 0 (success).

    Args:
        stdin_data: If provided, use this instead of reading from sys.stdin.
                    Useful for testing.
    """
    t_start = _perf_now() if PERF_ENABLED else 0.0

    # --- Gate ---
    if not TOGGLE_FILE.exists():
        return 0

    # --- Serialize ---
    if not _acquire_lock(HOOK_LOCK):
        return 0

    try:
        return _run_locked(stdin_data, t_start)
    finally:
        _release_lock(HOOK_LOCK)


def _run_locked(stdin_data: str | None, t_start: float) -> int:
    """Core logic executed while holding the hook lock."""

    # --- Read stdin ---
    if stdin_data is None:
        try:
            stdin_data = sys.stdin.read()
        except Exception as exc:
            _debug(f"Failed to read stdin: {exc}")
            return 0

    # --- Parse hook input ---
    hook = parse_hook_input(stdin_data)
    if hook is None:
        return 0

    if not Path(hook.transcript_path).is_file():
        _debug(f"Transcript file missing: {hook.transcript_path}")
        return 0

    t_parse = _perf_now() if PERF_ENABLED else 0.0

    # --- Position tracking ---
    last_pos = read_position(POS_FILE, hook.session_id)

    # --- For Stop hook, wait for transcript flush ---
    if hook.hook_event_name == "Stop":
        time.sleep(0.3)

    # --- Extract new assistant text ---
    text, total_lines = extract_assistant_text(hook.transcript_path, last_pos)

    t_extract = _perf_now() if PERF_ENABLED else 0.0

    # --- Update position (even if no text — avoids re-scanning) ---
    write_position(POS_FILE, hook.session_id, total_lines)

    if not text:
        _debug("No new assistant text found")
        return 0

    # --- Strip markdown ---
    clean = strip_markdown(text)
    if not clean:
        _debug("Text was empty after markdown stripping")
        return 0

    t_strip = _perf_now() if PERF_ENABLED else 0.0

    # --- Send to daemon (IPC socket fast path, file queue fallback) ---
    if not send_to_daemon(clean):
        # IPC unavailable — fall back to file queue + SIGUSR1
        _debug("IPC unavailable, falling back to file queue")
        queue_file = enqueue_text(clean, queue_dir=QUEUE_DIR)
        if queue_file is None:
            return 0
        signal_daemon(pid_file=PID_FILE)

    # --- Performance logging ---
    if PERF_ENABLED:
        t_end = _perf_now()
        new_lines = total_lines - last_pos
        try:
            PERF_LOG.open("a").write(
                f"[hook-perf] parse={t_parse - t_start:.3f}s "
                f"extract={t_extract - t_parse:.3f}s "
                f"strip={t_strip - t_extract:.3f}s "
                f"write={t_end - t_strip:.3f}s "
                f"TOTAL={t_end - t_start:.3f}s "
                f"lines={new_lines} chars={len(clean)}\n"
            )
        except OSError:
            pass

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point when invoked as ``python -m claude_speak.hooks.speak_response``."""
    try:
        sys.exit(run())
    except Exception as exc:
        # Absolute safety net — never propagate to Claude Code
        _debug(f"Unhandled exception: {exc}")
        sys.exit(0)


if __name__ == "__main__":
    main()
