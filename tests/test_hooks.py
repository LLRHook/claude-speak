"""
Comprehensive tests for claude_speak/hooks/speak_response.py.

Covers: JSON parsing, text extraction, markdown stripping, position tracking,
queue enqueue, daemon signalling, error handling, and backward compatibility.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_speak.hooks.speak_response import (
    HookInput,
    parse_hook_input,
    read_position,
    write_position,
    extract_assistant_text,
    strip_markdown,
    enqueue_text,
    signal_daemon,
    run,
    _acquire_lock,
    _release_lock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_queue(tmp_path: Path) -> Path:
    """Return a temporary queue directory."""
    q = tmp_path / "queue"
    q.mkdir()
    return q


@pytest.fixture
def tmp_pos_file(tmp_path: Path) -> Path:
    """Return a path for a temporary position file."""
    return tmp_path / "pos"


@pytest.fixture
def tmp_pid_file(tmp_path: Path) -> Path:
    """Return a path for a temporary PID file."""
    return tmp_path / "pid"


@pytest.fixture
def tmp_lock(tmp_path: Path) -> Path:
    """Return a path for a temporary lock directory."""
    return tmp_path / "lock"


@pytest.fixture
def transcript_file(tmp_path: Path) -> Path:
    """Create a sample JSONL transcript file with assistant messages."""
    p = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({"type": "human", "message": {"content": [{"type": "text", "text": "Hello"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Hi there!"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "How can I help?"}]}}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _make_hook_json(transcript_path: str, session_id: str = "sess-1", event: str = "PostToolUse") -> str:
    """Build a hook-input JSON string."""
    return json.dumps({
        "transcript_path": transcript_path,
        "session_id": session_id,
        "hook_event_name": event,
    })


# =========================================================================
# Tests: parse_hook_input
# =========================================================================

class TestParseHookInput:
    """Test JSON parsing of hook input from stdin."""

    def test_valid_input(self, tmp_path: Path):
        raw = _make_hook_json("/tmp/transcript.jsonl")
        result = parse_hook_input(raw)
        assert result is not None
        assert result.transcript_path == "/tmp/transcript.jsonl"
        assert result.session_id == "sess-1"
        assert result.hook_event_name == "PostToolUse"

    def test_valid_stop_event(self):
        raw = _make_hook_json("/tmp/t.jsonl", event="Stop")
        result = parse_hook_input(raw)
        assert result is not None
        assert result.hook_event_name == "Stop"

    def test_missing_transcript_path(self):
        raw = json.dumps({"session_id": "s1", "hook_event_name": "Stop"})
        assert parse_hook_input(raw) is None

    def test_empty_transcript_path(self):
        raw = json.dumps({"transcript_path": "", "session_id": "s1", "hook_event_name": "Stop"})
        assert parse_hook_input(raw) is None

    def test_invalid_json(self):
        assert parse_hook_input("{bad json}") is None

    def test_empty_string(self):
        assert parse_hook_input("") is None

    def test_json_array(self):
        assert parse_hook_input("[1, 2, 3]") is None

    def test_json_null(self):
        assert parse_hook_input("null") is None

    def test_extra_fields_ignored(self):
        raw = json.dumps({
            "transcript_path": "/tmp/t.jsonl",
            "session_id": "s1",
            "hook_event_name": "PostToolUse",
            "extra_field": "ignored",
        })
        result = parse_hook_input(raw)
        assert result is not None
        assert result.transcript_path == "/tmp/t.jsonl"

    def test_numeric_session_id_coerced_to_string(self):
        raw = json.dumps({
            "transcript_path": "/tmp/t.jsonl",
            "session_id": 12345,
            "hook_event_name": "PostToolUse",
        })
        result = parse_hook_input(raw)
        assert result is not None
        assert result.session_id == "12345"

    def test_missing_optional_fields_default_empty(self):
        """session_id and hook_event_name can be empty but transcript_path cannot."""
        raw = json.dumps({"transcript_path": "/tmp/t.jsonl"})
        result = parse_hook_input(raw)
        assert result is not None
        assert result.session_id == ""
        assert result.hook_event_name == ""


# =========================================================================
# Tests: extract_assistant_text
# =========================================================================

class TestExtractAssistantText:
    """Test extraction of assistant text from JSONL transcripts."""

    def test_extract_all(self, transcript_file: Path):
        text, total = extract_assistant_text(str(transcript_file), 0)
        assert "Hi there!" in text
        assert "How can I help?" in text
        assert total == 3

    def test_skip_already_read(self, transcript_file: Path):
        # Skip first two lines (human + first assistant), only get second assistant
        text, total = extract_assistant_text(str(transcript_file), 2)
        assert "Hi there!" not in text
        assert "How can I help?" in text
        assert total == 3

    def test_skip_all_lines(self, transcript_file: Path):
        text, total = extract_assistant_text(str(transcript_file), 3)
        assert text == ""
        assert total == 3

    def test_skip_past_end(self, transcript_file: Path):
        text, total = extract_assistant_text(str(transcript_file), 100)
        assert text == ""
        assert total == 3

    def test_missing_file(self):
        text, total = extract_assistant_text("/nonexistent/path.jsonl", 0)
        assert text == ""
        assert total == 0

    def test_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        text, total = extract_assistant_text(str(p), 0)
        assert text == ""
        assert total == 0

    def test_malformed_json_lines(self, tmp_path: Path):
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n{also bad\n", encoding="utf-8")
        text, total = extract_assistant_text(str(p), 0)
        assert text == ""
        assert total == 2

    def test_non_assistant_types_skipped(self, tmp_path: Path):
        p = tmp_path / "mixed.jsonl"
        lines = [
            json.dumps({"type": "system", "message": {"content": [{"type": "text", "text": "System msg"}]}}),
            json.dumps({"type": "human", "message": {"content": [{"type": "text", "text": "User msg"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Bot reply"}]}}),
        ]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text, total = extract_assistant_text(str(p), 0)
        assert text == "Bot reply"
        assert "System msg" not in text
        assert "User msg" not in text

    def test_multiple_content_blocks(self, tmp_path: Path):
        p = tmp_path / "multi.jsonl"
        line = json.dumps({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Part one."},
                    {"type": "tool_use", "name": "bash"},
                    {"type": "text", "text": "Part two."},
                ]
            }
        })
        p.write_text(line + "\n", encoding="utf-8")
        text, total = extract_assistant_text(str(p), 0)
        assert "Part one." in text
        assert "Part two." in text


# =========================================================================
# Tests: strip_markdown
# =========================================================================

class TestStripMarkdown:
    """Test markdown stripping (light pass matching the shell sed logic)."""

    def test_bold(self):
        assert strip_markdown("This is **bold** text") == "This is bold text"

    def test_italic_asterisk(self):
        assert strip_markdown("This is *italic* text") == "This is italic text"

    def test_bold_italic(self):
        assert strip_markdown("This is ***important*** text") == "This is important text"

    def test_underscore_emphasis(self):
        assert strip_markdown("This is _italic_ text") == "This is italic text"

    def test_double_underscore(self):
        assert strip_markdown("This is __bold__ text") == "This is bold text"

    def test_triple_underscore(self):
        assert strip_markdown("This is ___both___ text") == "This is both text"

    def test_link(self):
        assert strip_markdown("Click [here](https://example.com)") == "Click here"

    def test_image(self):
        assert strip_markdown("![alt text](image.png)") == "alt text"

    def test_horizontal_rule(self):
        result = strip_markdown("above\n---\nbelow")
        assert "---" not in result
        assert "above" in result
        assert "below" in result

    def test_html_tags(self):
        assert strip_markdown("Hello <b>world</b>!") == "Hello world!"

    def test_blank_line_removal(self):
        result = strip_markdown("line one\n\n\nline two")
        lines = result.split("\n")
        assert all(ln.strip() for ln in lines)

    def test_whitespace_trimming(self):
        result = strip_markdown("   leading\n   trailing   \n")
        for line in result.split("\n"):
            assert line == line.strip()

    def test_empty_after_stripping(self):
        assert strip_markdown("***") == ""

    def test_already_clean(self):
        assert strip_markdown("Just plain text.") == "Just plain text."

    def test_mixed_markdown(self):
        text = "**Bold** and *italic* with a [link](http://x.com) and <br> tag."
        result = strip_markdown(text)
        assert "**" not in result
        assert "*" not in result  # italic markers gone
        assert "[" not in result
        assert "<br>" not in result
        assert "Bold" in result
        assert "italic" in result
        assert "link" in result


# =========================================================================
# Tests: position tracking
# =========================================================================

class TestPositionTracking:
    """Test atomic position file read/write for transcript tracking."""

    def test_write_and_read(self, tmp_pos_file: Path):
        write_position(tmp_pos_file, "sess-1", 42)
        assert read_position(tmp_pos_file, "sess-1") == 42

    def test_different_session_resets(self, tmp_pos_file: Path):
        write_position(tmp_pos_file, "sess-1", 42)
        assert read_position(tmp_pos_file, "sess-2") == 0

    def test_missing_file_returns_zero(self, tmp_pos_file: Path):
        assert read_position(tmp_pos_file, "any") == 0

    def test_corrupt_file_returns_zero(self, tmp_pos_file: Path):
        tmp_pos_file.write_text("garbage", encoding="utf-8")
        assert read_position(tmp_pos_file, "any") == 0

    def test_overwrite_position(self, tmp_pos_file: Path):
        write_position(tmp_pos_file, "sess-1", 10)
        write_position(tmp_pos_file, "sess-1", 50)
        assert read_position(tmp_pos_file, "sess-1") == 50

    def test_write_failure_on_readonly(self, tmp_path: Path):
        """write_position should not raise on write failure."""
        bad_path = tmp_path / "no_such_dir" / "pos"
        # Should not raise — just logs debug
        write_position(bad_path, "sess-1", 10)


# =========================================================================
# Tests: lock acquisition
# =========================================================================

class TestLockAcquisition:
    """Test directory-based atomic lock mechanism."""

    def test_acquire_and_release(self, tmp_lock: Path):
        assert _acquire_lock(tmp_lock, timeout=1.0)
        assert tmp_lock.is_dir()
        _release_lock(tmp_lock)
        assert not tmp_lock.exists()

    def test_double_acquire_fails(self, tmp_lock: Path):
        assert _acquire_lock(tmp_lock, timeout=0.5)
        # Second acquire should fail (timeout quickly)
        assert not _acquire_lock(tmp_lock, timeout=0.3, poll=0.1)
        _release_lock(tmp_lock)

    def test_stale_lock_removed(self, tmp_lock: Path):
        """Locks older than 10s should be cleaned automatically."""
        tmp_lock.mkdir()
        # Backdate the lock directory's mtime by 15 seconds
        old_time = time.time() - 15
        os.utime(tmp_lock, (old_time, old_time))
        # Should succeed because the stale lock gets removed
        assert _acquire_lock(tmp_lock, timeout=1.0)
        _release_lock(tmp_lock)

    def test_release_nonexistent_lock(self, tmp_lock: Path):
        """Releasing a non-existent lock should not raise."""
        _release_lock(tmp_lock)  # no-op, should not raise


# =========================================================================
# Tests: enqueue_text
# =========================================================================

class TestEnqueueText:
    """Test writing text to the daemon queue directory."""

    def test_enqueue_creates_file(self, tmp_queue: Path):
        path = enqueue_text("Hello world", queue_dir=tmp_queue)
        assert path is not None
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "Hello world"
        assert path.suffix == ".txt"

    def test_enqueue_creates_directory(self, tmp_path: Path):
        q = tmp_path / "new_queue"
        path = enqueue_text("Test", queue_dir=q)
        assert path is not None
        assert q.is_dir()

    def test_enqueue_filename_is_timestamp(self, tmp_queue: Path):
        path = enqueue_text("Test", queue_dir=tmp_queue)
        assert path is not None
        stem = path.stem
        # Stem should be a valid float (timestamp)
        ts = float(stem)
        assert ts > 0

    def test_enqueue_ordering(self, tmp_queue: Path):
        p1 = enqueue_text("First", queue_dir=tmp_queue)
        time.sleep(0.01)  # tiny delay for unique timestamps
        p2 = enqueue_text("Second", queue_dir=tmp_queue)
        assert p1 is not None and p2 is not None
        files = sorted(tmp_queue.glob("*.txt"))
        assert files[0].read_text(encoding="utf-8") == "First"
        assert files[1].read_text(encoding="utf-8") == "Second"

    def test_enqueue_failure_returns_none(self):
        # Using a path that cannot be created
        bad_path = Path("/proc/nonexistent/queue") if sys.platform != "win32" else Path("Z:\\nonexistent\\queue")
        result = enqueue_text("Test", queue_dir=bad_path)
        assert result is None


# =========================================================================
# Tests: signal_daemon
# =========================================================================

class TestSignalDaemon:
    """Test SIGUSR1 signalling to the daemon."""

    def test_signal_missing_pid_file(self, tmp_pid_file: Path):
        assert signal_daemon(pid_file=tmp_pid_file) is False

    def test_signal_invalid_pid_content(self, tmp_pid_file: Path):
        tmp_pid_file.write_text("not_a_number", encoding="utf-8")
        assert signal_daemon(pid_file=tmp_pid_file) is False

    def test_signal_nonexistent_pid(self, tmp_pid_file: Path):
        tmp_pid_file.write_text("999999999", encoding="utf-8")
        assert signal_daemon(pid_file=tmp_pid_file) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1 not available on Windows")
    @patch("os.kill")
    def test_signal_success(self, mock_kill: MagicMock, tmp_pid_file: Path):
        tmp_pid_file.write_text("12345", encoding="utf-8")
        assert signal_daemon(pid_file=tmp_pid_file) is True
        mock_kill.assert_called_once_with(12345, signal.SIGUSR1)


# =========================================================================
# Tests: error handling for malformed input
# =========================================================================

class TestErrorHandling:
    """Test that the hook handles all error conditions gracefully."""

    def test_run_toggle_file_missing(self, tmp_path: Path):
        """When toggle file is absent, run() exits immediately with 0."""
        with patch.object(
            __import__("claude_speak.hooks.speak_response", fromlist=["TOGGLE_FILE"]),
            "TOGGLE_FILE",
            tmp_path / "nonexistent",
        ):
            assert run(stdin_data="{}") == 0

    def test_run_empty_stdin(self, tmp_path: Path):
        """Empty stdin produces no output and no crash."""
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        with patch.multiple(
            "claude_speak.hooks.speak_response",
            TOGGLE_FILE=toggle,
            HOOK_LOCK=lock,
        ):
            assert run(stdin_data="") == 0

    def test_run_malformed_json_stdin(self, tmp_path: Path):
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        with patch.multiple(
            "claude_speak.hooks.speak_response",
            TOGGLE_FILE=toggle,
            HOOK_LOCK=lock,
        ):
            assert run(stdin_data="{bad json!!}") == 0

    def test_run_transcript_file_missing(self, tmp_path: Path):
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        pos = tmp_path / "pos"
        stdin_data = _make_hook_json("/no/such/transcript.jsonl")
        with patch.multiple(
            "claude_speak.hooks.speak_response",
            TOGGLE_FILE=toggle,
            HOOK_LOCK=lock,
            POS_FILE=pos,
        ):
            assert run(stdin_data=stdin_data) == 0


# =========================================================================
# Tests: full run() integration (end-to-end with mocked filesystem paths)
# =========================================================================

@pytest.mark.skipif(sys.platform == "win32", reason="SIGUSR1 not available on Windows")
class TestRunIntegration:
    """End-to-end test of run() with a real transcript and queue."""

    def test_full_pipeline(self, tmp_path: Path, transcript_file: Path):
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        pos = tmp_path / "pos"
        queue = tmp_path / "queue"
        pid = tmp_path / "pid"
        # Write our own PID so signal_daemon doesn't crash
        pid.write_text(str(os.getpid()), encoding="utf-8")

        stdin_data = _make_hook_json(str(transcript_file), session_id="s1")

        # We need to handle the SIGUSR1 that will be sent to ourselves
        received_signal = []
        old_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, lambda s, f: received_signal.append(s))

        try:
            with patch.multiple(
                "claude_speak.hooks.speak_response",
                TOGGLE_FILE=toggle,
                HOOK_LOCK=lock,
                POS_FILE=pos,
                QUEUE_DIR=queue,
                PID_FILE=pid,
            ):
                result = run(stdin_data=stdin_data)
        finally:
            signal.signal(signal.SIGUSR1, old_handler)

        assert result == 0

        # Queue should have a file with assistant text
        files = sorted(queue.glob("*.txt"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "Hi there!" in content
        assert "How can I help?" in content

        # Position should be updated
        assert read_position(pos, "s1") == 3

        # Signal should have been received
        assert len(received_signal) == 1

    def test_incremental_reads(self, tmp_path: Path):
        """Run the hook twice; second run should only get new lines."""
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        pos = tmp_path / "pos"
        queue = tmp_path / "queue"
        pid = tmp_path / "pid"

        # Transcript with one assistant line
        transcript = tmp_path / "t.jsonl"
        line1 = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "First reply."}]}})
        transcript.write_text(line1 + "\n", encoding="utf-8")

        # Suppress SIGUSR1 side effects
        old_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, lambda s, f: None)

        try:
            with patch.multiple(
                "claude_speak.hooks.speak_response",
                TOGGLE_FILE=toggle,
                HOOK_LOCK=lock,
                POS_FILE=pos,
                QUEUE_DIR=queue,
                PID_FILE=pid,
            ):
                run(stdin_data=_make_hook_json(str(transcript), session_id="s1"))

                # Check queue after first run
                files = sorted(queue.glob("*.txt"))
                assert len(files) == 1
                assert "First reply" in files[0].read_text(encoding="utf-8")

                # Append a second assistant message
                line2 = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "Second reply."}]}})
                with open(transcript, "a", encoding="utf-8") as fh:
                    fh.write(line2 + "\n")

                # Second run
                run(stdin_data=_make_hook_json(str(transcript), session_id="s1"))

                files = sorted(queue.glob("*.txt"))
                assert len(files) == 2
                second_content = files[1].read_text(encoding="utf-8")
                assert "Second reply" in second_content
                assert "First reply" not in second_content
        finally:
            signal.signal(signal.SIGUSR1, old_handler)

    def test_no_new_text_produces_no_queue_file(self, tmp_path: Path):
        """When there are no new assistant messages, nothing is enqueued."""
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        pos = tmp_path / "pos"
        queue = tmp_path / "queue"
        pid = tmp_path / "pid"

        transcript = tmp_path / "t.jsonl"
        # Only human messages
        line = json.dumps({"type": "human", "message": {"content": [{"type": "text", "text": "Hi"}]}})
        transcript.write_text(line + "\n", encoding="utf-8")

        with patch.multiple(
            "claude_speak.hooks.speak_response",
            TOGGLE_FILE=toggle,
            HOOK_LOCK=lock,
            POS_FILE=pos,
            QUEUE_DIR=queue,
            PID_FILE=pid,
        ):
            result = run(stdin_data=_make_hook_json(str(transcript), session_id="s1"))

        assert result == 0
        # Queue dir may or may not exist, but should have no files
        files = list(queue.glob("*.txt")) if queue.exists() else []
        assert len(files) == 0


# =========================================================================
# Tests: backward compatibility
# =========================================================================

class TestBackwardCompatibility:
    """Verify the Python hook maintains compatibility with the shell version."""

    @pytest.mark.skipif(sys.platform == "win32", reason="macOS backward-compat paths")
    def test_same_queue_directory(self):
        """Python hook uses the same queue path as the shell script."""
        from claude_speak.hooks.speak_response import QUEUE_DIR
        assert str(QUEUE_DIR) == "/tmp/claude-speak-queue"

    @pytest.mark.skipif(sys.platform == "win32", reason="macOS backward-compat paths")
    def test_same_pos_file(self):
        from claude_speak.hooks.speak_response import POS_FILE
        assert str(POS_FILE) == "/tmp/claude-speak-pos"

    @pytest.mark.skipif(sys.platform == "win32", reason="macOS backward-compat paths")
    def test_same_pid_file(self):
        from claude_speak.hooks.speak_response import PID_FILE
        assert str(PID_FILE) == "/tmp/claude-speak-daemon.pid"

    def test_same_toggle_file(self):
        from claude_speak.hooks.speak_response import TOGGLE_FILE
        assert str(TOGGLE_FILE) == str(Path.home() / ".claude-speak-enabled")

    @pytest.mark.skipif(sys.platform == "win32", reason="macOS backward-compat paths")
    def test_same_lock_path(self):
        from claude_speak.hooks.speak_response import HOOK_LOCK
        assert str(HOOK_LOCK) == "/tmp/claude-speak-hook.lock"

    def test_position_file_format(self, tmp_pos_file: Path):
        """Position file format: line 1 = session_id, line 2 = position."""
        write_position(tmp_pos_file, "my-session", 42)
        content = tmp_pos_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert lines[0] == "my-session"
        assert lines[1] == "42"

    def test_queue_file_naming(self, tmp_queue: Path):
        """Queue files are named <timestamp>.txt — same as shell version."""
        path = enqueue_text("test", queue_dir=tmp_queue)
        assert path is not None
        assert path.suffix == ".txt"
        # Timestamp format: digits, a dot, more digits
        parts = path.stem.split(".")
        assert len(parts) == 2
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_hook_input_format(self):
        """Verify the expected Claude Code hook input fields are handled."""
        raw = json.dumps({
            "transcript_path": "/path/to/transcript",
            "session_id": "abc-123",
            "hook_event_name": "PostToolUse",
        })
        result = parse_hook_input(raw)
        assert result == HookInput(
            transcript_path="/path/to/transcript",
            session_id="abc-123",
            hook_event_name="PostToolUse",
        )

    def test_stop_event_waits(self, tmp_path: Path, transcript_file: Path):
        """Stop events should include a 0.3s wait (transcript flush delay)."""
        toggle = tmp_path / "toggle"
        toggle.touch()
        lock = tmp_path / "lock"
        pos = tmp_path / "pos"
        queue = tmp_path / "queue"
        pid = tmp_path / "pid"

        stdin_data = _make_hook_json(str(transcript_file), session_id="s1", event="Stop")

        with patch.multiple(
            "claude_speak.hooks.speak_response",
            TOGGLE_FILE=toggle,
            HOOK_LOCK=lock,
            POS_FILE=pos,
            QUEUE_DIR=queue,
            PID_FILE=pid,
        ), patch("claude_speak.hooks.speak_response.time") as mock_time:
            # We need time.time() and time.monotonic() to still work
            mock_time.time.return_value = 1000000.123456
            mock_time.monotonic.return_value = 0.0
            mock_time.sleep = MagicMock()

            run(stdin_data=stdin_data)

            # Verify that sleep(0.3) was called for the Stop event
            mock_time.sleep.assert_called_with(0.3)
