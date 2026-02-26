"""
Unit tests for claude_speak/permissions.py — macOS permission checks.

All external calls (subprocess, sounddevice) are mocked so tests run on any
platform without requiring actual hardware or macOS permissions.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from claude_speak.permissions import (
    CheckResult,
    check_accessibility,
    check_audio_input,
    check_audio_output,
    format_results,
    run_all_checks,
)


# ---------------------------------------------------------------------------
# Tests: check_audio_output
# ---------------------------------------------------------------------------


class TestCheckAudioOutput:
    """Tests for check_audio_output()."""

    def test_pass_with_device(self):
        """PASS when sounddevice lists a default output device."""
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [{"name": "Speakers", "max_output_channels": 2}]
        mock_sd.query_devices.side_effect = None
        # query_devices(kind="output") returns a dict
        def _query(kind=None):
            if kind == "output":
                return {"name": "MacBook Pro Speakers", "max_output_channels": 2}
            return [{"name": "MacBook Pro Speakers"}]
        mock_sd.query_devices = MagicMock(side_effect=_query)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_output()

        assert result.passed is True
        assert "MacBook Pro Speakers" in result.detail
        assert result.hint is None

    def test_fail_sounddevice_not_installed(self):
        """FAIL when sounddevice is not importable."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Force ImportError by removing the module
            import sys
            saved = sys.modules.pop("sounddevice", None)
            try:
                # We need to make import fail, patch builtins
                import builtins
                original_import = builtins.__import__

                def _mock_import(name, *args, **kwargs):
                    if name == "sounddevice":
                        raise ImportError("No module named 'sounddevice'")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", side_effect=_mock_import):
                    result = check_audio_output()
            finally:
                if saved is not None:
                    sys.modules["sounddevice"] = saved

        assert result.passed is False
        assert "not installed" in result.detail
        assert result.hint is not None

    def test_fail_device_error(self):
        """FAIL when sounddevice raises an exception querying devices."""
        mock_sd = MagicMock()
        mock_sd.query_devices = MagicMock(side_effect=OSError("No output device"))

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_output()

        assert result.passed is False
        assert "No output device" in result.detail


# ---------------------------------------------------------------------------
# Tests: check_audio_input
# ---------------------------------------------------------------------------


class TestCheckAudioInput:
    """Tests for check_audio_input()."""

    def test_pass_with_microphone(self):
        """PASS when sounddevice can open an input stream."""
        mock_sd = MagicMock()
        mock_stream = MagicMock()

        def _query(kind=None):
            if kind == "input":
                return {"name": "MacBook Pro Microphone", "max_input_channels": 1}
            return [{"name": "MacBook Pro Microphone"}]

        mock_sd.query_devices = MagicMock(side_effect=_query)
        mock_sd.InputStream = MagicMock(return_value=mock_stream)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input()

        assert result.passed is True
        assert "MacBook Pro Microphone" in result.detail
        mock_stream.start.assert_called_once()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_fail_permission_denied(self):
        """FAIL with helpful hint when microphone permission is denied."""
        mock_sd = MagicMock()

        def _query(kind=None):
            if kind == "input":
                return {"name": "MacBook Pro Microphone"}
            return []

        mock_sd.query_devices = MagicMock(side_effect=_query)
        mock_sd.InputStream = MagicMock(
            side_effect=OSError("permission denied: microphone access not allowed")
        )

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input()

        assert result.passed is False
        assert "permission" in result.detail.lower() or "denied" in result.detail.lower()
        assert result.hint is not None
        assert "Microphone" in result.hint

    def test_fail_no_device(self):
        """FAIL when no input device is available."""
        mock_sd = MagicMock()
        mock_sd.query_devices = MagicMock(
            side_effect=OSError("No default input device")
        )

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = check_audio_input()

        assert result.passed is False
        assert "No default input device" in result.detail

    def test_fail_sounddevice_not_installed(self):
        """FAIL when sounddevice is not importable."""
        import builtins
        import sys

        saved = sys.modules.pop("sounddevice", None)
        try:
            original_import = builtins.__import__

            def _mock_import(name, *args, **kwargs):
                if name == "sounddevice":
                    raise ImportError("No module named 'sounddevice'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=_mock_import):
                result = check_audio_input()
        finally:
            if saved is not None:
                sys.modules["sounddevice"] = saved

        assert result.passed is False
        assert "not installed" in result.detail


# ---------------------------------------------------------------------------
# Tests: check_accessibility
# ---------------------------------------------------------------------------


class TestCheckAccessibility:
    """Tests for check_accessibility()."""

    @patch("claude_speak.permissions._is_macos", return_value=True)
    @patch("claude_speak.permissions.subprocess.run")
    def test_pass_on_macos(self, mock_run, _mock_is_macos):
        """PASS when osascript successfully queries System Events."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["osascript", "-e", "..."],
            returncode=0,
            stdout="loginwindow\n",
            stderr="",
        )

        result = check_accessibility()

        assert result.passed is True
        assert "System Events access OK" in result.detail
        assert result.hint is None

    @patch("claude_speak.permissions._is_macos", return_value=True)
    @patch("claude_speak.permissions.subprocess.run")
    def test_fail_on_macos_no_permission(self, mock_run, _mock_is_macos):
        """FAIL when osascript is rejected (Accessibility not granted)."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["osascript", "-e", "..."],
            returncode=1,
            stdout="",
            stderr="execution error: Not authorized to send Apple events (-1743)",
        )

        result = check_accessibility()

        assert result.passed is False
        assert "Not authorized" in result.detail
        assert result.hint is not None
        assert "Accessibility" in result.hint

    @patch("claude_speak.permissions._is_macos", return_value=True)
    @patch("claude_speak.permissions.subprocess.run")
    def test_fail_on_macos_timeout(self, mock_run, _mock_is_macos):
        """FAIL when osascript times out."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="osascript", timeout=5.0)

        result = check_accessibility()

        assert result.passed is False
        assert "timed out" in result.detail
        assert result.hint is not None

    @patch("claude_speak.permissions._is_macos", return_value=True)
    @patch("claude_speak.permissions.subprocess.run")
    def test_fail_osascript_not_found(self, mock_run, _mock_is_macos):
        """N/A (passes) when osascript binary is not found (non-macOS PATH)."""
        mock_run.side_effect = FileNotFoundError("osascript not found")

        result = check_accessibility()

        assert result.passed is True
        assert "N/A" in result.detail

    @patch("claude_speak.permissions._is_macos", return_value=False)
    def test_not_macos_returns_na(self, _mock_is_macos):
        """On non-macOS platforms, return N/A and pass."""
        result = check_accessibility()

        assert result.passed is True
        assert "N/A" in result.detail
        assert "macOS only" in result.detail

    @patch("claude_speak.permissions._is_macos", return_value=True)
    @patch("claude_speak.permissions.subprocess.run")
    def test_fail_generic_exception(self, mock_run, _mock_is_macos):
        """FAIL on unexpected exceptions with helpful hint."""
        mock_run.side_effect = RuntimeError("unexpected error")

        result = check_accessibility()

        assert result.passed is False
        assert "unexpected error" in result.detail
        assert result.hint is not None


# ---------------------------------------------------------------------------
# Tests: run_all_checks
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    """Tests for run_all_checks()."""

    @patch("claude_speak.permissions.check_accessibility")
    @patch("claude_speak.permissions.check_audio_input")
    @patch("claude_speak.permissions.check_audio_output")
    def test_returns_three_results(self, mock_output, mock_input, mock_access):
        """run_all_checks returns one result per check."""
        mock_output.return_value = CheckResult("Audio output", True, "ok")
        mock_input.return_value = CheckResult("Audio input", True, "ok")
        mock_access.return_value = CheckResult("Accessibility", True, "ok")

        results = run_all_checks()

        assert len(results) == 3
        assert results[0].name == "Audio output"
        assert results[1].name == "Audio input"
        assert results[2].name == "Accessibility"

    @patch("claude_speak.permissions.check_accessibility")
    @patch("claude_speak.permissions.check_audio_input")
    @patch("claude_speak.permissions.check_audio_output")
    def test_all_pass(self, mock_output, mock_input, mock_access):
        mock_output.return_value = CheckResult("Audio output", True, "ok")
        mock_input.return_value = CheckResult("Audio input", True, "ok")
        mock_access.return_value = CheckResult("Accessibility", True, "ok")

        results = run_all_checks()
        assert all(r.passed for r in results)

    @patch("claude_speak.permissions.check_accessibility")
    @patch("claude_speak.permissions.check_audio_input")
    @patch("claude_speak.permissions.check_audio_output")
    def test_mixed_pass_fail(self, mock_output, mock_input, mock_access):
        mock_output.return_value = CheckResult("Audio output", True, "ok")
        mock_input.return_value = CheckResult("Audio input", True, "ok")
        mock_access.return_value = CheckResult(
            "Accessibility", False, "denied",
            hint="System Settings > Accessibility",
        )

        results = run_all_checks()
        assert results[0].passed is True
        assert results[1].passed is True
        assert results[2].passed is False


# ---------------------------------------------------------------------------
# Tests: format_results
# ---------------------------------------------------------------------------


class TestFormatResults:
    """Tests for format_results()."""

    def test_all_pass_output(self):
        """Formatted output for all-pass scenario."""
        results = [
            CheckResult("Audio output", True, "default: MacBook Pro Speakers"),
            CheckResult("Audio input", True, "default: MacBook Pro Microphone"),
            CheckResult("Accessibility", True, "System Events access OK"),
        ]

        output = format_results(results)

        assert "claude-speak permission check" in output
        assert "PASS" in output
        assert "FAIL" not in output
        assert "MacBook Pro Speakers" in output
        assert "MacBook Pro Microphone" in output
        assert "System Events access OK" in output

    def test_fail_output_includes_hint(self):
        """Formatted output for a failure includes the hint text."""
        results = [
            CheckResult("Audio output", True, "default: Speakers"),
            CheckResult("Audio input", True, "default: Microphone"),
            CheckResult(
                "Accessibility", False, "Not authorized",
                hint="System Settings > Privacy & Security > Accessibility\nAdd your terminal app",
            ),
        ]

        output = format_results(results)

        assert "FAIL" in output
        assert "Not authorized" in output
        assert "-> System Settings > Privacy & Security > Accessibility" in output
        assert "-> Add your terminal app" in output

    def test_multiple_failures(self):
        """Formatted output handles multiple failures."""
        results = [
            CheckResult("Audio output", True, "default: Speakers"),
            CheckResult(
                "Audio input", False, "permission denied",
                hint="System Settings > Microphone",
            ),
            CheckResult(
                "Accessibility", False, "Not authorized",
                hint="System Settings > Accessibility",
            ),
        ]

        output = format_results(results)

        # Count FAIL occurrences
        assert output.count("FAIL") == 2
        assert output.count("PASS") == 1

    def test_header_present(self):
        """Output starts with the header and separator."""
        results = [CheckResult("Audio output", True, "ok")]
        output = format_results(results)

        lines = output.split("\n")
        assert lines[0] == "claude-speak permission check"
        # Second line is the separator (unicode box-drawing chars)
        assert len(lines[1]) > 0
        assert lines[1] == "\u2500" * 36

    def test_alignment_dots(self):
        """Names are padded with dots for visual alignment."""
        results = [
            CheckResult("Audio output", True, "ok"),
            CheckResult("Audio input", True, "ok"),
            CheckResult("Accessibility", True, "ok"),
        ]

        output = format_results(results)

        # Each result line should contain dots for alignment
        for line in output.split("\n")[2:]:
            if line.strip() and ("PASS" in line or "FAIL" in line):
                assert "." in line

    def test_empty_results(self):
        """format_results handles empty list gracefully."""
        # max() on empty list would raise, so handle it
        results: list[CheckResult] = []
        # This should not raise. With no results, just show the header.
        output = format_results(results)
        assert "claude-speak permission check" in output
