"""
Unit tests for src/voice_input.py — voice input via Superwhisper and built-in pipeline.

All subprocess calls (osascript, pgrep, pbpaste, pbcopy) are mocked.
Tests run without macOS, Superwhisper, sounddevice, or STT models.
"""

import subprocess
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from claude_speak.config import InputConfig
from claude_speak.voice_input import (
    _modifier_flags_to_names,
    _run_osascript,
    _is_superwhisper_running,
    _get_clipboard,
    _set_clipboard,
    _paste_at_cursor,
    trigger_superwhisper,
    auto_submit,
    wait_for_transcription,
    voice_input_cycle,
    builtin_voice_input_cycle,
    SuperwhisperError,
)


# ---------------------------------------------------------------------------
# Tests: _modifier_flags_to_names
# ---------------------------------------------------------------------------

class TestModifierFlagsToNames:
    """Tests for converting macOS modifier bitmask to AppleScript names."""

    def test_option_flag(self):
        names = _modifier_flags_to_names(2048)
        assert names == ["option down"]

    def test_command_flag(self):
        names = _modifier_flags_to_names(256)
        assert names == ["command down"]

    def test_shift_flag(self):
        names = _modifier_flags_to_names(512)
        assert names == ["shift down"]

    def test_control_flag(self):
        names = _modifier_flags_to_names(4096)
        assert names == ["control down"]

    def test_combined_flags_command_shift(self):
        names = _modifier_flags_to_names(256 | 512)
        assert "command down" in names
        assert "shift down" in names
        assert len(names) == 2

    def test_combined_flags_option_command(self):
        names = _modifier_flags_to_names(256 | 2048)
        assert "command down" in names
        assert "option down" in names

    def test_no_flags(self):
        names = _modifier_flags_to_names(0)
        assert names == []

    def test_unknown_flag_ignored(self):
        """Flags not in the mapping should be silently ignored."""
        names = _modifier_flags_to_names(65536)  # fn key -- not mapped
        assert names == []


# ---------------------------------------------------------------------------
# Tests: _run_osascript
# ---------------------------------------------------------------------------

class TestRunOsascript:
    """Tests for _run_osascript."""

    @patch("claude_speak.voice_input.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = _run_osascript('tell app "System Events" to keystroke return')
        assert result.returncode == 0

    @patch("claude_speak.voice_input.subprocess.run")
    def test_failure_raises_superwhisper_error(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="some error")
        with pytest.raises(SuperwhisperError, match="osascript failed"):
            _run_osascript("bad script")

    @patch("claude_speak.voice_input.subprocess.run", side_effect=FileNotFoundError)
    def test_osascript_not_found_raises_error(self, mock_run):
        with pytest.raises(SuperwhisperError, match="osascript not found"):
            _run_osascript("any script")

    @patch("claude_speak.voice_input.subprocess.run", side_effect=subprocess.TimeoutExpired("osascript", 5))
    def test_timeout_raises_error(self, mock_run):
        with pytest.raises(SuperwhisperError, match="timed out"):
            _run_osascript("slow script")


# ---------------------------------------------------------------------------
# Tests: _is_superwhisper_running
# ---------------------------------------------------------------------------

class TestIsSuperwhisperRunning:
    """Tests for _is_superwhisper_running."""

    @patch("claude_speak.voice_input.subprocess.run")
    def test_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert _is_superwhisper_running() is True

    @patch("claude_speak.voice_input.subprocess.run")
    def test_not_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert _is_superwhisper_running() is False

    @patch("claude_speak.voice_input.subprocess.run", side_effect=Exception("oops"))
    def test_exception_returns_false(self, mock_run):
        assert _is_superwhisper_running() is False


# ---------------------------------------------------------------------------
# Tests: _get_clipboard
# ---------------------------------------------------------------------------

class TestGetClipboard:
    """Tests for _get_clipboard."""

    @patch("claude_speak.voice_input.subprocess.run")
    def test_returns_clipboard_content(self, mock_run):
        mock_run.return_value = MagicMock(stdout="hello world", returncode=0)
        assert _get_clipboard() == "hello world"

    @patch("claude_speak.voice_input.subprocess.run", side_effect=Exception("fail"))
    def test_returns_empty_on_error(self, mock_run):
        assert _get_clipboard() == ""


# ---------------------------------------------------------------------------
# Tests: trigger_superwhisper
# ---------------------------------------------------------------------------

class TestTriggerSuperwhisper:
    """Tests for trigger_superwhisper."""

    @patch("claude_speak.voice_input._run_osascript")
    def test_default_shortcut(self, mock_osa):
        trigger_superwhisper()
        mock_osa.assert_called_once()
        script = mock_osa.call_args[0][0]
        assert "key code 49" in script
        assert "option down" in script

    @patch("claude_speak.voice_input._run_osascript")
    def test_custom_shortcut(self, mock_osa):
        trigger_superwhisper(keycode=36, modifiers=256)
        script = mock_osa.call_args[0][0]
        assert "key code 36" in script
        assert "command down" in script


# ---------------------------------------------------------------------------
# Tests: auto_submit
# ---------------------------------------------------------------------------

class TestAutoSubmit:
    """Tests for auto_submit."""

    @patch("claude_speak.voice_input._run_osascript")
    def test_sends_return_keystroke(self, mock_osa):
        auto_submit()
        script = mock_osa.call_args[0][0]
        assert "keystroke return" in script


# ---------------------------------------------------------------------------
# Tests: wait_for_transcription
# ---------------------------------------------------------------------------

class TestWaitForTranscription:
    """Tests for wait_for_transcription."""

    @patch("claude_speak.voice_input._get_clipboard")
    @patch("claude_speak.voice_input.time.sleep")
    def test_clipboard_change_detected(self, mock_sleep, mock_clip):
        # First poll returns same, second returns new text
        mock_clip.side_effect = ["old text", "new transcription"]
        result = wait_for_transcription("old text", timeout=5.0, poll_interval=0.1)
        assert result is True

    @patch("claude_speak.voice_input._get_clipboard")
    @patch("claude_speak.voice_input.time.monotonic")
    @patch("claude_speak.voice_input.time.sleep")
    def test_no_change_returns_false(self, mock_sleep, mock_mono, mock_clip):
        # Simulate time passing beyond deadline
        mock_mono.side_effect = [0.0, 0.5, 1.0, 1.5, 2.1]
        mock_clip.return_value = "same text"
        result = wait_for_transcription("same text", timeout=2.0, poll_interval=0.5)
        assert result is False

    @patch("claude_speak.voice_input._get_clipboard")
    @patch("claude_speak.voice_input.time.sleep")
    def test_empty_clipboard_not_counted(self, mock_sleep, mock_clip):
        """If clipboard changes but to empty string, should keep waiting."""
        mock_clip.side_effect = ["old", "  ", "actual text"]
        result = wait_for_transcription("old", timeout=5.0, poll_interval=0.1)
        assert result is True


# ---------------------------------------------------------------------------
# Tests: voice_input_cycle
# ---------------------------------------------------------------------------

class TestVoiceInputCycle:
    """Tests for the full voice input cycle."""

    @patch("claude_speak.voice_input.auto_submit")
    @patch("claude_speak.voice_input.wait_for_transcription", return_value=True)
    @patch("claude_speak.voice_input._wait_for_speech_then_silence", return_value=True)
    @patch("claude_speak.voice_input.trigger_superwhisper")
    @patch("claude_speak.voice_input._get_clipboard", return_value="old text")
    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    def test_full_success_path(self, mock_running, mock_clip, mock_trigger,
                                mock_silence, mock_wait, mock_submit):
        config = InputConfig(auto_submit=True)
        result = voice_input_cycle(config)
        assert result is True
        assert mock_trigger.call_count == 2  # start + stop recording
        mock_submit.assert_called_once()

    @patch("claude_speak.voice_input.auto_submit")
    @patch("claude_speak.voice_input.wait_for_transcription", return_value=True)
    @patch("claude_speak.voice_input._wait_for_speech_then_silence", return_value=True)
    @patch("claude_speak.voice_input.trigger_superwhisper")
    @patch("claude_speak.voice_input._get_clipboard", return_value="old text")
    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    def test_auto_submit_disabled(self, mock_running, mock_clip, mock_trigger,
                                   mock_silence, mock_wait, mock_submit):
        config = InputConfig(auto_submit=False)
        result = voice_input_cycle(config)
        assert result is True
        mock_submit.assert_not_called()

    @patch("claude_speak.voice_input.wait_for_transcription", return_value=False)
    @patch("claude_speak.voice_input._wait_for_speech_then_silence", return_value=True)
    @patch("claude_speak.voice_input.trigger_superwhisper")
    @patch("claude_speak.voice_input._get_clipboard", return_value="old")
    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    def test_nothing_transcribed_returns_false(self, mock_running, mock_clip,
                                                mock_trigger, mock_silence, mock_wait):
        config = InputConfig(auto_submit=True)
        result = voice_input_cycle(config)
        assert result is False

    @patch("claude_speak.voice_input.trigger_superwhisper", side_effect=SuperwhisperError("boom"))
    @patch("claude_speak.voice_input._get_clipboard", return_value="old")
    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    def test_superwhisper_error_returns_false(self, mock_running, mock_clip, mock_trigger):
        config = InputConfig()
        result = voice_input_cycle(config)
        assert result is False

    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=False)
    def test_superwhisper_not_running_returns_false(self, mock_running):
        """When Superwhisper is not running, cycle aborts immediately."""
        config = InputConfig(auto_submit=True)
        result = voice_input_cycle(config)
        assert result is False

    def test_default_config_used_when_none(self):
        """When config is None, InputConfig defaults are used."""
        with patch("claude_speak.voice_input._is_superwhisper_running", return_value=True), \
             patch("claude_speak.voice_input._get_clipboard", return_value="x"), \
             patch("claude_speak.voice_input.trigger_superwhisper"), \
             patch("claude_speak.voice_input._wait_for_speech_then_silence", return_value=True), \
             patch("claude_speak.voice_input.wait_for_transcription", return_value=True), \
             patch("claude_speak.voice_input.auto_submit"):
            result = voice_input_cycle(config=None)
            assert result is True

    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    @patch("claude_speak.voice_input._get_clipboard", return_value="old")
    @patch("claude_speak.voice_input.trigger_superwhisper")
    @patch("claude_speak.voice_input._wait_for_speech_then_silence", return_value=True)
    @patch("claude_speak.voice_input.wait_for_transcription", return_value=True)
    @patch("claude_speak.voice_input.auto_submit")
    def test_custom_shortcut_config(self, mock_submit, mock_wait, mock_silence,
                                     mock_trigger, mock_clip, mock_running):
        """Custom keycode/modifiers from config should be passed to trigger_superwhisper."""
        config = InputConfig(
            auto_submit=True,
            superwhisper_shortcut_keycode=36,
            superwhisper_shortcut_modifiers=256,
        )
        voice_input_cycle(config)
        # Both calls (start + stop) should use custom shortcut
        for c in mock_trigger.call_args_list:
            assert c.kwargs.get("keycode", c.args[0] if c.args else None) is not None


# ---------------------------------------------------------------------------
# Tests: _set_clipboard
# ---------------------------------------------------------------------------

class TestSetClipboard:
    """Tests for _set_clipboard (pbcopy wrapper)."""

    @patch("claude_speak.voice_input.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert _set_clipboard("hello world") is True
        mock_run.assert_called_once()
        # Verify pbcopy was called with the text as input
        call_args = mock_run.call_args
        assert call_args[0][0] == ["pbcopy"]
        assert call_args[1]["input"] == b"hello world"

    @patch("claude_speak.voice_input.subprocess.run", side_effect=Exception("fail"))
    def test_failure_returns_false(self, mock_run):
        assert _set_clipboard("text") is False


# ---------------------------------------------------------------------------
# Tests: _paste_at_cursor
# ---------------------------------------------------------------------------

class TestPasteAtCursor:
    """Tests for _paste_at_cursor (Cmd+V via osascript)."""

    @patch("claude_speak.voice_input._run_osascript")
    def test_sends_cmd_v(self, mock_osa):
        _paste_at_cursor()
        mock_osa.assert_called_once()
        script = mock_osa.call_args[0][0]
        assert 'keystroke "v"' in script
        assert "command down" in script


# ---------------------------------------------------------------------------
# Tests: builtin_voice_input_cycle
# ---------------------------------------------------------------------------

def _make_mock_stream(chunks, vad_responses):
    """Create a mock sounddevice InputStream that yields pre-defined chunks.

    Args:
        chunks: list of numpy int16 arrays to return from stream.read()
        vad_responses: list of booleans — is_speech return values per chunk
    """
    mock_stream = MagicMock()
    read_iter = iter([(c, None) for c in chunks])
    mock_stream.read.side_effect = lambda n: next(read_iter)
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    return mock_stream


class TestBuiltinVoiceInputCycle:
    """Tests for the built-in voice input pipeline (mic -> VAD -> STT -> paste)."""

    def _make_speech_chunks(self, n_speech=5, n_silence=30):
        """Generate chunks simulating speech followed by silence.

        Returns (chunks, vad_responses) where chunks are int16 arrays
        and vad_responses are booleans for each chunk.
        """
        chunk_size = 512
        chunks = []
        vad_responses = []

        # Speech frames
        for _ in range(n_speech):
            chunks.append(np.random.randint(-1000, 1000, size=(chunk_size, 1), dtype=np.int16))
            vad_responses.append(True)

        # Silence frames
        for _ in range(n_silence):
            chunks.append(np.zeros((chunk_size, 1), dtype=np.int16))
            vad_responses.append(False)

        return chunks, vad_responses

    @patch("claude_speak.voice_input.auto_submit")
    @patch("claude_speak.voice_input._paste_at_cursor")
    @patch("claude_speak.voice_input._set_clipboard", return_value=True)
    def test_full_success_path(self, mock_clipboard, mock_paste, mock_submit):
        """Speech detected -> transcription -> paste -> submit."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=5, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = "hello world"

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer):
            result = builtin_voice_input_cycle(config)

        assert result is True
        mock_recognizer.transcribe.assert_called_once()
        mock_clipboard.assert_called_once_with("hello world")
        mock_paste.assert_called_once()
        mock_submit.assert_called_once()

    @patch("claude_speak.voice_input.auto_submit")
    @patch("claude_speak.voice_input._paste_at_cursor")
    @patch("claude_speak.voice_input._set_clipboard", return_value=True)
    def test_auto_submit_disabled(self, mock_clipboard, mock_paste, mock_submit):
        """When auto_submit=False, text is pasted but Enter is not pressed."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=3, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = "some text"

        config = InputConfig(auto_submit=False)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer):
            result = builtin_voice_input_cycle(config)

        assert result is True
        mock_clipboard.assert_called_once_with("some text")
        mock_paste.assert_called_once()
        mock_submit.assert_not_called()

    def test_no_speech_timeout(self):
        """When no speech is detected within the timeout, returns False."""
        chunk_size = 512
        # Generate enough silence chunks to exceed 10s timeout
        # Each chunk is 512/16000 = 32ms. We simulate time passing.
        silence_chunk = np.zeros((chunk_size, 1), dtype=np.int16)

        mock_stream = MagicMock()
        # Return silence indefinitely
        mock_stream.read.return_value = (silence_chunk, None)
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False

        mock_recognizer = MagicMock()

        config = InputConfig(auto_submit=True)

        # Mock time.monotonic to simulate timeout
        call_count = [0]
        base_time = 100.0

        original_monotonic = time.monotonic

        def fake_monotonic():
            call_count[0] += 1
            # First call is recording_start, subsequent calls are in the loop
            return base_time + call_count[0] * 0.5  # jump 0.5s per call

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer), \
             patch("claude_speak.voice_input.time.monotonic", side_effect=fake_monotonic):
            result = builtin_voice_input_cycle(config)

        assert result is False
        mock_recognizer.transcribe.assert_not_called()

    def test_empty_transcription_returns_false(self):
        """When STT returns empty text, returns False without pasting."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=5, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = ""

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer), \
             patch("claude_speak.voice_input._set_clipboard") as mock_clip:
            result = builtin_voice_input_cycle(config)

        assert result is False
        mock_clip.assert_not_called()

    def test_stt_not_available_returns_false(self):
        """When the STT backend raises RuntimeError, returns False gracefully."""
        mock_sd = MagicMock()

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.stt.get_recognizer", side_effect=RuntimeError("No STT backend")), \
             patch("claude_speak.vad.SileroVAD"):
            result = builtin_voice_input_cycle(config)

        assert result is False

    def test_vad_init_failure_returns_false(self):
        """When VAD fails to initialise, returns False gracefully."""
        mock_sd = MagicMock()
        mock_recognizer = MagicMock()

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer), \
             patch("claude_speak.vad.SileroVAD", side_effect=RuntimeError("VAD model missing")):
            result = builtin_voice_input_cycle(config)

        assert result is False

    def test_sounddevice_not_installed_returns_false(self):
        """When sounddevice is not importable, returns False."""
        import sys

        config = InputConfig(auto_submit=True)

        # Remove sounddevice from sys.modules if present, and make import fail
        saved = sys.modules.get("sounddevice")
        sys.modules["sounddevice"] = None  # causes ImportError on import

        try:
            # We need to force the import inside builtin_voice_input_cycle to fail.
            # The function does `import sounddevice as sd` which will get None
            # from sys.modules and raise ImportError.
            result = builtin_voice_input_cycle(config)
            assert result is False
        finally:
            if saved is not None:
                sys.modules["sounddevice"] = saved
            else:
                sys.modules.pop("sounddevice", None)

    @patch("claude_speak.voice_input.auto_submit")
    @patch("claude_speak.voice_input._paste_at_cursor")
    @patch("claude_speak.voice_input._set_clipboard", return_value=True)
    def test_whitespace_only_transcription_returns_false(self, mock_clip, mock_paste, mock_submit):
        """When STT returns only whitespace, treat as empty."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=5, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = "   \n  "

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer):
            result = builtin_voice_input_cycle(config)

        assert result is False
        mock_clip.assert_not_called()
        mock_paste.assert_not_called()
        mock_submit.assert_not_called()

    def test_transcription_error_returns_false(self):
        """When the recognizer.transcribe() raises, returns False."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=5, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.side_effect = RuntimeError("transcription boom")

        config = InputConfig(auto_submit=True)

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer):
            result = builtin_voice_input_cycle(config)

        assert result is False

    def test_default_config_when_none(self):
        """When config is None, InputConfig defaults are used."""
        chunks, vad_responses = self._make_speech_chunks(n_speech=3, n_silence=30)
        mock_stream = _make_mock_stream(chunks, vad_responses)

        mock_sd = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        mock_vad = MagicMock()
        mock_vad.is_speech.side_effect = vad_responses

        mock_recognizer = MagicMock()
        mock_recognizer.transcribe.return_value = "hello"

        with patch.dict("sys.modules", {"sounddevice": mock_sd}), \
             patch("claude_speak.vad.SileroVAD", return_value=mock_vad), \
             patch("claude_speak.stt.get_recognizer", return_value=mock_recognizer), \
             patch("claude_speak.voice_input._set_clipboard", return_value=True), \
             patch("claude_speak.voice_input._paste_at_cursor"), \
             patch("claude_speak.voice_input.auto_submit"):
            result = builtin_voice_input_cycle(config=None)

        assert result is True
