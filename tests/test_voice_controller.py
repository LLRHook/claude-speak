"""
Unit tests for src/voice_controller.py — VoiceController orchestrator.

Mocks WakeWordListener, voice_input, chimes, and queue to isolate controller logic.
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claude_speak.config import Config, WakeWordConfig, InputConfig, AudioConfig
from claude_speak.voice_controller import VoiceController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(wakeword_enabled=True, chimes=True, auto_submit=True,
                     tts_stop_callback=None):
    config = Config(
        wakeword=WakeWordConfig(enabled=wakeword_enabled),
        input=InputConfig(auto_submit=auto_submit),
        audio=AudioConfig(chimes=chimes),
    )
    return VoiceController(config=config, tts_stop_callback=tts_stop_callback)


# ---------------------------------------------------------------------------
# Tests: start/stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for start/stop lifecycle."""

    @patch("claude_speak.voice_controller.WakeWordListener")
    def test_start_returns_true(self, mock_ww_cls):
        mock_ww = MagicMock()
        mock_ww.start.return_value = True
        mock_ww_cls.return_value = mock_ww
        vc = _make_controller()
        result = vc.start()
        assert result is True
        assert vc.is_running is True

    @patch("claude_speak.voice_controller.WakeWordListener")
    def test_start_when_already_running_returns_false(self, mock_ww_cls):
        mock_ww = MagicMock()
        mock_ww.start.return_value = True
        mock_ww_cls.return_value = mock_ww
        vc = _make_controller()
        vc.start()
        result = vc.start()
        assert result is False

    @patch("claude_speak.voice_controller.WakeWordListener")
    def test_stop_clears_running_flag(self, mock_ww_cls):
        mock_ww = MagicMock()
        mock_ww.start.return_value = True
        mock_ww_cls.return_value = mock_ww
        vc = _make_controller()
        vc.start()
        vc.stop()
        assert vc.is_running is False
        assert vc._wakeword_listener is None

    def test_start_with_wakeword_disabled(self):
        vc = _make_controller(wakeword_enabled=False)
        result = vc.start()
        assert result is True
        assert vc._wakeword_listener is None


# ---------------------------------------------------------------------------
# Tests: handle_stop
# ---------------------------------------------------------------------------

class TestHandleStop:
    """Tests for handle_stop."""

    @patch("claude_speak.voice_controller.Q")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_handle_stop_clears_queue_and_calls_callback(self, mock_mute, mock_q):
        stop_cb = MagicMock()
        vc = _make_controller(tts_stop_callback=stop_cb)
        vc.handle_stop("stop")
        mock_q.clear.assert_called_once()
        stop_cb.assert_called_once()
        mock_mute.unlink.assert_called_once_with(missing_ok=True)

    @patch("claude_speak.voice_controller.Q")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_handle_stop_no_callback(self, mock_mute, mock_q):
        vc = _make_controller(tts_stop_callback=None)
        # Should not raise
        vc.handle_stop("stop")
        mock_q.clear.assert_called_once()

    @patch("claude_speak.voice_controller.Q")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_handle_stop_callback_exception_caught(self, mock_mute, mock_q):
        stop_cb = MagicMock(side_effect=RuntimeError("boom"))
        vc = _make_controller(tts_stop_callback=stop_cb)
        # Should not raise
        vc.handle_stop("stop")
        stop_cb.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _on_wake_word (context-aware)
# ---------------------------------------------------------------------------

class TestOnWakeWord:
    """Tests for _on_wake_word — mute toggle vs voice input."""

    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_toggles_mute_when_playing(self, mock_mute, mock_playing):
        """If TTS is playing and not muted, wake word should mute."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        stop_cb = MagicMock()
        vc = _make_controller(tts_stop_callback=stop_cb)
        vc._running = True
        vc._on_wake_word()
        mock_mute.touch.assert_called_once()
        stop_cb.assert_called_once()

    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_unmutes_when_muted(self, mock_mute, mock_playing):
        """If TTS is playing AND muted, wake word should unmute."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = True
        vc = _make_controller()
        vc._running = True
        vc._on_wake_word()
        mock_mute.unlink.assert_called_once_with(missing_ok=True)

    @patch("claude_speak.voice_controller.voice_input_cycle")
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    def test_starts_voice_input_when_idle(self, mock_playing, mock_vic):
        """If TTS is not playing, wake word should start voice input."""
        mock_playing.exists.return_value = False
        mock_vic.return_value = True
        vc = _make_controller()
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        # Call directly — _on_wake_word spawns a thread
        vc._on_wake_word()
        # Give the thread a moment
        time.sleep(0.2)
        mock_vic.assert_called_once()

    def test_does_not_act_when_not_running(self):
        vc = _make_controller()
        vc._running = False
        # Should return without doing anything — no error
        vc._on_wake_word()


# ---------------------------------------------------------------------------
# Tests: _handle_wake (voice input cycle + lock)
# ---------------------------------------------------------------------------

class TestHandleWake:
    """Tests for _handle_wake — input lock and voice input cycle."""

    @patch("claude_speak.voice_controller.play_error_chime")
    @patch("claude_speak.voice_controller.play_ack_chime")
    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    def test_successful_cycle(self, mock_vic, mock_ack, mock_err):
        vc = _make_controller(chimes=True)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc._handle_wake()
        assert result is True
        mock_vic.assert_called_once()
        mock_ack.assert_called_once()
        assert vc._voice_input_active is False

    @patch("claude_speak.voice_controller.play_error_chime")
    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=False)
    def test_failed_cycle_plays_error_chime(self, mock_vic, mock_err):
        vc = _make_controller(chimes=True)
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc._handle_wake()
        assert result is False
        mock_err.assert_called_once()

    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    def test_double_trigger_prevented_by_lock(self, mock_vic):
        """Second concurrent call should return False due to _input_lock."""
        vc = _make_controller()
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True

        # Acquire the lock before calling _handle_wake
        vc._input_lock.acquire()
        result = vc._handle_wake()
        assert result is False
        mock_vic.assert_not_called()
        vc._input_lock.release()

    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    def test_wakeword_paused_during_cycle(self, mock_vic):
        """Wake word listener should be paused during voice input."""
        vc = _make_controller()
        vc._running = True
        mock_ww = MagicMock()
        mock_ww.is_running = True
        vc._wakeword_listener = mock_ww
        vc._handle_wake()
        mock_ww.pause.assert_called_once()
        mock_ww.resume.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: trigger_voice_input
# ---------------------------------------------------------------------------

class TestTriggerVoiceInput:
    """Tests for the public trigger_voice_input method."""

    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    def test_trigger_voice_input_delegates_to_handle_wake(self, mock_vic):
        vc = _make_controller()
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc.trigger_voice_input()
        assert result is True
