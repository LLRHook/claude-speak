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
                     tts_stop_callback=None, backend="builtin"):
    config = Config(
        wakeword=WakeWordConfig(enabled=wakeword_enabled),
        input=InputConfig(auto_submit=auto_submit, backend=backend),
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
    """Tests for _on_wake_word — interrupt during playback vs voice input."""

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupts_when_playing(self, mock_mute, mock_playing, mock_builtin):
        """If TTS is playing, wake word should interrupt and start voice input."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = False
        interrupt_cb = MagicMock()
        vc = _make_controller(tts_stop_callback=None)
        vc._interrupt_callback = interrupt_cb
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        vc._on_wake_word()
        time.sleep(0.2)
        interrupt_cb.assert_called_once()
        # Sentinel files should be cleaned up
        mock_playing.unlink.assert_called_with(missing_ok=True)
        mock_mute.unlink.assert_called_with(missing_ok=True)
        # Voice input should start after interrupt
        mock_builtin.assert_called_once()

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_interrupts_when_muted(self, mock_mute, mock_playing, mock_builtin):
        """If TTS is playing AND muted, wake word should still interrupt."""
        mock_playing.exists.return_value = True
        mock_mute.exists.return_value = True
        interrupt_cb = MagicMock()
        vc = _make_controller()
        vc._interrupt_callback = interrupt_cb
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        vc._on_wake_word()
        time.sleep(0.2)
        interrupt_cb.assert_called_once()
        mock_mute.unlink.assert_called_with(missing_ok=True)

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle")
    @patch("claude_speak.voice_controller.PLAYING_FILE")
    def test_starts_voice_input_when_idle(self, mock_playing, mock_builtin):
        """If TTS is not playing, wake word should start voice input."""
        mock_playing.exists.return_value = False
        mock_builtin.return_value = True
        vc = _make_controller(backend="builtin")
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        # Call directly — _on_wake_word spawns a thread
        vc._on_wake_word()
        # Give the thread a moment
        time.sleep(0.2)
        mock_builtin.assert_called_once()

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
    def test_successful_cycle_superwhisper(self, mock_vic, mock_ack, mock_err):
        vc = _make_controller(chimes=True, backend="superwhisper")
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
        vc = _make_controller(chimes=True, backend="superwhisper")
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc._handle_wake()
        assert result is False
        mock_err.assert_called_once()

    @patch("claude_speak.voice_controller.voice_input_cycle", return_value=True)
    def test_double_trigger_prevented_by_lock(self, mock_vic):
        """Second concurrent call should return False due to _input_lock."""
        vc = _make_controller(backend="superwhisper")
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
        vc = _make_controller(backend="superwhisper")
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
        vc = _make_controller(backend="superwhisper")
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc.trigger_voice_input()
        assert result is True


# ---------------------------------------------------------------------------
# Tests: _choose_voice_input (flow selection)
# ---------------------------------------------------------------------------

class TestChooseVoiceInput:
    """Tests for _choose_voice_input — selecting Superwhisper vs built-in."""

    def test_backend_superwhisper_always_returns_superwhisper(self):
        """backend='superwhisper' always chooses superwhisper, regardless of process state."""
        config = Config(input=InputConfig(backend="superwhisper"))
        vc = VoiceController(config=config)
        assert vc._choose_voice_input() == "superwhisper"

    def test_backend_builtin_always_returns_builtin(self):
        """backend='builtin' always chooses builtin, regardless of process state."""
        config = Config(input=InputConfig(backend="builtin"))
        vc = VoiceController(config=config)
        assert vc._choose_voice_input() == "builtin"

    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=True)
    def test_backend_auto_with_superwhisper_running(self, mock_sw):
        """backend='auto' chooses superwhisper when the process is running."""
        config = Config(input=InputConfig(backend="auto"))
        vc = VoiceController(config=config)
        assert vc._choose_voice_input() == "superwhisper"

    @patch("claude_speak.voice_input._is_superwhisper_running", return_value=False)
    def test_backend_auto_without_superwhisper_running(self, mock_sw):
        """backend='auto' falls back to builtin when the process is not running."""
        config = Config(input=InputConfig(backend="auto"))
        vc = VoiceController(config=config)
        assert vc._choose_voice_input() == "builtin"

    def test_default_backend_returns_builtin(self):
        """Default InputConfig (backend='builtin') always chooses builtin."""
        vc = _make_controller()
        assert vc._choose_voice_input() == "builtin"

    @patch("claude_speak.voice_controller.builtin_voice_input_cycle", return_value=True)
    @patch("claude_speak.voice_controller.play_ack_chime")
    def test_handle_wake_uses_builtin_when_backend_builtin(self, mock_ack, mock_builtin):
        """When backend='builtin', _handle_wake uses builtin flow."""
        vc = _make_controller(chimes=True, backend="builtin")
        vc._running = True
        vc._wakeword_listener = MagicMock()
        vc._wakeword_listener.is_running = True
        result = vc._handle_wake()
        assert result is True
        mock_builtin.assert_called_once()
        mock_ack.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: bt_workaround_active property
# ---------------------------------------------------------------------------

class TestBtWorkaroundActive:
    """Tests for the bt_workaround_active property."""

    def test_bt_workaround_active_false_when_no_listener(self):
        """Returns False when no wakeword listener is set."""
        vc = _make_controller(wakeword_enabled=False)
        assert vc.bt_workaround_active is False

    def test_bt_workaround_active_false_when_listener_not_bt(self):
        """Returns False when the listener has _bt_always_builtin=False."""
        vc = _make_controller()
        mock_ww = MagicMock()
        mock_ww._bt_always_builtin = False
        vc._wakeword_listener = mock_ww
        assert vc.bt_workaround_active is False

    def test_bt_workaround_active_true_when_listener_bt(self):
        """Returns True when the listener has _bt_always_builtin=True."""
        vc = _make_controller()
        mock_ww = MagicMock()
        mock_ww._bt_always_builtin = True
        vc._wakeword_listener = mock_ww
        assert vc.bt_workaround_active is True

    @patch("claude_speak.voice_controller.WakeWordListener")
    def test_start_wakeword_passes_audio_config(self, mock_ww_cls):
        """_start_wakeword passes the AudioConfig to WakeWordListener."""
        mock_ww = MagicMock()
        mock_ww.start.return_value = True
        mock_ww._bt_always_builtin = False
        mock_ww_cls.return_value = mock_ww

        audio_cfg = AudioConfig(bt_mic_workaround=True)
        config = Config(
            wakeword=WakeWordConfig(enabled=True),
            input=InputConfig(backend="builtin"),
            audio=audio_cfg,
        )
        vc = VoiceController(config=config)
        vc.start()

        # Verify WakeWordListener was constructed with the audio_config kwarg
        _, kwargs = mock_ww_cls.call_args
        assert "audio_config" in kwargs
        assert kwargs["audio_config"] is audio_cfg

    @patch("claude_speak.voice_controller.WakeWordListener")
    def test_start_wakeword_passes_wakeword_config_as_positional(self, mock_ww_cls):
        """_start_wakeword passes the WakeWordConfig as the first positional arg."""
        mock_ww = MagicMock()
        mock_ww.start.return_value = True
        mock_ww._bt_always_builtin = False
        mock_ww_cls.return_value = mock_ww

        ww_cfg = WakeWordConfig(enabled=True, sensitivity=0.7)
        config = Config(wakeword=ww_cfg, audio=AudioConfig())
        vc = VoiceController(config=config)
        vc.start()

        args, _ = mock_ww_cls.call_args
        assert args[0] is ww_cfg
