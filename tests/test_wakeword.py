"""
Unit tests for src/wakeword.py — WakeWordListener.

Mocks openwakeword.model.Model and sounddevice to avoid audio hardware.
"""

import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from claude_speak.config import WakeWordConfig
from claude_speak.audio_devices import AudioDeviceManager, get_device_manager
from claude_speak.wakeword import WakeWordListener, _openwakeword_available, _sounddevice_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_listener(**overrides):
    """Create a WakeWordListener with test-friendly config."""
    defaults = dict(
        enabled=True,
        model="hey_jarvis",
        stop_model="",
        sensitivity=0.5,
        stop_sensitivity=0.5,
    )
    defaults.update(overrides)
    config = WakeWordConfig(**defaults)
    return WakeWordListener(config)


# ---------------------------------------------------------------------------
# Tests: _process_audio — wake word detection
# ---------------------------------------------------------------------------

class TestProcessAudio:
    """Tests for _process_audio with mock predictions."""

    def test_fires_wake_callback_when_score_exceeds_threshold(self):
        listener = _make_listener(sensitivity=0.5)
        listener._model = MagicMock()
        listener._model.predict.return_value = {"hey_jarvis": 0.8}

        callback = MagicMock()
        listener.on_wake(callback)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        callback.assert_called_once()

    def test_does_not_fire_when_score_below_threshold(self):
        listener = _make_listener(sensitivity=0.5)
        listener._model = MagicMock()
        listener._model.predict.return_value = {"hey_jarvis": 0.3}

        callback = MagicMock()
        listener.on_wake(callback)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        callback.assert_not_called()

    def test_cooldown_prevents_rapid_fire(self):
        listener = _make_listener(sensitivity=0.5)
        listener._cooldown_seconds = 2.0
        listener._model = MagicMock()
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        callback = MagicMock()
        listener.on_wake(callback)

        samples = np.zeros(1280, dtype=np.int16)

        # First detection should fire
        listener._process_audio(samples)
        assert callback.call_count == 1

        # Second detection immediately after should be suppressed (cooldown)
        listener._process_audio(samples)
        assert callback.call_count == 1

    def test_stop_model_takes_priority_over_wake(self):
        listener = _make_listener(sensitivity=0.5, stop_sensitivity=0.5)
        listener._stop_model_name = "stop"
        listener._model = MagicMock()
        # Both models trigger -- stop should win
        listener._model.predict.return_value = {"hey_jarvis": 0.9, "stop": 0.8}

        wake_cb = MagicMock()
        stop_cb = MagicMock()
        listener.on_wake(wake_cb)
        listener.on_stop_phrase(stop_cb)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        stop_cb.assert_called_once_with("stop")
        wake_cb.assert_not_called()

    def test_stop_callback_receives_phrase(self):
        listener = _make_listener(stop_sensitivity=0.5)
        listener._stop_model_name = "stop"
        listener._model = MagicMock()
        listener._model.predict.return_value = {"stop": 0.9}

        stop_cb = MagicMock()
        listener.on_stop_phrase(stop_cb)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        stop_cb.assert_called_once_with("stop")

    def test_stop_cooldown_prevents_rapid_fire(self):
        listener = _make_listener(stop_sensitivity=0.5)
        listener._cooldown_seconds = 2.0
        listener._stop_model_name = "stop"
        listener._model = MagicMock()
        listener._model.predict.return_value = {"stop": 0.9}

        stop_cb = MagicMock()
        listener.on_stop_phrase(stop_cb)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)
        listener._process_audio(samples)

        assert stop_cb.call_count == 1

    def test_no_processing_when_model_is_none(self):
        listener = _make_listener()
        listener._model = None

        callback = MagicMock()
        listener.on_wake(callback)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    """Tests for callback management and error handling."""

    def test_multiple_wake_callbacks_fired_in_order(self):
        listener = _make_listener(sensitivity=0.5)
        listener._model = MagicMock()
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        order = []
        listener.on_wake(lambda: order.append("first"))
        listener.on_wake(lambda: order.append("second"))

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        assert order == ["first", "second"]

    def test_callback_exception_is_caught(self):
        """A failing callback should not crash the listener."""
        listener = _make_listener(sensitivity=0.5)
        listener._model = MagicMock()
        listener._model.predict.return_value = {"hey_jarvis": 0.9}

        failing_cb = MagicMock(side_effect=RuntimeError("boom"))
        ok_cb = MagicMock()
        listener.on_wake(failing_cb)
        listener.on_wake(ok_cb)

        samples = np.zeros(1280, dtype=np.int16)
        listener._process_audio(samples)

        failing_cb.assert_called_once()
        ok_cb.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: pause/resume and mic swap
# ---------------------------------------------------------------------------

class TestPauseResumeAndMicSwap:
    """Tests for pause/resume and use_builtin_mic/use_default_mic.

    The WakeWordListener uses threading.Event objects:
      _paused_event:   set() = paused, clear() = active
      _swap_mic_event: set() = builtin mic, clear() = default mic
    """

    def test_pause_sets_paused_event(self):
        listener = _make_listener()
        listener.pause()
        assert listener._paused_event.is_set() is True

    def test_resume_clears_paused_event(self):
        listener = _make_listener()
        listener._paused_event.set()
        listener.resume()
        assert listener._paused_event.is_set() is False

    def test_use_builtin_mic_sets_swap_event(self):
        listener = _make_listener()
        listener.use_builtin_mic()
        assert listener._swap_mic_event.is_set() is True

    def test_use_default_mic_clears_swap_event(self):
        listener = _make_listener()
        listener._swap_mic_event.set()
        listener.use_default_mic()
        assert listener._swap_mic_event.is_set() is False


# ---------------------------------------------------------------------------
# Tests: start/stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for start/stop lifecycle.

    Uses _running_event (threading.Event): set() = running, clear() = stopped.
    """

    def test_start_when_disabled_returns_false(self):
        listener = _make_listener(enabled=False)
        result = listener.start()
        assert result is False
        assert listener._running_event.is_set() is False

    def test_start_when_already_running_returns_false(self):
        listener = _make_listener()
        listener._running_event.set()  # simulate already running
        result = listener.start()
        assert result is False

    def test_start_without_openwakeword_returns_false(self):
        listener = _make_listener()
        with patch("claude_speak.wakeword._openwakeword_available", return_value=False):
            result = listener.start()
            assert result is False

    def test_stop_clears_running_event(self):
        listener = _make_listener()
        listener._running_event.set()
        listener._thread = None
        listener._model = MagicMock()
        listener.stop()
        assert listener._running_event.is_set() is False
        assert listener._model is None

    def test_is_running_property(self):
        listener = _make_listener()
        assert listener.is_running is False
        listener._running_event.set()
        assert listener.is_running is True


# ---------------------------------------------------------------------------
# Tests: _find_builtin_mic
# ---------------------------------------------------------------------------

class TestFindBuiltinMic:
    """Tests for find_builtin_mic via AudioDeviceManager."""

    def test_finds_macbook_mic(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "AirPods Pro", "max_input_channels": 1},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.find_builtin_mic()
            assert result == 1

    def test_returns_none_when_no_builtin(self):
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "AirPods Pro", "max_input_channels": 1},
        ]
        dm = AudioDeviceManager()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            result = dm.find_builtin_mic()
            assert result is None
