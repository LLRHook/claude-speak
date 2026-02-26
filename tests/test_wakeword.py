"""
Unit tests for src/wakeword.py — WakeWordListener.

Mocks openwakeword.model.Model and sounddevice to avoid audio hardware.
"""

import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from claude_speak.config import AudioConfig, WakeWordConfig
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


# ---------------------------------------------------------------------------
# Tests: Bluetooth mic workaround
# ---------------------------------------------------------------------------

def _make_listener_with_audio_config(bt_mic_workaround: bool = True, **ww_overrides):
    """Create a WakeWordListener with explicit AudioConfig for BT tests."""
    ww_defaults = dict(
        enabled=True,
        model="hey_jarvis",
        stop_model="",
        sensitivity=0.5,
        stop_sensitivity=0.5,
    )
    ww_defaults.update(ww_overrides)
    ww_config = WakeWordConfig(**ww_defaults)
    audio_config = AudioConfig(bt_mic_workaround=bt_mic_workaround)
    return WakeWordListener(ww_config, audio_config=audio_config)


class TestBluetoothMicWorkaround:
    """Tests for the bt_mic_workaround feature in WakeWordListener."""

    def test_bt_always_builtin_false_by_default_before_loop_starts(self):
        """_bt_always_builtin starts as False before _listen_loop runs."""
        listener = _make_listener_with_audio_config(bt_mic_workaround=True)
        assert listener._bt_always_builtin is False

    def test_audio_config_stored_on_listener(self):
        """AudioConfig is stored and accessible on the listener."""
        audio_cfg = AudioConfig(bt_mic_workaround=False)
        ww_cfg = WakeWordConfig(enabled=True)
        listener = WakeWordListener(ww_cfg, audio_config=audio_cfg)
        assert listener._audio_config.bt_mic_workaround is False

    def test_default_audio_config_has_bt_workaround_enabled(self):
        """When no AudioConfig is given, the workaround defaults to True."""
        listener = WakeWordListener(WakeWordConfig())
        assert listener._audio_config.bt_mic_workaround is True

    @patch("claude_speak.wakeword.get_device_manager")
    def test_bt_always_builtin_set_when_bt_output_detected(self, mock_get_dm):
        """_listen_loop sets _bt_always_builtin when BT output is detected."""
        mock_dm = MagicMock()
        mock_dm.find_builtin_mic.return_value = 1
        mock_dm.get_default_output.return_value = 0
        mock_dm.is_bluetooth.return_value = True
        mock_get_dm.return_value = mock_dm

        listener = _make_listener_with_audio_config(bt_mic_workaround=True)
        listener._running_event.set()  # will be cleared by _load_model failure — that's fine

        # Stub _load_model to fail immediately so _listen_loop exits cleanly
        listener._load_model = MagicMock(return_value=False)

        listener._listen_loop()

        # Even though load_model failed, the BT check happens first only if
        # load_model succeeds. Let's patch it to succeed and confirm the flag.
        listener._bt_always_builtin = False  # reset
        listener._running_event.set()

        # Now make load succeed and PortAudio fail immediately after BT check
        import sounddevice as sd_real
        mock_sd = MagicMock()
        mock_sd.InputStream.return_value.__enter__ = MagicMock(side_effect=Exception("stop"))
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
        mock_sd.PortAudioError = OSError

        def fake_load_model(self_inner=listener):
            self_inner._model = MagicMock()
            self_inner._model.predict.return_value = {}
            return True

        listener._load_model = fake_load_model

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": MagicMock()}):
            import numpy as np_mock
            mock_sd.InputStream.return_value.__enter__.side_effect = None
            mock_sd.InputStream.return_value.__enter__.return_value = MagicMock()

            # Let the loop run briefly then stop it
            import threading

            def stop_soon():
                time.sleep(0.05)
                listener._running_event.clear()

            stopper = threading.Thread(target=stop_soon, daemon=True)
            stopper.start()
            listener._listen_loop()

        assert listener._bt_always_builtin is True

    @patch("claude_speak.wakeword.get_device_manager")
    def test_bt_always_builtin_not_set_when_non_bt_output(self, mock_get_dm):
        """_listen_loop does NOT set _bt_always_builtin for non-BT output."""
        mock_dm = MagicMock()
        mock_dm.find_builtin_mic.return_value = 1
        mock_dm.get_default_output.return_value = 2
        mock_dm.is_bluetooth.return_value = False
        mock_get_dm.return_value = mock_dm

        listener = _make_listener_with_audio_config(bt_mic_workaround=True)

        def fake_load_model(self_inner=listener):
            self_inner._model = MagicMock()
            return True

        listener._load_model = fake_load_model

        mock_sd = MagicMock()
        mock_sd.PortAudioError = OSError
        inner_stream = MagicMock()
        # Make read() raise immediately so the loop exits without sleeping
        inner_stream.read.side_effect = Exception("stop loop")
        mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=inner_stream)
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

        import threading

        def stop_soon():
            time.sleep(0.05)
            listener._running_event.clear()

        stopper = threading.Thread(target=stop_soon, daemon=True)
        stopper.start()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": MagicMock()}):
            listener._listen_loop()

        assert listener._bt_always_builtin is False

    @patch("claude_speak.wakeword.get_device_manager")
    def test_bt_always_builtin_not_set_when_workaround_disabled(self, mock_get_dm):
        """When bt_mic_workaround=False, BT output does not set _bt_always_builtin."""
        mock_dm = MagicMock()
        mock_dm.find_builtin_mic.return_value = 1
        mock_dm.get_default_output.return_value = 0
        mock_dm.is_bluetooth.return_value = True  # BT output is present
        mock_get_dm.return_value = mock_dm

        # Workaround disabled
        listener = _make_listener_with_audio_config(bt_mic_workaround=False)

        def fake_load_model(self_inner=listener):
            self_inner._model = MagicMock()
            return True

        listener._load_model = fake_load_model

        mock_sd = MagicMock()
        mock_sd.PortAudioError = OSError
        inner_stream = MagicMock()
        inner_stream.read.side_effect = Exception("stop loop")
        mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=inner_stream)
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

        import threading

        def stop_soon():
            time.sleep(0.05)
            listener._running_event.clear()

        stopper = threading.Thread(target=stop_soon, daemon=True)
        stopper.start()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": MagicMock()}):
            listener._listen_loop()

        assert listener._bt_always_builtin is False

    @patch("claude_speak.wakeword.get_device_manager")
    def test_bt_always_builtin_not_set_when_no_builtin_mic_found(self, mock_get_dm):
        """When no built-in mic is found, BT workaround stays inactive."""
        mock_dm = MagicMock()
        mock_dm.find_builtin_mic.return_value = None  # no built-in mic
        mock_dm.get_default_output.return_value = 0
        mock_dm.is_bluetooth.return_value = True
        mock_get_dm.return_value = mock_dm

        listener = _make_listener_with_audio_config(bt_mic_workaround=True)

        def fake_load_model(self_inner=listener):
            self_inner._model = MagicMock()
            return True

        listener._load_model = fake_load_model

        mock_sd = MagicMock()
        mock_sd.PortAudioError = OSError
        inner_stream = MagicMock()
        inner_stream.read.side_effect = Exception("stop loop")
        mock_sd.InputStream.return_value.__enter__ = MagicMock(return_value=inner_stream)
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)

        import threading

        def stop_soon():
            time.sleep(0.05)
            listener._running_event.clear()

        stopper = threading.Thread(target=stop_soon, daemon=True)
        stopper.start()

        with patch.dict("sys.modules", {"sounddevice": mock_sd, "numpy": MagicMock()}):
            listener._listen_loop()

        assert listener._bt_always_builtin is False

    def test_use_builtin_mic_still_works_as_no_op_when_bt_always_builtin(self):
        """use_builtin_mic() is safe to call even when bt_always_builtin is True."""
        listener = _make_listener_with_audio_config(bt_mic_workaround=True)
        listener._bt_always_builtin = True  # simulate post-startup BT detection
        listener.use_builtin_mic()
        assert listener._swap_mic_event.is_set() is True  # event is set, but listener ignores it

    def test_use_default_mic_still_works_as_no_op_when_bt_always_builtin(self):
        """use_default_mic() is safe to call even when bt_always_builtin is True."""
        listener = _make_listener_with_audio_config(bt_mic_workaround=True)
        listener._bt_always_builtin = True
        listener._swap_mic_event.set()
        listener.use_default_mic()
        assert listener._swap_mic_event.is_set() is False  # event cleared, but builtin still used
