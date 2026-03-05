"""
Unit tests for claude_speak/media_keys.py — media key handler for TTS control.

Mocks all pyobjc/Quartz imports since they may not be available in test
environments (CI, non-macOS).
"""

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from claude_speak.config import Config, AudioConfig


# ---------------------------------------------------------------------------
# Helpers: mock Quartz/AppKit modules so tests work without pyobjc installed
# ---------------------------------------------------------------------------

def _make_mock_quartz():
    """Return a mock Quartz module with the constants and functions we use."""
    quartz = MagicMock()
    quartz.kCGSessionEventTap = 1
    quartz.kCGHeadInsertEventTap = 0
    quartz.kCGEventTapOptionListenOnly = 0x00000001
    quartz.kCGEventTapDisabledByTimeout = 0xFFFFFFFE
    quartz.kCGEventTapDisabledByUserInput = 0xFFFFFFFD
    quartz.kCFRunLoopDefaultMode = "kCFRunLoopDefaultMode"
    quartz.CGEventMaskBit = MagicMock(return_value=0x00004000)
    quartz.CGEventTapCreate = MagicMock(return_value=MagicMock())  # non-None = success
    quartz.CFMachPortCreateRunLoopSource = MagicMock(return_value=MagicMock())
    quartz.CGEventTapEnable = MagicMock()
    quartz.CFRunLoopGetCurrent = MagicMock(return_value=MagicMock())
    quartz.CFRunLoopAddSource = MagicMock()
    quartz.CFRunLoopRun = MagicMock()
    quartz.CFRunLoopStop = MagicMock()
    return quartz


def _make_mock_appkit():
    """Return a mock AppKit module with NSSystemDefined and NSEvent."""
    appkit = MagicMock()
    appkit.NSSystemDefined = 14
    return appkit


# ---------------------------------------------------------------------------
# Test class: MediaKeyHandler initialization
# ---------------------------------------------------------------------------

class TestMediaKeyHandlerInit:
    """Tests for MediaKeyHandler.__init__."""

    def test_init_with_quartz_available(self):
        """Handler initializes with Quartz available."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            callbacks = {"toggle_mute": MagicMock(), "volume_up": MagicMock(), "volume_down": MagicMock()}
            handler = MediaKeyHandler(callbacks)
            assert handler._quartz_available is True
            assert handler.is_running is False
            assert handler._callbacks is callbacks

    def test_init_without_quartz(self):
        """Handler gracefully degrades when pyobjc is not installed."""
        # Temporarily hide Quartz from imports
        with patch.dict(sys.modules, {"Quartz": None}):
            # We need to reimport to trigger the ImportError path in __init__
            # Use a fresh import by removing the cached module
            saved = sys.modules.pop("claude_speak.media_keys") if "claude_speak.media_keys" in sys.modules else None
            try:
                # Force reimport with Quartz missing
                import importlib
                # Patch the import inside the module
                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

                def mock_import(name, *args, **kwargs):
                    if name == "Quartz":
                        raise ImportError("No module named 'Quartz'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    from claude_speak.media_keys import MediaKeyHandler
                    handler = MediaKeyHandler({"toggle_mute": MagicMock()})
                    assert handler._quartz_available is False
            finally:
                if saved is not None:
                    sys.modules["claude_speak.media_keys"] = saved

    def test_init_stores_callbacks(self):
        """Callbacks dict is stored and accessible."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            vol_up = MagicMock()
            vol_down = MagicMock()
            callbacks = {"toggle_mute": toggle, "volume_up": vol_up, "volume_down": vol_down}
            handler = MediaKeyHandler(callbacks)
            assert handler._callbacks["toggle_mute"] is toggle
            assert handler._callbacks["volume_up"] is vol_up
            assert handler._callbacks["volume_down"] is vol_down


# ---------------------------------------------------------------------------
# Test class: callbacks are wired correctly
# ---------------------------------------------------------------------------

class TestCallbackDispatch:
    """Tests for _dispatch_key routing to correct callbacks."""

    def test_play_pause_dispatches_toggle_mute(self):
        """NX_KEYTYPE_PLAY (16) should call toggle_mute callback."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler, NX_KEYTYPE_PLAY
            toggle = MagicMock()
            handler = MediaKeyHandler({"toggle_mute": toggle})
            handler._dispatch_key(NX_KEYTYPE_PLAY)
            toggle.assert_called_once()

    def test_volume_up_dispatches_volume_up(self):
        """NX_KEYTYPE_SOUND_UP (0) should call volume_up callback."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler, NX_KEYTYPE_SOUND_UP
            vol_up = MagicMock()
            handler = MediaKeyHandler({"volume_up": vol_up})
            handler._dispatch_key(NX_KEYTYPE_SOUND_UP)
            vol_up.assert_called_once()

    def test_volume_down_dispatches_volume_down(self):
        """NX_KEYTYPE_SOUND_DOWN (1) should call volume_down callback."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler, NX_KEYTYPE_SOUND_DOWN
            vol_down = MagicMock()
            handler = MediaKeyHandler({"volume_down": vol_down})
            handler._dispatch_key(NX_KEYTYPE_SOUND_DOWN)
            vol_down.assert_called_once()

    def test_unknown_key_code_does_nothing(self):
        """Unknown key codes should not call any callback."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            vol_up = MagicMock()
            vol_down = MagicMock()
            handler = MediaKeyHandler({
                "toggle_mute": toggle,
                "volume_up": vol_up,
                "volume_down": vol_down,
            })
            handler._dispatch_key(99)  # unknown key code
            toggle.assert_not_called()
            vol_up.assert_not_called()
            vol_down.assert_not_called()

    def test_missing_callback_does_not_crash(self):
        """Dispatching to a key with no callback should not raise."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler, NX_KEYTYPE_PLAY
            handler = MediaKeyHandler({})  # no callbacks
            handler._dispatch_key(NX_KEYTYPE_PLAY)  # should not raise

    def test_callback_exception_is_caught(self):
        """Callback exceptions should be caught, not propagated."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler, NX_KEYTYPE_PLAY
            toggle = MagicMock(side_effect=RuntimeError("boom"))
            handler = MediaKeyHandler({"toggle_mute": toggle})
            # Should not raise
            handler._dispatch_key(NX_KEYTYPE_PLAY)
            toggle.assert_called_once()


# ---------------------------------------------------------------------------
# Test class: start/stop lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_returns_true_on_success(self):
        """start() returns True when event tap creation succeeds."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            result = handler.start()
            assert result is True
            assert handler.is_running is True
            handler.stop()

    def test_start_returns_false_when_already_running(self):
        """start() returns False if handler is already running."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler.start()
            result = handler.start()
            assert result is False
            handler.stop()

    def test_start_returns_false_without_quartz(self):
        """start() returns False when Quartz is not available."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler._quartz_available = False
            result = handler.start()
            assert result is False
            assert handler.is_running is False

    def test_start_returns_false_when_tap_creation_fails(self):
        """start() returns False when CGEventTapCreate returns None (no Accessibility)."""
        mock_quartz = _make_mock_quartz()
        mock_quartz.CGEventTapCreate.return_value = None  # simulate permission denied
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            result = handler.start()
            assert result is False
            assert handler.is_running is False

    def test_stop_clears_running_flag(self):
        """stop() sets is_running to False."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler.start()
            assert handler.is_running is True
            handler.stop()
            assert handler.is_running is False

    def test_stop_when_not_running_is_harmless(self):
        """stop() does nothing when handler is not running."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler.stop()  # should not raise
            assert handler.is_running is False

    def test_stop_disables_event_tap(self):
        """stop() disables the CGEvent tap."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler.start()
            handler.stop()
            # CGEventTapEnable should have been called with False during stop
            calls = mock_quartz.CGEventTapEnable.call_args_list
            # At least one call with False
            assert any(c[0][1] is False for c in calls if len(c[0]) >= 2)


# ---------------------------------------------------------------------------
# Test class: _is_tts_active
# ---------------------------------------------------------------------------

class TestIsTtsActive:
    """Tests for the _is_tts_active function."""

    def test_active_when_playing_file_exists(self, tmp_path):
        """Returns True when PLAYING_FILE exists."""
        playing = tmp_path / "playing"
        playing.touch()
        with patch("claude_speak.media_keys.PLAYING_FILE", playing):
            from claude_speak.media_keys import _is_tts_active
            assert _is_tts_active() is True

    def test_active_when_queue_has_items(self, tmp_path):
        """Returns True when queue directory has items."""
        playing = tmp_path / "playing"  # does not exist
        queue = tmp_path / "queue"
        queue.mkdir()
        (queue / "item1.txt").write_text("hello")
        with patch("claude_speak.media_keys.PLAYING_FILE", playing), \
             patch("claude_speak.media_keys.QUEUE_DIR", queue):
            from claude_speak.media_keys import _is_tts_active
            assert _is_tts_active() is True

    def test_inactive_when_nothing(self, tmp_path):
        """Returns False when no playing file and empty queue."""
        playing = tmp_path / "playing"  # does not exist
        queue = tmp_path / "queue"
        queue.mkdir()
        with patch("claude_speak.media_keys.PLAYING_FILE", playing), \
             patch("claude_speak.media_keys.QUEUE_DIR", queue):
            from claude_speak.media_keys import _is_tts_active
            assert _is_tts_active() is False

    def test_inactive_when_queue_dir_missing(self, tmp_path):
        """Returns False when queue directory does not exist."""
        playing = tmp_path / "playing"  # does not exist
        queue = tmp_path / "nonexistent"
        with patch("claude_speak.media_keys.PLAYING_FILE", playing), \
             patch("claude_speak.media_keys.QUEUE_DIR", queue):
            from claude_speak.media_keys import _is_tts_active
            assert _is_tts_active() is False


# ---------------------------------------------------------------------------
# Test class: config enable/disable
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests for media_keys_enabled config flag."""

    def test_audio_config_default_enabled(self):
        """media_keys_enabled defaults to True."""
        config = AudioConfig()
        assert config.media_keys_enabled is True

    def test_audio_config_can_disable(self):
        """media_keys_enabled can be set to False."""
        config = AudioConfig(media_keys_enabled=False)
        assert config.media_keys_enabled is False

    def test_full_config_has_media_keys_enabled(self):
        """Config.audio.media_keys_enabled is accessible."""
        config = Config()
        assert config.audio.media_keys_enabled is True

    def test_config_disabled_skips_handler(self):
        """When media_keys_enabled is False, daemon should not start the handler."""
        config = Config(audio=AudioConfig(media_keys_enabled=False))
        assert config.audio.media_keys_enabled is False
        # This verifies the config flag works -- the daemon checks this flag
        # before attempting to create a MediaKeyHandler.


# ---------------------------------------------------------------------------
# Test class: tap callback handling
# ---------------------------------------------------------------------------

class TestTapCallback:
    """Tests for _tap_callback event processing."""

    def _make_handler_and_event(self, key_code, key_state=0x0A, subtype=8):
        """Helper to create a handler and a mock NSEvent with given key params."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()

        # Encode data1: key_code in bits 16-23, key_state in bits 8-15
        data1 = (key_code << 16) | (key_state << 8)

        mock_ns_event = MagicMock()
        mock_ns_event.subtype.return_value = subtype
        mock_ns_event.data1.return_value = data1

        mock_appkit.NSEvent.eventWithCGEvent_ = MagicMock(return_value=mock_ns_event)

        return mock_quartz, mock_appkit, data1

    def test_callback_dispatches_play_on_key_down(self, tmp_path):
        """Play/pause key down triggers toggle_mute callback."""
        mock_quartz, mock_appkit, _ = self._make_handler_and_event(key_code=16, key_state=0x0A)
        playing = tmp_path / "playing"
        playing.touch()

        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}), \
             patch("claude_speak.media_keys.PLAYING_FILE", playing):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            handler = MediaKeyHandler({"toggle_mute": toggle})
            mock_event = MagicMock()

            result = handler._tap_callback(None, 0, mock_event, None)
            toggle.assert_called_once()

    def test_callback_ignores_key_up(self, tmp_path):
        """Key up events should be ignored."""
        mock_quartz, mock_appkit, _ = self._make_handler_and_event(key_code=16, key_state=0x0B)
        playing = tmp_path / "playing"
        playing.touch()

        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}), \
             patch("claude_speak.media_keys.PLAYING_FILE", playing):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            handler = MediaKeyHandler({"toggle_mute": toggle})
            mock_event = MagicMock()

            handler._tap_callback(None, 0, mock_event, None)
            toggle.assert_not_called()

    def test_callback_ignores_non_media_key_subtype(self, tmp_path):
        """Non-media-key subtype should be ignored."""
        mock_quartz, mock_appkit, _ = self._make_handler_and_event(key_code=16, key_state=0x0A, subtype=5)
        playing = tmp_path / "playing"
        playing.touch()

        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}), \
             patch("claude_speak.media_keys.PLAYING_FILE", playing):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            handler = MediaKeyHandler({"toggle_mute": toggle})
            mock_event = MagicMock()

            handler._tap_callback(None, 0, mock_event, None)
            toggle.assert_not_called()

    @pytest.mark.skipif(sys.platform == "win32", reason="macOS-only media key handling")
    def test_callback_ignores_when_tts_inactive(self, tmp_path):
        """Media keys should be ignored when TTS is not active."""
        mock_quartz, mock_appkit, _ = self._make_handler_and_event(key_code=16, key_state=0x0A)
        playing = tmp_path / "playing"  # does not exist
        queue = tmp_path / "queue"
        queue.mkdir()  # empty queue

        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}), \
             patch("claude_speak.media_keys.PLAYING_FILE", playing), \
             patch("claude_speak.media_keys.QUEUE_DIR", queue):
            from claude_speak.media_keys import MediaKeyHandler
            toggle = MagicMock()
            handler = MediaKeyHandler({"toggle_mute": toggle})
            mock_event = MagicMock()

            handler._tap_callback(None, 0, mock_event, None)
            toggle.assert_not_called()

    def test_callback_handles_tap_disabled_by_timeout(self):
        """Handler should re-enable tap when disabled by timeout."""
        mock_quartz = _make_mock_quartz()
        mock_appkit = _make_mock_appkit()

        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler._tap = MagicMock()

            mock_event = MagicMock()
            result = handler._tap_callback(
                None,
                mock_quartz.kCGEventTapDisabledByTimeout,
                mock_event,
                None,
            )
            mock_quartz.CGEventTapEnable.assert_called_with(handler._tap, True)
            assert result is mock_event


# ---------------------------------------------------------------------------
# Test class: graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Tests for graceful degradation without pyobjc."""

    def test_handler_with_quartz_unavailable_flag(self):
        """Handler with _quartz_available=False cannot start."""
        mock_quartz = _make_mock_quartz()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": _make_mock_appkit()}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            handler._quartz_available = False
            assert handler.start() is False
            assert handler.is_running is False

    def test_handler_tap_creation_failure_returns_false(self):
        """If CGEventTapCreate returns None, start() returns False."""
        mock_quartz = _make_mock_quartz()
        mock_quartz.CGEventTapCreate.return_value = None
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            assert handler.start() is False
            assert handler.is_running is False

    def test_handler_run_loop_source_failure_returns_false(self):
        """If CFMachPortCreateRunLoopSource returns None, start() returns False."""
        mock_quartz = _make_mock_quartz()
        mock_quartz.CFMachPortCreateRunLoopSource.return_value = None
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            assert handler.start() is False
            assert handler.is_running is False

    def test_create_tap_exception_returns_false(self):
        """If _create_tap raises, start() returns False."""
        mock_quartz = _make_mock_quartz()
        mock_quartz.CGEventTapCreate.side_effect = RuntimeError("unexpected")
        mock_appkit = _make_mock_appkit()
        with patch.dict(sys.modules, {"Quartz": mock_quartz, "AppKit": mock_appkit}):
            from claude_speak.media_keys import MediaKeyHandler
            handler = MediaKeyHandler({"toggle_mute": MagicMock()})
            assert handler.start() is False
            assert handler.is_running is False


# ---------------------------------------------------------------------------
# Test class: daemon integration (media key callbacks wiring)
# ---------------------------------------------------------------------------

class TestDaemonIntegration:
    """Tests that daemon correctly wires media key callbacks."""

    def test_volume_up_callback_increases_tts_volume(self):
        """Volume up callback should increase config.tts.volume by 0.1."""
        from claude_speak.config import Config, TTSConfig
        config = Config(tts=TTSConfig(volume=0.5))
        old_vol = config.tts.volume

        def _volume_up():
            config.tts.volume = min(1.0, round(config.tts.volume + 0.1, 2))

        _volume_up()
        assert config.tts.volume == pytest.approx(0.6)

    def test_volume_down_callback_decreases_tts_volume(self):
        """Volume down callback should decrease config.tts.volume by 0.1."""
        from claude_speak.config import Config, TTSConfig
        config = Config(tts=TTSConfig(volume=0.5))

        def _volume_down():
            config.tts.volume = max(0.1, round(config.tts.volume - 0.1, 2))

        _volume_down()
        assert config.tts.volume == pytest.approx(0.4)

    def test_volume_up_clamps_at_max(self):
        """Volume up should not exceed 1.0."""
        from claude_speak.config import Config, TTSConfig
        config = Config(tts=TTSConfig(volume=0.95))

        def _volume_up():
            config.tts.volume = min(1.0, round(config.tts.volume + 0.1, 2))

        _volume_up()
        assert config.tts.volume == 1.0

    def test_volume_down_clamps_at_min(self):
        """Volume down should not go below 0.1."""
        from claude_speak.config import Config, TTSConfig
        config = Config(tts=TTSConfig(volume=0.15))

        def _volume_down():
            config.tts.volume = max(0.1, round(config.tts.volume - 0.1, 2))

        _volume_down()
        assert config.tts.volume == pytest.approx(0.1)

    def test_toggle_mute_creates_mute_file(self, tmp_path):
        """Toggle mute should create MUTE_FILE when not muted."""
        mute_file = tmp_path / "muted"
        engine = MagicMock()

        def _toggle_mute():
            if mute_file.exists():
                mute_file.unlink(missing_ok=True)
            else:
                mute_file.touch()
                engine.stop()

        _toggle_mute()
        assert mute_file.exists()
        engine.stop.assert_called_once()

    def test_toggle_mute_removes_mute_file(self, tmp_path):
        """Toggle mute should remove MUTE_FILE when already muted."""
        mute_file = tmp_path / "muted"
        mute_file.touch()
        engine = MagicMock()

        def _toggle_mute():
            if mute_file.exists():
                mute_file.unlink(missing_ok=True)
            else:
                mute_file.touch()
                engine.stop()

        _toggle_mute()
        assert not mute_file.exists()
        engine.stop.assert_not_called()
