"""
Unit tests for voice command support.

Tests cover:
  - VoiceCommandsConfig defaults and custom mapping
  - VoiceController.match_voice_command recognition
  - VoiceController.handle_voice_command dispatch
  - Daemon-level _match_voice_command / _handle_voice_command
  - Volume bounds (0.1 to 1.0)
  - Speed adjustments
  - Pause/resume state
  - Repeat stores and replays last message
  - Disabled commands (empty string)
"""

from unittest.mock import MagicMock, patch

import pytest

from claude_speak.config import (
    Config,
    VoiceCommandsConfig,
    WakeWordConfig,
    InputConfig,
    AudioConfig,
    TTSConfig,
)
from claude_speak.voice_controller import VoiceController
from claude_speak.daemon import (
    _match_voice_command,
    _handle_voice_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**vc_overrides) -> Config:
    """Build a Config with optional VoiceCommandsConfig overrides."""
    vc = VoiceCommandsConfig(**vc_overrides)
    return Config(
        tts=TTSConfig(volume=0.5, speed=1.0),
        wakeword=WakeWordConfig(enabled=False),
        input=InputConfig(backend="builtin"),
        audio=AudioConfig(chimes=False),
        voice_commands=vc,
    )


def _make_controller(voice_commands=None, tts_stop_callback=None,
                     voice_command_callback=None) -> VoiceController:
    """Build a VoiceController with a given VoiceCommandsConfig."""
    vc = voice_commands or VoiceCommandsConfig()
    config = Config(
        wakeword=WakeWordConfig(enabled=False),
        input=InputConfig(backend="builtin"),
        audio=AudioConfig(chimes=False),
        voice_commands=vc,
    )
    return VoiceController(
        config=config,
        tts_stop_callback=tts_stop_callback,
        voice_command_callback=voice_command_callback,
    )


# ===========================================================================
# VoiceCommandsConfig dataclass
# ===========================================================================

class TestVoiceCommandsConfig:
    """Tests for VoiceCommandsConfig defaults and custom values."""

    def test_defaults(self):
        vc = VoiceCommandsConfig()
        assert vc.pause == "pause"
        assert vc.resume == "resume"
        assert vc.repeat == "repeat"
        assert vc.louder == "louder"
        assert vc.quieter == "quieter"
        assert vc.faster == "faster"
        assert vc.slower == "slower"
        assert vc.stop == "stop"

    def test_custom_mapping(self):
        vc = VoiceCommandsConfig(pause="hold", resume="go", louder="up")
        assert vc.pause == "hold"
        assert vc.resume == "go"
        assert vc.louder == "up"
        # Unspecified remain default
        assert vc.repeat == "repeat"

    def test_disabled_command_empty_string(self):
        vc = VoiceCommandsConfig(pause="", resume="")
        assert vc.pause == ""
        assert vc.resume == ""

    def test_config_includes_voice_commands(self):
        config = Config()
        assert isinstance(config.voice_commands, VoiceCommandsConfig)
        assert config.voice_commands.pause == "pause"


# ===========================================================================
# VoiceController.match_voice_command
# ===========================================================================

class TestControllerMatchVoiceCommand:
    """Tests for VoiceController.match_voice_command."""

    def test_each_command_recognized(self):
        vc = _make_controller()
        assert vc.match_voice_command("pause") == "pause"
        assert vc.match_voice_command("resume") == "resume"
        assert vc.match_voice_command("repeat") == "repeat"
        assert vc.match_voice_command("louder") == "louder"
        assert vc.match_voice_command("quieter") == "quieter"
        assert vc.match_voice_command("faster") == "faster"
        assert vc.match_voice_command("slower") == "slower"
        assert vc.match_voice_command("stop") == "stop"

    def test_case_insensitive(self):
        vc = _make_controller()
        assert vc.match_voice_command("PAUSE") == "pause"
        assert vc.match_voice_command("Louder") == "louder"
        assert vc.match_voice_command("STOP") == "stop"

    def test_whitespace_stripped(self):
        vc = _make_controller()
        assert vc.match_voice_command("  pause  ") == "pause"
        assert vc.match_voice_command("\tlouder\n") == "louder"

    def test_no_match_returns_none(self):
        vc = _make_controller()
        assert vc.match_voice_command("hello world") is None
        assert vc.match_voice_command("") is None

    def test_partial_match_not_accepted(self):
        vc = _make_controller()
        assert vc.match_voice_command("pause please") is None
        assert vc.match_voice_command("get louder") is None

    def test_custom_command_word(self):
        vc = _make_controller(voice_commands=VoiceCommandsConfig(pause="hold"))
        assert vc.match_voice_command("hold") == "pause"
        assert vc.match_voice_command("pause") is None  # original word no longer works

    def test_disabled_command_not_recognized(self):
        vc = _make_controller(voice_commands=VoiceCommandsConfig(pause=""))
        assert vc.match_voice_command("pause") is None
        assert vc.match_voice_command("") is None


# ===========================================================================
# VoiceController.handle_voice_command
# ===========================================================================

class TestControllerHandleVoiceCommand:
    """Tests for VoiceController.handle_voice_command dispatch."""

    @patch("claude_speak.voice_controller.Q")
    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_stop_delegates_to_handle_stop(self, mock_mute, mock_q):
        stop_cb = MagicMock()
        vc = _make_controller(tts_stop_callback=stop_cb)
        result = vc.handle_voice_command("stop")
        assert result is True
        mock_q.clear.assert_called_once()
        stop_cb.assert_called_once()

    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_pause_creates_mute_file_and_stops_tts(self, mock_mute):
        stop_cb = MagicMock()
        vc = _make_controller(tts_stop_callback=stop_cb)
        result = vc.handle_voice_command("pause")
        assert result is True
        mock_mute.touch.assert_called_once()
        stop_cb.assert_called_once()

    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_resume_removes_mute_file(self, mock_mute):
        vc = _make_controller()
        result = vc.handle_voice_command("resume")
        assert result is True
        mock_mute.unlink.assert_called_once_with(missing_ok=True)

    def test_repeat_delegates_to_callback(self):
        cb = MagicMock()
        vc = _make_controller(voice_command_callback=cb)
        result = vc.handle_voice_command("repeat")
        assert result is True
        cb.assert_called_once_with("repeat")

    def test_louder_delegates_to_callback(self):
        cb = MagicMock()
        vc = _make_controller(voice_command_callback=cb)
        result = vc.handle_voice_command("louder")
        assert result is True
        cb.assert_called_once_with("louder")

    def test_quieter_delegates_to_callback(self):
        cb = MagicMock()
        vc = _make_controller(voice_command_callback=cb)
        result = vc.handle_voice_command("quieter")
        assert result is True
        cb.assert_called_once_with("quieter")

    def test_faster_delegates_to_callback(self):
        cb = MagicMock()
        vc = _make_controller(voice_command_callback=cb)
        result = vc.handle_voice_command("faster")
        assert result is True
        cb.assert_called_once_with("faster")

    def test_slower_delegates_to_callback(self):
        cb = MagicMock()
        vc = _make_controller(voice_command_callback=cb)
        result = vc.handle_voice_command("slower")
        assert result is True
        cb.assert_called_once_with("slower")

    def test_no_callback_returns_false_for_delegated_commands(self):
        vc = _make_controller(voice_command_callback=None)
        result = vc.handle_voice_command("repeat")
        assert result is False

    def test_unknown_command_returns_false(self):
        vc = _make_controller()
        result = vc.handle_voice_command("unknown_cmd")
        assert result is False

    @patch("claude_speak.voice_controller.MUTE_FILE")
    def test_pause_handles_callback_exception(self, mock_mute):
        stop_cb = MagicMock(side_effect=RuntimeError("boom"))
        vc = _make_controller(tts_stop_callback=stop_cb)
        # Should not raise
        result = vc.handle_voice_command("pause")
        assert result is True

    def test_delegated_command_handles_callback_exception(self):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        vc = _make_controller(voice_command_callback=cb)
        # Should not raise
        result = vc.handle_voice_command("louder")
        assert result is True


# ===========================================================================
# Daemon-level _match_voice_command
# ===========================================================================

class TestDaemonMatchVoiceCommand:
    """Tests for daemon._match_voice_command."""

    def test_each_command_recognized(self):
        config = _make_config()
        assert _match_voice_command("pause", config) == "pause"
        assert _match_voice_command("resume", config) == "resume"
        assert _match_voice_command("repeat", config) == "repeat"
        assert _match_voice_command("louder", config) == "louder"
        assert _match_voice_command("quieter", config) == "quieter"
        assert _match_voice_command("faster", config) == "faster"
        assert _match_voice_command("slower", config) == "slower"
        assert _match_voice_command("stop", config) == "stop"

    def test_case_insensitive(self):
        config = _make_config()
        assert _match_voice_command("PAUSE", config) == "pause"
        assert _match_voice_command("Resume", config) == "resume"

    def test_whitespace_stripped(self):
        config = _make_config()
        assert _match_voice_command("  repeat  ", config) == "repeat"

    def test_no_match(self):
        config = _make_config()
        assert _match_voice_command("hello", config) is None
        assert _match_voice_command("pause playback", config) is None

    def test_custom_mapping(self):
        config = _make_config(louder="volume up", quieter="volume down")
        assert _match_voice_command("volume up", config) == "louder"
        assert _match_voice_command("volume down", config) == "quieter"
        assert _match_voice_command("louder", config) is None

    def test_disabled_command(self):
        config = _make_config(pause="", resume="")
        assert _match_voice_command("pause", config) is None
        assert _match_voice_command("resume", config) is None
        # Other commands still work
        assert _match_voice_command("repeat", config) == "repeat"


# ===========================================================================
# Daemon-level _handle_voice_command — volume
# ===========================================================================

class TestDaemonVolumeCommand:
    """Tests for volume adjustment via _handle_voice_command."""

    def test_louder_increases_volume(self):
        config = _make_config()
        config.tts.volume = 0.5
        engine = MagicMock()
        result = _handle_voice_command("louder", config, engine)
        assert result is True
        assert config.tts.volume == pytest.approx(0.6)

    def test_quieter_decreases_volume(self):
        config = _make_config()
        config.tts.volume = 0.5
        engine = MagicMock()
        result = _handle_voice_command("quieter", config, engine)
        assert result is True
        assert config.tts.volume == pytest.approx(0.4)

    def test_volume_capped_at_1_0(self):
        config = _make_config()
        config.tts.volume = 0.9
        engine = MagicMock()
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == 1.0

    def test_volume_capped_at_1_0_from_exact(self):
        config = _make_config()
        config.tts.volume = 1.0
        engine = MagicMock()
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == 1.0

    def test_volume_min_at_0_1(self):
        config = _make_config()
        config.tts.volume = 0.2
        engine = MagicMock()
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.1)

    def test_volume_cannot_go_below_0_1(self):
        config = _make_config()
        config.tts.volume = 0.1
        engine = MagicMock()
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.1)

    def test_multiple_louder_increments(self):
        config = _make_config()
        config.tts.volume = 0.2
        engine = MagicMock()
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == pytest.approx(0.3)
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == pytest.approx(0.4)
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == pytest.approx(0.5)
        _handle_voice_command("louder", config, engine)
        assert config.tts.volume == pytest.approx(0.6)
        # Keep going to cap
        for _ in range(5):
            _handle_voice_command("louder", config, engine)
        assert config.tts.volume == pytest.approx(1.0)  # stays capped

    def test_multiple_quieter_decrements(self):
        config = _make_config()
        config.tts.volume = 0.5
        engine = MagicMock()
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.4)
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.3)
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.2)
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.1)
        _handle_voice_command("quieter", config, engine)
        assert config.tts.volume == pytest.approx(0.1)  # stays at min


# ===========================================================================
# Daemon-level _handle_voice_command — speed
# ===========================================================================

class TestDaemonSpeedCommand:
    """Tests for speed adjustment via _handle_voice_command."""

    def test_faster_increases_speed(self):
        config = _make_config()
        config.tts.speed = 1.0
        engine = MagicMock()
        result = _handle_voice_command("faster", config, engine)
        assert result is True
        assert config.tts.speed == pytest.approx(1.1)

    def test_slower_decreases_speed(self):
        config = _make_config()
        config.tts.speed = 1.0
        engine = MagicMock()
        result = _handle_voice_command("slower", config, engine)
        assert result is True
        assert config.tts.speed == pytest.approx(0.9)

    def test_speed_min_at_0_1(self):
        config = _make_config()
        config.tts.speed = 0.2
        engine = MagicMock()
        _handle_voice_command("slower", config, engine)
        assert config.tts.speed == pytest.approx(0.1)

    def test_speed_cannot_go_below_0_1(self):
        config = _make_config()
        config.tts.speed = 0.1
        engine = MagicMock()
        _handle_voice_command("slower", config, engine)
        assert config.tts.speed == pytest.approx(0.1)

    def test_faster_has_no_upper_bound(self):
        """Speed is not capped on the upper end (unlike volume)."""
        config = _make_config()
        config.tts.speed = 2.0
        engine = MagicMock()
        _handle_voice_command("faster", config, engine)
        assert config.tts.speed == pytest.approx(2.1)

    def test_multiple_speed_adjustments(self):
        config = _make_config()
        config.tts.speed = 1.0
        engine = MagicMock()
        _handle_voice_command("faster", config, engine)
        _handle_voice_command("faster", config, engine)
        assert config.tts.speed == pytest.approx(1.2)
        _handle_voice_command("slower", config, engine)
        assert config.tts.speed == pytest.approx(1.1)


# ===========================================================================
# Daemon-level _handle_voice_command — pause/resume
# ===========================================================================

class TestDaemonPauseResume:
    """Tests for pause/resume via _handle_voice_command."""

    @patch("claude_speak.daemon.MUTE_FILE")
    def test_pause_creates_mute_file(self, mock_mute):
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("pause", config, engine)
        assert result is True
        mock_mute.touch.assert_called_once()
        engine.stop.assert_called_once()

    @patch("claude_speak.daemon.MUTE_FILE")
    def test_resume_removes_mute_file(self, mock_mute):
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("resume", config, engine)
        assert result is True
        mock_mute.unlink.assert_called_once_with(missing_ok=True)

    @patch("claude_speak.daemon.MUTE_FILE")
    def test_pause_then_resume_cycle(self, mock_mute):
        config = _make_config()
        engine = MagicMock()

        _handle_voice_command("pause", config, engine)
        mock_mute.touch.assert_called_once()

        _handle_voice_command("resume", config, engine)
        mock_mute.unlink.assert_called_once_with(missing_ok=True)


# ===========================================================================
# Daemon-level _handle_voice_command — repeat
# ===========================================================================

class TestDaemonRepeat:
    """Tests for repeat via _handle_voice_command."""

    @patch("claude_speak.daemon.Q")
    def test_repeat_enqueues_last_spoken_text(self, mock_q):
        import claude_speak.daemon as d
        d._last_spoken_text = "Hello, world!"
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("repeat", config, engine)
        assert result is True
        mock_q.enqueue.assert_called_once_with("Hello, world!")

    @patch("claude_speak.daemon.Q")
    def test_repeat_with_no_previous_message(self, mock_q):
        import claude_speak.daemon as d
        d._last_spoken_text = None
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("repeat", config, engine)
        assert result is True
        mock_q.enqueue.assert_not_called()

    @patch("claude_speak.daemon.Q")
    def test_repeat_preserves_original_text(self, mock_q):
        """Repeat should enqueue the original text, not normalized."""
        import claude_speak.daemon as d
        original = "Check /usr/local/bin/python3 — it's version 3.11"
        d._last_spoken_text = original
        config = _make_config()
        engine = MagicMock()
        _handle_voice_command("repeat", config, engine)
        mock_q.enqueue.assert_called_once_with(original)


# ===========================================================================
# Daemon-level _handle_voice_command — stop
# ===========================================================================

class TestDaemonStop:
    """Tests for stop via _handle_voice_command."""

    @patch("claude_speak.daemon.Q")
    def test_stop_clears_queue_and_stops_engine(self, mock_q):
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("stop", config, engine)
        assert result is True
        engine.stop.assert_called_once()
        mock_q.clear.assert_called_once()


# ===========================================================================
# Daemon-level _handle_voice_command — unknown
# ===========================================================================

class TestDaemonUnknownCommand:
    """Tests for unknown commands."""

    def test_unknown_command_returns_false(self):
        config = _make_config()
        engine = MagicMock()
        result = _handle_voice_command("nonexistent", config, engine)
        assert result is False


# ===========================================================================
# Config TOML loading
# ===========================================================================

class TestConfigLoading:
    """Tests that voice_commands section loads from TOML."""

    def test_default_config_has_voice_commands(self):
        config = Config()
        assert config.voice_commands.pause == "pause"
        assert config.voice_commands.stop == "stop"

    @patch("claude_speak.config.CONFIG_PATH")
    def test_load_config_with_voice_commands_toml(self, mock_path, tmp_path):
        """Simulate loading voice_commands from a TOML file."""
        toml_file = tmp_path / "claude-speak.toml"
        toml_file.write_text(
            '[voice_commands]\npause = "hold"\nresume = "go"\nlouder = ""\n'
        )
        mock_path.exists.return_value = True

        # Use the actual load_config but point it at our tmp file
        from claude_speak.config import load_config
        with patch("claude_speak.config.CONFIG_PATH", toml_file):
            config = load_config()
            assert config.voice_commands.pause == "hold"
            assert config.voice_commands.resume == "go"
            assert config.voice_commands.louder == ""  # disabled
            # Unspecified remain default
            assert config.voice_commands.quieter == "quieter"


# ===========================================================================
# Integration: match + handle round-trip
# ===========================================================================

class TestMatchAndHandleIntegration:
    """Integration tests: match a word, then handle the command."""

    def test_match_and_handle_louder(self):
        config = _make_config()
        config.tts.volume = 0.5
        engine = MagicMock()
        cmd = _match_voice_command("louder", config)
        assert cmd == "louder"
        _handle_voice_command(cmd, config, engine)
        assert config.tts.volume == pytest.approx(0.6)

    def test_custom_word_match_and_handle(self):
        config = _make_config(faster="speed up")
        config.tts.speed = 1.0
        engine = MagicMock()
        cmd = _match_voice_command("speed up", config)
        assert cmd == "faster"
        _handle_voice_command(cmd, config, engine)
        assert config.tts.speed == pytest.approx(1.1)

    def test_disabled_command_not_handled(self):
        config = _make_config(pause="")
        engine = MagicMock()
        cmd = _match_voice_command("pause", config)
        assert cmd is None
        # No command to handle

    @patch("claude_speak.daemon.Q")
    @patch("claude_speak.daemon.MUTE_FILE")
    def test_pause_resume_round_trip(self, mock_mute, mock_q):
        config = _make_config()
        engine = MagicMock()

        # Pause
        cmd = _match_voice_command("pause", config)
        assert cmd == "pause"
        _handle_voice_command(cmd, config, engine)
        mock_mute.touch.assert_called_once()

        # Resume
        cmd = _match_voice_command("resume", config)
        assert cmd == "resume"
        _handle_voice_command(cmd, config, engine)
        mock_mute.unlink.assert_called_once_with(missing_ok=True)
