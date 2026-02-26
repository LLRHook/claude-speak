"""
Unit tests for engine hot-swap support (task 5.1.4).

Tests the create_backend factory, TTSEngine.swap_backend, and daemon
config reload engine-change detection.

All external dependencies are mocked — tests run without audio hardware
or model files.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.config import Config, TTSConfig
from claude_speak.tts_base import TTSBackend
import claude_speak.tts as tts_module
from claude_speak.tts import (
    KokoroBackend,
    TTSEngine,
    create_backend,
    AVAILABLE_ENGINES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(engine="kokoro", voice="af_sarah", speed=1.0, device="auto", volume=1.0):
    """Create a Config with the given TTS settings."""
    return Config(tts=TTSConfig(engine=engine, voice=voice, speed=speed, device=device, volume=volume))


class _StubBackend(TTSBackend):
    """Minimal concrete backend for testing."""

    def __init__(self, backend_name="stub"):
        self._loaded = False
        self._name = backend_name

    def load(self) -> None:
        self._loaded = True

    async def generate(self, text, voice, speed=1.0, lang="en-us"):
        yield (np.zeros(100, dtype=np.float32), 24000)

    def list_voices(self):
        return ["voice_a"]

    def is_loaded(self):
        return self._loaded

    @property
    def name(self):
        return self._name


# ---------------------------------------------------------------------------
# Tests: create_backend factory
# ---------------------------------------------------------------------------

class TestCreateBackend:
    """Tests for the create_backend factory function."""

    def test_create_kokoro_backend(self):
        """'kokoro' should return a KokoroBackend instance."""
        cfg = _make_config(engine="kokoro")
        backend = create_backend("kokoro", cfg)
        assert isinstance(backend, KokoroBackend)
        assert backend.name == "kokoro"

    def test_create_kokoro_case_insensitive(self):
        """Engine name should be case-insensitive."""
        cfg = _make_config()
        backend = create_backend("Kokoro", cfg)
        assert isinstance(backend, KokoroBackend)

    def test_create_kokoro_with_whitespace(self):
        """Engine name should be trimmed."""
        cfg = _make_config()
        backend = create_backend("  kokoro  ", cfg)
        assert isinstance(backend, KokoroBackend)

    def test_create_piper_lazy_import(self):
        """'piper' should lazy-import PiperBackend from tts_piper."""
        mock_piper_cls = MagicMock()
        mock_piper_instance = MagicMock(spec=TTSBackend)
        mock_piper_cls.return_value = mock_piper_instance

        mock_module = MagicMock()
        mock_module.PiperBackend = mock_piper_cls

        cfg = _make_config(engine="piper")
        with patch.dict("sys.modules", {"claude_speak.tts_piper": mock_module}):
            backend = create_backend("piper", cfg)
            assert backend is mock_piper_instance
            mock_piper_cls.assert_called_once_with(cfg)

    def test_create_elevenlabs_lazy_import(self):
        """'elevenlabs' should lazy-import ElevenLabsBackend from tts_elevenlabs."""
        mock_el_cls = MagicMock()
        mock_el_instance = MagicMock(spec=TTSBackend)
        mock_el_cls.return_value = mock_el_instance

        mock_module = MagicMock()
        mock_module.ElevenLabsBackend = mock_el_cls

        cfg = _make_config(engine="elevenlabs")
        with patch.dict("sys.modules", {"claude_speak.tts_elevenlabs": mock_module}):
            backend = create_backend("elevenlabs", cfg)
            assert backend is mock_el_instance
            mock_el_cls.assert_called_once_with(cfg)

    def test_unknown_engine_raises_valueerror(self):
        """Unknown engine names should raise ValueError with helpful message."""
        cfg = _make_config()
        with pytest.raises(ValueError, match="Unknown TTS engine.*'nonexistent'"):
            create_backend("nonexistent", cfg)

    def test_unknown_engine_lists_available(self):
        """ValueError message should list available engines."""
        cfg = _make_config()
        with pytest.raises(ValueError, match="kokoro.*piper.*elevenlabs"):
            create_backend("bad_engine", cfg)

    def test_piper_import_error_when_module_missing(self):
        """Should raise ImportError when tts_piper module is not available."""
        cfg = _make_config()
        # Remove the module from sys.modules if it exists, and ensure import fails
        with patch.dict("sys.modules", {"claude_speak.tts_piper": None}):
            with pytest.raises(ImportError, match="Piper backend"):
                create_backend("piper", cfg)

    def test_elevenlabs_import_error_when_module_missing(self):
        """Should raise ImportError when tts_elevenlabs module is not available."""
        cfg = _make_config()
        with patch.dict("sys.modules", {"claude_speak.tts_elevenlabs": None}):
            with pytest.raises(ImportError, match="ElevenLabs backend"):
                create_backend("elevenlabs", cfg)

    def test_available_engines_constant(self):
        """AVAILABLE_ENGINES should list all supported engines."""
        assert "kokoro" in AVAILABLE_ENGINES
        assert "piper" in AVAILABLE_ENGINES
        assert "elevenlabs" in AVAILABLE_ENGINES


# ---------------------------------------------------------------------------
# Tests: TTSEngine.swap_backend
# ---------------------------------------------------------------------------

class TestSwapBackend:
    """Tests for TTSEngine.swap_backend."""

    def test_swap_replaces_backend(self):
        """swap_backend should replace the internal _backend."""
        cfg = _make_config()
        old_backend = _StubBackend("old")
        new_backend = _StubBackend("new")
        engine = TTSEngine(cfg, backend=old_backend)

        engine.swap_backend(new_backend)

        assert engine._backend is new_backend

    def test_swap_stops_playback(self):
        """swap_backend should call stop() to halt current playback."""
        cfg = _make_config()
        backend = _StubBackend("initial")
        engine = TTSEngine(cfg, backend=backend)

        # Set up a mock stream to verify stop() is called
        mock_stream = MagicMock()
        engine._stream = mock_stream

        new_backend = _StubBackend("replacement")
        engine.swap_backend(new_backend)

        # stop() should have aborted and closed the stream
        mock_stream.abort.assert_called_once()
        mock_stream.close.assert_called_once()
        assert engine._stream is None

    def test_swap_clears_stopped_event_state(self):
        """After swap, the _stopped event should be set (from stop call)."""
        cfg = _make_config()
        old_backend = _StubBackend("old")
        engine = TTSEngine(cfg, backend=old_backend)

        new_backend = _StubBackend("new")
        engine.swap_backend(new_backend)

        # stop() sets the _stopped event
        assert engine._stopped.is_set()

    def test_swap_logs_transition(self, caplog):
        """swap_backend should log the backend transition."""
        cfg = _make_config()
        old_backend = _StubBackend("alpha")
        new_backend = _StubBackend("beta")
        engine = TTSEngine(cfg, backend=old_backend)

        with caplog.at_level(logging.INFO, logger="claude_speak.tts"):
            engine.swap_backend(new_backend)

        assert any(
            "Backend swapped: alpha" in r.message and "beta" in r.message
            for r in caplog.records
        ), f"Expected swap log message, got: {[r.message for r in caplog.records]}"

    def test_swap_backend_works_with_no_active_stream(self):
        """swap_backend should work even if no stream exists."""
        cfg = _make_config()
        backend = _StubBackend("old")
        engine = TTSEngine(cfg, backend=backend)
        assert engine._stream is None

        new_backend = _StubBackend("new")
        # Should not raise
        engine.swap_backend(new_backend)
        assert engine._backend is new_backend

    def test_swap_backend_new_backend_usable(self):
        """After swap, the engine should use the new backend for generation."""
        cfg = _make_config()
        old_backend = _StubBackend("old")
        old_backend._loaded = True
        new_backend = _StubBackend("new")
        new_backend._loaded = True

        engine = TTSEngine(cfg, backend=old_backend)
        engine.swap_backend(new_backend)

        # generate_audio should use the new backend
        result = asyncio.run(engine.generate_audio("test"))
        assert len(result) == 1
        # Verify it came from the new backend (stub yields 100-sample chunks)
        assert len(result[0][0]) == 100


# ---------------------------------------------------------------------------
# Tests: Daemon config reload engine swap
# ---------------------------------------------------------------------------

class TestDaemonEngineSwap:
    """Tests for _try_reload_config detecting engine changes."""

    def test_engine_change_triggers_swap(self, tmp_path):
        """When tts.engine changes in config, the backend should be swapped."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nengine = 'piper'\n")

        old_config = Config(tts=TTSConfig(engine="kokoro"))
        new_config = Config(tts=TTSConfig(engine="piper"))
        engine = MagicMock()
        old_mtime = 100.0

        mock_backend = MagicMock(spec=TTSBackend)
        mock_backend.name = "piper"

        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config), \
             patch("claude_speak.daemon.create_backend", return_value=mock_backend) as mock_create:
            result_config, result_mtime = _try_reload_config(old_config, engine, old_mtime)

            # create_backend should have been called with the new engine name
            mock_create.assert_called_once_with("piper", new_config)
            # Backend should have been loaded
            mock_backend.load.assert_called_once()
            # Engine swap should have been called
            engine.swap_backend.assert_called_once_with(mock_backend)

    def test_same_engine_no_swap(self, tmp_path):
        """When tts.engine stays the same, no backend swap occurs."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nengine = 'kokoro'\n")

        old_config = Config(tts=TTSConfig(engine="kokoro"))
        new_config = Config(tts=TTSConfig(engine="kokoro"))
        engine = MagicMock()
        old_mtime = 100.0

        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config), \
             patch("claude_speak.daemon.create_backend") as mock_create:
            _try_reload_config(old_config, engine, old_mtime)

            # No swap should have happened
            mock_create.assert_not_called()
            engine.swap_backend.assert_not_called()

    def test_engine_swap_failure_keeps_old_engine(self, tmp_path, caplog):
        """If backend creation fails, the old engine should be kept."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nengine = 'piper'\n")

        old_config = Config(tts=TTSConfig(engine="kokoro"))
        new_config = Config(tts=TTSConfig(engine="piper"))
        engine = MagicMock()
        old_mtime = 100.0

        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config), \
             patch("claude_speak.daemon.create_backend", side_effect=ValueError("test error")), \
             caplog.at_level(logging.ERROR, logger="claude_speak.daemon"):
            result_config, _ = _try_reload_config(old_config, engine, old_mtime)

            # Swap should NOT have been called
            engine.swap_backend.assert_not_called()
            # Config engine should be reverted to old value
            assert result_config.tts.engine == "kokoro"
            # Error should be logged
            assert any("Failed to swap" in r.message for r in caplog.records)

    def test_engine_swap_load_failure_keeps_old_engine(self, tmp_path, caplog):
        """If new backend.load() fails, the old engine should be kept."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nengine = 'elevenlabs'\n")

        old_config = Config(tts=TTSConfig(engine="kokoro"))
        new_config = Config(tts=TTSConfig(engine="elevenlabs"))
        engine = MagicMock()
        old_mtime = 100.0

        mock_backend = MagicMock(spec=TTSBackend)
        mock_backend.load.side_effect = RuntimeError("API key missing")

        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config), \
             patch("claude_speak.daemon.create_backend", return_value=mock_backend), \
             caplog.at_level(logging.ERROR, logger="claude_speak.daemon"):
            result_config, _ = _try_reload_config(old_config, engine, old_mtime)

            # Swap should NOT have been called (load failed before swap)
            engine.swap_backend.assert_not_called()
            # Config engine should be reverted
            assert result_config.tts.engine == "kokoro"

    def test_engine_swap_logs_success(self, tmp_path, caplog):
        """Successful engine swap should log the transition."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\nengine = 'piper'\n")

        old_config = Config(tts=TTSConfig(engine="kokoro"))
        new_config = Config(tts=TTSConfig(engine="piper"))
        engine = MagicMock()
        old_mtime = 100.0

        mock_backend = MagicMock(spec=TTSBackend)
        mock_backend.name = "piper"

        with patch("claude_speak.daemon.CONFIG_PATH", config_file), \
             patch("claude_speak.daemon.load_config", return_value=new_config), \
             patch("claude_speak.daemon.create_backend", return_value=mock_backend), \
             caplog.at_level(logging.INFO, logger="claude_speak.daemon"):
            _try_reload_config(old_config, engine, old_mtime)

        assert any("Engine swap complete" in r.message for r in caplog.records)
        assert any("kokoro" in r.message and "piper" in r.message for r in caplog.records)

    def test_config_mtime_unchanged_no_reload(self, tmp_path):
        """No reload should happen when mtime hasn't changed."""
        from claude_speak.daemon import _try_reload_config

        config_file = tmp_path / "claude-speak.toml"
        config_file.write_text("[tts]\n")
        current_mtime = config_file.stat().st_mtime

        old_config = Config()
        engine = MagicMock()

        with patch("claude_speak.daemon.CONFIG_PATH", config_file):
            result_config, result_mtime = _try_reload_config(old_config, engine, current_mtime)
            assert result_config is old_config
            assert result_mtime == current_mtime
