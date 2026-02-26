"""
Unit tests for TTS backend abstraction layer.

Tests the TTSBackend ABC, KokoroBackend, and TTSEngine with mock backends.
All external dependencies (kokoro_onnx, sounddevice) are mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from claude_speak.config import Config, TTSConfig
from claude_speak.tts_base import TTSBackend
import claude_speak.tts as tts_module
from claude_speak.tts import KokoroBackend, TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(voice="af_sarah", speed=1.0, device="auto", volume=1.0, engine="kokoro"):
    """Create a Config with the given TTS settings."""
    return Config(tts=TTSConfig(engine=engine, voice=voice, speed=speed, device=device, volume=volume))


def _fake_kokoro():
    """Return a mock Kokoro instance with common stubs."""
    mock = MagicMock()
    mock.get_voice_style.return_value = np.ones(256, dtype=np.float32)
    mock.get_voices.return_value = ["af_sarah", "bm_fable", "bm_george"]
    return mock


class _StubBackend(TTSBackend):
    """Minimal concrete backend for testing the interface."""

    def __init__(self):
        self._loaded = False
        self._voices = ["voice_a", "voice_b"]

    def load(self) -> None:
        self._loaded = True

    async def generate(self, text, voice, speed=1.0, lang="en-us"):
        yield (np.zeros(100, dtype=np.float32), 24000)

    def list_voices(self):
        return list(self._voices)

    def is_loaded(self):
        return self._loaded

    @property
    def name(self):
        return "stub"


# ---------------------------------------------------------------------------
# Tests: TTSBackend is abstract
# ---------------------------------------------------------------------------

class TestTTSBackendAbstract:
    """Verify TTSBackend cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self):
        """TTSBackend is abstract — instantiating it must raise TypeError."""
        with pytest.raises(TypeError):
            TTSBackend()

    def test_incomplete_subclass_raises(self):
        """A subclass that doesn't implement all methods cannot be instantiated."""
        class Partial(TTSBackend):
            def load(self):
                pass
            # Missing: generate, list_voices, is_loaded, name

        with pytest.raises(TypeError):
            Partial()

    def test_complete_subclass_works(self):
        """A fully implemented subclass can be instantiated."""
        backend = _StubBackend()
        assert backend.name == "stub"


# ---------------------------------------------------------------------------
# Tests: KokoroBackend.name
# ---------------------------------------------------------------------------

class TestKokoroBackendName:
    """Tests for KokoroBackend.name property."""

    def test_name_is_kokoro(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        assert backend.name == "kokoro"

    def test_name_type_is_string(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        assert isinstance(backend.name, str)


# ---------------------------------------------------------------------------
# Tests: KokoroBackend.is_loaded before/after load
# ---------------------------------------------------------------------------

class TestKokoroBackendIsLoaded:
    """Tests for KokoroBackend.is_loaded state transitions."""

    def test_not_loaded_initially(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        assert backend.is_loaded() is False

    def test_loaded_after_load(self):
        """After calling load(), is_loaded() returns True."""
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        # Directly set the internal kokoro instance (skip real model loading)
        backend._kokoro = _fake_kokoro()
        backend._voice_style = "af_sarah"
        assert backend.is_loaded() is True

    def test_loaded_after_full_load_with_mock(self):
        """load() sets the model and is_loaded() reflects that."""
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        mock_kokoro_cls = MagicMock()
        mock_instance = _fake_kokoro()
        mock_kokoro_cls.return_value = mock_instance

        with patch("claude_speak.tts.Kokoro", mock_kokoro_cls, create=True), \
             patch("claude_speak.tts.Path") as mock_path:
            # Make both model files appear to exist
            mock_path.return_value.exists.return_value = True
            mock_path.return_value.name = "kokoro-v1.0.onnx"
            # Patch the Path() calls within load
            with patch("pathlib.Path.exists", return_value=True), \
                 patch("pathlib.Path.name", new_callable=lambda: property(lambda self: "kokoro-v1.0.onnx")):
                # Use a simpler approach: just mock kokoro_onnx import
                with patch.dict("sys.modules", {"kokoro_onnx": MagicMock(Kokoro=mock_kokoro_cls)}):
                    backend.load()

        assert backend.is_loaded() is True


# ---------------------------------------------------------------------------
# Tests: KokoroBackend.list_voices
# ---------------------------------------------------------------------------

class TestKokoroBackendListVoices:
    """Tests for KokoroBackend.list_voices."""

    def test_list_voices_returns_list(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        backend._kokoro = _fake_kokoro()
        voices = backend.list_voices()
        assert voices == ["af_sarah", "bm_fable", "bm_george"]
        assert isinstance(voices, list)

    def test_list_voices_calls_kokoro_get_voices(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        backend._kokoro = _fake_kokoro()
        backend.list_voices()
        backend._kokoro.get_voices.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: KokoroBackend.generate
# ---------------------------------------------------------------------------

class TestKokoroBackendGenerate:
    """Tests for KokoroBackend.generate streaming."""

    def test_generate_yields_segments(self):
        cfg = _make_config()
        backend = KokoroBackend(cfg)
        backend._kokoro = _fake_kokoro()
        backend._voice_style = "af_sarah"

        async def fake_stream(text, voice, speed, lang):
            yield (np.zeros(500, dtype=np.float32), 24000)
            yield (np.zeros(300, dtype=np.float32), 24000)

        backend._kokoro.create_stream = fake_stream

        async def run():
            segments = []
            async for samples, sr in backend.generate("Hello", voice="af_sarah"):
                segments.append((samples, sr))
            return segments

        result = asyncio.run(run())
        assert len(result) == 2
        assert result[0][1] == 24000
        assert len(result[0][0]) == 500


# ---------------------------------------------------------------------------
# Tests: TTSEngine with a mock backend
# ---------------------------------------------------------------------------

class TestTTSEngineWithMockBackend:
    """Tests for TTSEngine using a custom TTSBackend."""

    def test_engine_accepts_custom_backend(self):
        """TTSEngine can be created with a non-Kokoro backend."""
        cfg = _make_config()
        backend = _StubBackend()
        engine = TTSEngine(cfg, backend=backend)
        assert engine._backend is backend

    def test_engine_defaults_to_kokoro_backend(self):
        """TTSEngine creates a KokoroBackend when no backend is given."""
        cfg = _make_config()
        engine = TTSEngine(cfg)
        assert isinstance(engine._backend, KokoroBackend)

    def test_list_voices_delegates_to_backend(self):
        """list_voices() should call the backend's list_voices()."""
        cfg = _make_config()
        backend = _StubBackend()
        backend._loaded = True
        engine = TTSEngine(cfg, backend=backend)
        voices = engine.list_voices()
        assert voices == ["voice_a", "voice_b"]

    def test_generate_audio_delegates_to_backend(self):
        """generate_audio() should use the backend's generate()."""
        cfg = _make_config()
        backend = _StubBackend()
        backend._loaded = True
        engine = TTSEngine(cfg, backend=backend)
        result = asyncio.run(engine.generate_audio("test"))
        assert len(result) == 1
        assert result[0][1] == 24000
        assert len(result[0][0]) == 100

    def test_load_delegates_to_backend(self):
        """load() should call the backend's load()."""
        cfg = _make_config()
        backend = _StubBackend()

        mock_dm = MagicMock()
        mock_dm.resolve_output.return_value = 0
        mock_dm.get_device_name.return_value = "Mock Speaker"

        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = TTSEngine(cfg, backend=backend)
            engine.load()

        assert backend.is_loaded() is True

    def test_speak_delegates_to_backend_generate(self):
        """speak() should stream from the backend and play audio."""
        cfg = _make_config()
        backend = _StubBackend()
        backend._loaded = True

        mock_dm = MagicMock()
        mock_dm.maybe_resolve_output.return_value = 0
        mock_dm.get_default_output.return_value = 0
        mock_dm.is_device_available.return_value = True

        mock_sd = MagicMock()
        mock_stream = MagicMock(active=True)
        mock_sd.OutputStream.return_value = mock_stream
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})

        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm), \
             patch.object(tts_module, "sd", mock_sd):
            engine = TTSEngine(cfg, backend=backend)
            asyncio.run(engine.speak("Hello world"))

        # The stream should have been written to
        assert mock_stream.write.called

    def test_generate_audio_auto_loads_backend(self):
        """generate_audio() should call load() if backend is not loaded."""
        cfg = _make_config()
        backend = _StubBackend()
        assert not backend.is_loaded()

        mock_dm = MagicMock()
        mock_dm.resolve_output.return_value = 0
        mock_dm.get_device_name.return_value = "Mock Speaker"

        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = TTSEngine(cfg, backend=backend)
            result = asyncio.run(engine.generate_audio("test"))

        assert backend.is_loaded()
        assert len(result) == 1

    def test_stop_works_with_custom_backend(self):
        """stop() should work regardless of backend type."""
        cfg = _make_config()
        backend = _StubBackend()
        engine = TTSEngine(cfg, backend=backend)
        mock_stream = MagicMock()
        engine._stream = mock_stream
        engine.stop()
        assert engine._stopped.is_set()
        mock_stream.abort.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: TTSConfig engine field
# ---------------------------------------------------------------------------

class TestTTSConfigEngine:
    """Tests for the tts.engine config option."""

    def test_default_engine_is_kokoro(self):
        cfg = TTSConfig()
        assert cfg.engine == "kokoro"

    def test_engine_can_be_set(self):
        cfg = TTSConfig(engine="piper")
        assert cfg.engine == "piper"

    def test_engine_from_config(self):
        cfg = _make_config(engine="elevenlabs")
        assert cfg.tts.engine == "elevenlabs"
