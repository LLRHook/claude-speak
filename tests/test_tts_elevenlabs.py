"""
Unit tests for the ElevenLabs TTS backend.

All external dependencies (elevenlabs SDK) are mocked — tests run without
the elevenlabs package installed and without network access.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.config import Config, TTSConfig
from claude_speak.tts_base import TTSBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    voice: str = "JBFqnCBsd6RMkjVDRZzb",
    speed: float = 1.0,
    engine: str = "elevenlabs",
    elevenlabs_api_key: str = "",
) -> Config:
    """Create a Config with ElevenLabs TTS settings."""
    return Config(tts=TTSConfig(
        engine=engine,
        voice=voice,
        speed=speed,
        elevenlabs_api_key=elevenlabs_api_key,
    ))


def _fake_elevenlabs_modules():
    """Create mock elevenlabs modules for sys.modules patching.

    Returns a dict suitable for use with patch.dict("sys.modules", ...).
    """
    # Create the main elevenlabs module
    mock_elevenlabs = MagicMock()

    # ElevenLabs client class
    mock_client_class = MagicMock()
    mock_elevenlabs.ElevenLabs = mock_client_class

    # AsyncElevenLabs client class
    mock_async_client_class = MagicMock()
    mock_elevenlabs.client = MagicMock()
    mock_elevenlabs.client.AsyncElevenLabs = mock_async_client_class

    # VoiceSettings
    mock_voice_settings = MagicMock()
    mock_elevenlabs.VoiceSettings = mock_voice_settings

    return {
        "elevenlabs": mock_elevenlabs,
        "elevenlabs.client": mock_elevenlabs.client,
    }


def _generate_pcm_bytes(n_samples: int = 4800, amplitude: float = 0.5) -> bytes:
    """Generate synthetic PCM int16 bytes for testing.

    Default: 4800 samples = 0.2s at 24kHz.
    """
    t = np.linspace(0, 1, n_samples, dtype=np.float32)
    signal = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    return signal.tobytes()


# ---------------------------------------------------------------------------
# Tests: ElevenLabsBackend implements TTSBackend
# ---------------------------------------------------------------------------

class TestElevenLabsBackendInterface:
    """Verify ElevenLabsBackend correctly implements the TTSBackend ABC."""

    def test_is_subclass_of_tts_backend(self):
        """ElevenLabsBackend should be a subclass of TTSBackend."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            assert issubclass(ElevenLabsBackend, TTSBackend)

    def test_can_be_instantiated(self):
        """ElevenLabsBackend should be instantiable (all abstract methods implemented)."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            config = _make_config()
            backend = ElevenLabsBackend(config)
            assert backend is not None

    def test_has_required_methods(self):
        """ElevenLabsBackend should expose all TTSBackend methods."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            config = _make_config()
            backend = ElevenLabsBackend(config)
            assert callable(backend.load)
            assert callable(backend.generate)
            assert callable(backend.list_voices)
            assert callable(backend.is_loaded)
            assert hasattr(backend, "name")


# ---------------------------------------------------------------------------
# Tests: name property
# ---------------------------------------------------------------------------

class TestElevenLabsBackendName:
    """Tests for the name property."""

    def test_name_returns_elevenlabs(self):
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            assert backend.name == "elevenlabs"

    def test_name_is_string(self):
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            assert isinstance(backend.name, str)


# ---------------------------------------------------------------------------
# Tests: is_loaded state transitions
# ---------------------------------------------------------------------------

class TestElevenLabsBackendIsLoaded:
    """Tests for is_loaded() before and after load()."""

    def test_not_loaded_initially(self):
        """is_loaded() should return False before load() is called."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            assert backend.is_loaded() is False

    def test_loaded_after_load(self):
        """is_loaded() should return True after successful load()."""
        modules = _fake_elevenlabs_modules()
        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key-123"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()
            assert backend.is_loaded() is True

    def test_not_loaded_returns_false_type(self):
        """is_loaded() should return an actual bool, not a truthy/falsy value."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            result = backend.is_loaded()
            assert result is False
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests: load()
# ---------------------------------------------------------------------------

class TestElevenLabsBackendLoad:
    """Tests for the load() method."""

    def test_load_without_api_key_raises(self):
        """load() should raise RuntimeError with a helpful message when no API key."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()), \
             patch.dict("os.environ", {}, clear=False):
            # Make sure env var is not set
            import os
            env = os.environ.copy()
            env.pop("ELEVENLABS_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                from claude_speak.tts_elevenlabs import ElevenLabsBackend
                backend = ElevenLabsBackend(_make_config(elevenlabs_api_key=""))
                with pytest.raises(RuntimeError, match="API key not found"):
                    backend.load()

    def test_load_with_env_var_key(self):
        """load() should succeed when ELEVENLABS_API_KEY env var is set."""
        modules = _fake_elevenlabs_modules()
        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key-from-env"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()
            assert backend.is_loaded() is True

    def test_load_with_config_key(self):
        """load() should succeed when config provides elevenlabs_api_key."""
        modules = _fake_elevenlabs_modules()
        env = {k: v for k, v in __import__("os").environ.items() if k != "ELEVENLABS_API_KEY"}
        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", env, clear=True):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config(elevenlabs_api_key="config-key-456"))
            backend.load()
            assert backend.is_loaded() is True

    def test_load_initializes_both_clients(self):
        """load() should create both sync and async ElevenLabs clients."""
        modules = _fake_elevenlabs_modules()
        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()
            assert backend._client is not None
            assert backend._async_client is not None


# ---------------------------------------------------------------------------
# Tests: API key resolution order
# ---------------------------------------------------------------------------

class TestApiKeyResolution:
    """Tests for API key resolution priority."""

    def test_env_var_takes_priority_over_config(self):
        """ELEVENLABS_API_KEY env var should be preferred over config field."""
        modules = _fake_elevenlabs_modules()
        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "env-key"}):
            from claude_speak.tts_elevenlabs import _resolve_api_key
            config = _make_config(elevenlabs_api_key="config-key")
            assert _resolve_api_key(config) == "env-key"

    def test_config_used_when_no_env_var(self):
        """Config elevenlabs_api_key should be used when env var is absent."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "ELEVENLABS_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            from claude_speak.tts_elevenlabs import _resolve_api_key
            config = _make_config(elevenlabs_api_key="config-key-789")
            assert _resolve_api_key(config) == "config-key-789"

    def test_file_config_used_as_last_resort(self):
        """~/.claude-speak/config.toml should be checked when env and config are empty."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "ELEVENLABS_API_KEY"}
        toml_content = b'[elevenlabs]\napi_key = "file-key-abc"\n'

        with patch.dict("os.environ", env, clear=True):
            from claude_speak.tts_elevenlabs import _resolve_api_key
            config = _make_config(elevenlabs_api_key="")

            mock_path = MagicMock(spec=Path)
            mock_path.exists.return_value = True

            with patch("claude_speak.tts_elevenlabs.Path") as MockPath:
                MockPath.home.return_value.__truediv__ = MagicMock(
                    return_value=MagicMock(
                        __truediv__=MagicMock(return_value=mock_path)
                    )
                )
                # Use a simpler approach: patch the path resolution chain
                home_dir = MockPath.home.return_value
                claude_speak_dir = MagicMock()
                home_dir.__truediv__ = MagicMock(return_value=claude_speak_dir)
                config_path = MagicMock()
                config_path.exists.return_value = True
                claude_speak_dir.__truediv__ = MagicMock(return_value=config_path)

                import io
                mock_open = MagicMock(return_value=io.BytesIO(toml_content))

                with patch("builtins.open", mock_open):
                    result = _resolve_api_key(config)
                    assert result == "file-key-abc"

    def test_returns_empty_when_no_key_anywhere(self):
        """_resolve_api_key should return empty string when no key is found."""
        env = {k: v for k, v in __import__("os").environ.items() if k != "ELEVENLABS_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            from claude_speak.tts_elevenlabs import _resolve_api_key
            config = _make_config(elevenlabs_api_key="")
            # Patch Path.home() so the file check fails gracefully
            with patch("claude_speak.tts_elevenlabs.Path") as MockPath:
                mock_config_path = MagicMock()
                mock_config_path.exists.return_value = False
                home = MagicMock()
                home.__truediv__ = MagicMock(return_value=MagicMock(
                    __truediv__=MagicMock(return_value=mock_config_path)
                ))
                MockPath.home.return_value = home
                assert _resolve_api_key(config) == ""


# ---------------------------------------------------------------------------
# Tests: generate yields audio segments
# ---------------------------------------------------------------------------

class TestElevenLabsBackendGenerate:
    """Tests for the generate() async generator."""

    def test_generate_yields_audio_segments(self):
        """generate() should yield (np.ndarray, int) tuples."""
        modules = _fake_elevenlabs_modules()
        # Create PCM data — 2 chunks of 0.2s each
        pcm_chunk_1 = _generate_pcm_bytes(n_samples=24000)  # 1 second
        pcm_chunk_2 = _generate_pcm_bytes(n_samples=12000)  # 0.5 seconds

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            # Mock the stream to return PCM byte chunks
            backend._client.text_to_speech.stream.return_value = iter([
                pcm_chunk_1, pcm_chunk_2
            ])

            async def run():
                segments = []
                async for samples, sr in backend.generate("Hello world", voice="test-voice"):
                    segments.append((samples, sr))
                return segments

            result = asyncio.run(run())
            assert len(result) >= 1
            for samples, sr in result:
                assert isinstance(samples, np.ndarray)
                assert samples.dtype == np.float32
                assert sr == 24000
                # Samples should be normalized to [-1, 1]
                assert np.all(np.abs(samples) <= 1.0)

    def test_generate_calls_stream_with_correct_params(self):
        """generate() should pass voice_id, text, and settings to the API."""
        modules = _fake_elevenlabs_modules()
        pcm_data = _generate_pcm_bytes(n_samples=2400)

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config(speed=1.2))
            backend.load()

            backend._client.text_to_speech.stream.return_value = iter([pcm_data])

            async def run():
                async for _ in backend.generate("test text", voice="my-voice-id", speed=1.2):
                    pass

            asyncio.run(run())

            call_kwargs = backend._client.text_to_speech.stream.call_args
            assert call_kwargs is not None
            _, kwargs = call_kwargs
            assert kwargs["voice_id"] == "my-voice-id"
            assert kwargs["text"] == "test text"
            assert kwargs["output_format"] == "pcm_24000"

    def test_generate_flushes_remaining_buffer(self):
        """generate() should yield remaining buffered audio after stream ends."""
        modules = _fake_elevenlabs_modules()
        # Small chunk that is less than the min_chunk_bytes threshold
        small_pcm = _generate_pcm_bytes(n_samples=100)  # 200 bytes, well under 48000

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            backend._client.text_to_speech.stream.return_value = iter([small_pcm])

            async def run():
                segments = []
                async for samples, sr in backend.generate("hi", voice="v"):
                    segments.append((samples, sr))
                return segments

            result = asyncio.run(run())
            # Even a small chunk should be flushed
            assert len(result) >= 1
            total_samples = sum(len(s) for s, _ in result)
            assert total_samples == 100

    def test_generate_without_load_raises(self):
        """generate() should raise RuntimeError if load() was not called."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())

            async def run():
                async for _ in backend.generate("test", voice="v"):
                    pass

            with pytest.raises(RuntimeError, match="not loaded"):
                asyncio.run(run())

    def test_generate_skips_empty_chunks(self):
        """generate() should gracefully skip empty byte chunks."""
        modules = _fake_elevenlabs_modules()
        pcm_data = _generate_pcm_bytes(n_samples=24000)

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            # Mix in empty chunks and non-bytes items
            backend._client.text_to_speech.stream.return_value = iter([
                b"",  # empty chunk
                pcm_data,
                b"",  # another empty chunk
            ])

            async def run():
                segments = []
                async for samples, sr in backend.generate("test", voice="v"):
                    segments.append((samples, sr))
                return segments

            result = asyncio.run(run())
            total_samples = sum(len(s) for s, _ in result)
            assert total_samples == 24000


# ---------------------------------------------------------------------------
# Tests: list_voices
# ---------------------------------------------------------------------------

class TestElevenLabsBackendListVoices:
    """Tests for list_voices()."""

    def test_list_voices_returns_api_voices(self):
        """list_voices() should return voice names and IDs from the API."""
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            # Mock the voices response
            mock_voice_1 = SimpleNamespace(name="Rachel", voice_id="abc123")
            mock_voice_2 = SimpleNamespace(name="Adam", voice_id="def456")
            mock_response = SimpleNamespace(voices=[mock_voice_1, mock_voice_2])
            backend._client.voices.get_all.return_value = mock_response

            voices = backend.list_voices()
            assert len(voices) == 2
            assert "Rachel (abc123)" in voices
            assert "Adam (def456)" in voices

    def test_list_voices_caches_results(self):
        """list_voices() should cache results after the first API call."""
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            mock_voice = SimpleNamespace(name="Rachel", voice_id="abc123")
            mock_response = SimpleNamespace(voices=[mock_voice])
            backend._client.voices.get_all.return_value = mock_response

            # Call twice
            voices_1 = backend.list_voices()
            voices_2 = backend.list_voices()

            # API should only be called once (cached)
            assert backend._client.voices.get_all.call_count == 1
            assert voices_1 == voices_2

    def test_list_voices_returns_list_type(self):
        """list_voices() should return a list."""
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            mock_response = SimpleNamespace(voices=[])
            backend._client.voices.get_all.return_value = mock_response

            result = backend.list_voices()
            assert isinstance(result, list)

    def test_list_voices_without_load_raises(self):
        """list_voices() should raise RuntimeError if load() was not called."""
        with patch.dict("sys.modules", _fake_elevenlabs_modules()):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())

            with pytest.raises(RuntimeError, match="not loaded"):
                backend.list_voices()


# ---------------------------------------------------------------------------
# Tests: network error handling
# ---------------------------------------------------------------------------

class TestElevenLabsNetworkErrors:
    """Tests for graceful degradation on network failures."""

    def test_generate_network_error_raises(self):
        """generate() should raise on network errors (for TTSEngine to catch)."""
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            backend._client.text_to_speech.stream.side_effect = ConnectionError(
                "Network unreachable"
            )

            async def run():
                async for _ in backend.generate("test", voice="v"):
                    pass

            with pytest.raises(ConnectionError, match="Network unreachable"):
                asyncio.run(run())

    def test_list_voices_network_error_raises(self):
        """list_voices() should raise on network errors."""
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            backend._client.voices.get_all.side_effect = ConnectionError("timeout")

            with pytest.raises(ConnectionError, match="timeout"):
                backend.list_voices()

    def test_generate_logs_warning_on_error(self, caplog):
        """generate() should log a warning before re-raising API errors."""
        import logging
        modules = _fake_elevenlabs_modules()

        with patch.dict("sys.modules", modules), \
             patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}), \
             caplog.at_level(logging.WARNING, logger="claude_speak.tts_elevenlabs"):
            from claude_speak.tts_elevenlabs import ElevenLabsBackend
            backend = ElevenLabsBackend(_make_config())
            backend.load()

            backend._client.text_to_speech.stream.side_effect = Exception("API error")

            async def run():
                async for _ in backend.generate("test", voice="v"):
                    pass

            with pytest.raises(Exception, match="API error"):
                asyncio.run(run())

            assert any("ElevenLabs API error" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: PCM conversion helper
# ---------------------------------------------------------------------------

class TestPcmConversion:
    """Tests for the _pcm_bytes_to_float32 helper."""

    def test_converts_int16_to_float32(self):
        """PCM int16 bytes should be converted to float32 numpy array."""
        from claude_speak.tts_elevenlabs import _pcm_bytes_to_float32
        # Create known int16 data
        int16_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        result = _pcm_bytes_to_float32(int16_data.tobytes())
        assert result.dtype == np.float32
        assert len(result) == 5
        # Check normalization
        assert abs(result[0]) < 1e-6  # 0 -> 0.0
        assert abs(result[1] - 0.5) < 0.01  # 16384 -> ~0.5
        assert abs(result[2] + 0.5) < 0.01  # -16384 -> ~-0.5
        assert result[3] > 0.99  # 32767 -> ~1.0
        assert result[4] <= -0.99  # -32768 -> ~-1.0

    def test_empty_bytes_returns_empty_array(self):
        """Empty input should return an empty numpy array."""
        from claude_speak.tts_elevenlabs import _pcm_bytes_to_float32
        result = _pcm_bytes_to_float32(b"")
        assert len(result) == 0
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: TTSConfig elevenlabs_api_key field
# ---------------------------------------------------------------------------

class TestTTSConfigElevenlabsApiKey:
    """Tests for the elevenlabs_api_key config field."""

    def test_default_is_empty_string(self):
        """elevenlabs_api_key should default to empty string."""
        config = TTSConfig()
        assert config.elevenlabs_api_key == ""

    def test_can_be_set(self):
        """elevenlabs_api_key should be settable."""
        config = TTSConfig(elevenlabs_api_key="my-api-key")
        assert config.elevenlabs_api_key == "my-api-key"

    def test_available_in_config(self):
        """elevenlabs_api_key should be available on Config.tts."""
        config = _make_config(elevenlabs_api_key="test-key")
        assert config.tts.elevenlabs_api_key == "test-key"
