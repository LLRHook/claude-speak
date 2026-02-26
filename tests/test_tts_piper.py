"""
Unit tests for the Piper TTS backend.

All external dependencies (piper, piper-tts) are mocked so tests run
without piper installed or any model files present.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.config import Config, TTSConfig
from claude_speak.tts_base import TTSBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(voice="en_US-lessac-medium", speed=1.0, engine="piper"):
    """Create a Config with the given Piper TTS settings."""
    return Config(tts=TTSConfig(engine=engine, voice=voice, speed=speed))


def _fake_piper_voice(sample_rate=22050):
    """Return a mock PiperVoice instance with common stubs."""
    mock = MagicMock()
    # Mock the config attribute for sample_rate
    mock.config.sample_rate = sample_rate

    # synthesize_stream_raw yields raw 16-bit PCM bytes per sentence
    def fake_stream_raw(text, length_scale=1.0, sentence_silence=0.2):
        # Generate 100 samples of 16-bit PCM silence per sentence
        samples = np.zeros(100, dtype=np.int16)
        yield samples.tobytes()

    mock.synthesize_stream_raw = fake_stream_raw
    return mock


def _install_piper_mock():
    """Install a mock 'piper' module in sys.modules and return the mock PiperVoice class."""
    mock_piper_module = MagicMock()
    mock_piper_voice_cls = MagicMock()
    mock_piper_module.PiperVoice = mock_piper_voice_cls
    return mock_piper_module, mock_piper_voice_cls


# ---------------------------------------------------------------------------
# Tests: PiperBackend implements TTSBackend
# ---------------------------------------------------------------------------

class TestPiperBackendInterface:
    """Verify PiperBackend is a proper TTSBackend subclass."""

    def test_is_subclass_of_tts_backend(self):
        """PiperBackend should inherit from TTSBackend."""
        from claude_speak.tts_piper import PiperBackend
        assert issubclass(PiperBackend, TTSBackend)

    def test_can_be_instantiated(self):
        """PiperBackend can be created without errors."""
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        assert backend is not None

    def test_has_all_abstract_methods(self):
        """PiperBackend implements all required abstract methods."""
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        assert hasattr(backend, "load")
        assert hasattr(backend, "generate")
        assert hasattr(backend, "list_voices")
        assert hasattr(backend, "is_loaded")
        assert hasattr(backend, "name")


# ---------------------------------------------------------------------------
# Tests: name property
# ---------------------------------------------------------------------------

class TestPiperBackendName:
    """Tests for PiperBackend.name property."""

    def test_name_is_piper(self):
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        assert backend.name == "piper"

    def test_name_type_is_string(self):
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        assert isinstance(backend.name, str)


# ---------------------------------------------------------------------------
# Tests: is_loaded state transitions
# ---------------------------------------------------------------------------

class TestPiperBackendIsLoaded:
    """Tests for PiperBackend.is_loaded state transitions."""

    def test_not_loaded_initially(self):
        """is_loaded() returns False before load() is called."""
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        assert backend.is_loaded() is False

    def test_loaded_after_setting_voice(self):
        """is_loaded() returns True after the internal voice is set."""
        from claude_speak.tts_piper import PiperBackend
        cfg = _make_config()
        backend = PiperBackend(cfg)
        backend._voice = _fake_piper_voice()
        assert backend.is_loaded() is True

    def test_loaded_after_load_with_mock(self):
        """load() sets internal state and is_loaded() returns True."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="en_US-lessac-medium")
        backend = PiperBackend(cfg)

        mock_piper_module, mock_piper_voice_cls = _install_piper_mock()
        mock_voice_instance = _fake_piper_voice()
        mock_piper_voice_cls.load.return_value = mock_voice_instance

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path") as mock_resolve:
            # Pretend the model file exists
            mock_resolve.return_value = Path("/fake/en_US-lessac-medium.onnx")
            with patch.object(Path, "exists", return_value=True):
                backend.load()

        assert backend.is_loaded() is True


# ---------------------------------------------------------------------------
# Tests: load with mocked piper
# ---------------------------------------------------------------------------

class TestPiperBackendLoad:
    """Tests for PiperBackend.load()."""

    def test_load_calls_piper_voice_load(self):
        """load() should call PiperVoice.load with the resolved model path."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="en_US-lessac-medium")
        backend = PiperBackend(cfg)

        mock_piper_module, mock_piper_voice_cls = _install_piper_mock()
        mock_voice_instance = _fake_piper_voice()
        mock_piper_voice_cls.load.return_value = mock_voice_instance

        fake_model_path = Path("/fake/models/piper/en_US-lessac-medium.onnx")

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path") as mock_resolve:
            mock_resolve.return_value = fake_model_path
            with patch.object(Path, "exists", return_value=True):
                backend.load()

        mock_piper_voice_cls.load.assert_called_once_with(str(fake_model_path))

    def test_load_downloads_if_model_missing(self):
        """load() should download the model if it's not cached."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="en_US-lessac-medium")
        backend = PiperBackend(cfg)

        mock_piper_module, mock_piper_voice_cls = _install_piper_mock()
        mock_voice_instance = _fake_piper_voice()
        mock_piper_voice_cls.load.return_value = mock_voice_instance

        downloaded_path = Path("/fake/models/piper/en_US-lessac-medium.onnx")

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path", return_value=None), \
             patch("claude_speak.tts_piper.download_piper_voice", return_value=downloaded_path) as mock_download:
            backend.load()

        mock_download.assert_called_once_with("en_US-lessac-medium")

    def test_load_raises_import_error_without_piper(self):
        """load() raises ImportError if piper-tts is not installed."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        # Remove piper from sys.modules if present, and make import fail
        with patch.dict("sys.modules", {"piper": None}):
            with pytest.raises(ImportError, match="piper-tts is not installed"):
                backend.load()

    def test_load_raises_file_not_found_for_unknown_voice(self):
        """load() raises FileNotFoundError for an unregistered voice name."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="xx_XX-unknown-voice")
        backend = PiperBackend(cfg)

        mock_piper_module, _ = _install_piper_mock()

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path", return_value=None):
            with pytest.raises(FileNotFoundError, match="Piper model not found"):
                backend.load()

    def test_load_reads_sample_rate_from_voice_config(self):
        """load() should read the sample rate from the voice's config."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="en_US-lessac-medium")
        backend = PiperBackend(cfg)

        mock_piper_module, mock_piper_voice_cls = _install_piper_mock()
        mock_voice_instance = _fake_piper_voice(sample_rate=16000)
        mock_piper_voice_cls.load.return_value = mock_voice_instance

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path") as mock_resolve:
            mock_resolve.return_value = Path("/fake/en_US-lessac-medium.onnx")
            with patch.object(Path, "exists", return_value=True):
                backend.load()

        assert backend._sample_rate == 16000


# ---------------------------------------------------------------------------
# Tests: generate yields audio segments
# ---------------------------------------------------------------------------

class TestPiperBackendGenerate:
    """Tests for PiperBackend.generate streaming."""

    def test_generate_yields_segments(self):
        """generate() should yield (float32_array, sample_rate) tuples."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(speed=1.0)
        backend = PiperBackend(cfg)
        backend._voice = _fake_piper_voice(sample_rate=22050)
        backend._sample_rate = 22050

        async def run():
            segments = []
            async for samples, sr in backend.generate("Hello world.", voice="en_US-lessac-medium"):
                segments.append((samples, sr))
            return segments

        result = asyncio.run(run())
        assert len(result) >= 1
        assert result[0][1] == 22050  # sample rate
        assert result[0][0].dtype == np.float32

    def test_generate_multiple_sentences(self):
        """generate() should yield one segment per sentence."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(speed=1.0)
        backend = PiperBackend(cfg)

        # Create a mock that yields two sentence chunks
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        def fake_stream_raw(text, length_scale=1.0, sentence_silence=0.2):
            yield np.zeros(200, dtype=np.int16).tobytes()
            yield np.ones(150, dtype=np.int16).tobytes()

        mock_voice.synthesize_stream_raw = fake_stream_raw
        backend._voice = mock_voice
        backend._sample_rate = 22050

        async def run():
            segments = []
            async for samples, sr in backend.generate("Hello. World.", voice="en_US-lessac-medium"):
                segments.append((samples, sr))
            return segments

        result = asyncio.run(run())
        assert len(result) == 2
        assert len(result[0][0]) == 200
        assert len(result[1][0]) == 150

    def test_generate_converts_int16_to_float32(self):
        """generate() should convert 16-bit PCM to float32 in [-1, 1] range."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        # Create known int16 values
        int16_samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)

        def fake_stream_raw(text, length_scale=1.0, sentence_silence=0.2):
            yield int16_samples.tobytes()

        mock_voice.synthesize_stream_raw = fake_stream_raw
        backend._voice = mock_voice
        backend._sample_rate = 22050

        async def run():
            segments = []
            async for samples, sr in backend.generate("Test", voice="test"):
                segments.append((samples, sr))
            return segments

        result = asyncio.run(run())
        samples = result[0][0]
        assert samples.dtype == np.float32
        # Check conversion: 16384 / 32768 = 0.5
        np.testing.assert_allclose(samples[1], 16384.0 / 32768.0, atol=1e-5)
        np.testing.assert_allclose(samples[2], -16384.0 / 32768.0, atol=1e-5)
        # Max int16 (32767) should be ~1.0
        assert samples[3] > 0.99

    def test_generate_raises_if_not_loaded(self):
        """generate() should raise RuntimeError if load() has not been called."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        async def run():
            async for _ in backend.generate("Hello", voice="test"):
                pass

        with pytest.raises(RuntimeError, match="load\\(\\) must be called"):
            asyncio.run(run())

    def test_generate_applies_speed_as_length_scale(self):
        """generate() should convert speed to length_scale and pass it to piper."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(speed=2.0)
        backend = PiperBackend(cfg)

        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050

        calls = []

        def fake_stream_raw(text, length_scale=1.0, sentence_silence=0.2):
            calls.append({"length_scale": length_scale})
            yield np.zeros(100, dtype=np.int16).tobytes()

        mock_voice.synthesize_stream_raw = fake_stream_raw
        backend._voice = mock_voice
        backend._sample_rate = 22050

        async def run():
            async for _ in backend.generate("Hello", voice="test", speed=2.0):
                pass

        asyncio.run(run())
        assert len(calls) == 1
        # speed=2.0 -> length_scale=0.5
        assert abs(calls[0]["length_scale"] - 0.5) < 1e-5


# ---------------------------------------------------------------------------
# Tests: list_voices
# ---------------------------------------------------------------------------

class TestPiperBackendListVoices:
    """Tests for PiperBackend.list_voices."""

    def test_list_voices_returns_list(self):
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        with patch("claude_speak.tts_piper.list_downloaded_piper_voices", return_value=["en_US-lessac-medium"]):
            voices = backend.list_voices()

        assert isinstance(voices, list)
        assert "en_US-lessac-medium" in voices

    def test_list_voices_includes_known_voices(self):
        """list_voices should include known (but not yet downloaded) voices."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        with patch("claude_speak.tts_piper.list_downloaded_piper_voices", return_value=[]):
            voices = backend.list_voices()

        # Should contain all three registered voices even if none downloaded
        assert "en_US-lessac-medium" in voices
        assert "en_US-ryan-medium" in voices
        assert "en_GB-alan-medium" in voices

    def test_list_voices_no_duplicates(self):
        """Downloaded voices should not be listed twice."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config()
        backend = PiperBackend(cfg)

        with patch("claude_speak.tts_piper.list_downloaded_piper_voices",
                    return_value=["en_US-lessac-medium", "en_US-ryan-medium"]):
            voices = backend.list_voices()

        # No duplicates
        assert len(voices) == len(set(voices))


# ---------------------------------------------------------------------------
# Tests: speed conversion
# ---------------------------------------------------------------------------

class TestSpeedConversion:
    """Tests for speed-to-length_scale conversion."""

    def test_speed_1_0_gives_length_scale_1_0(self):
        from claude_speak.tts_piper import PiperBackend
        assert PiperBackend.speed_to_length_scale(1.0) == 1.0

    def test_speed_1_5_gives_length_scale_approx_0_67(self):
        from claude_speak.tts_piper import PiperBackend
        result = PiperBackend.speed_to_length_scale(1.5)
        assert abs(result - (1.0 / 1.5)) < 1e-5
        # ~0.6667
        assert abs(result - 0.6667) < 0.001

    def test_speed_2_0_gives_length_scale_0_5(self):
        from claude_speak.tts_piper import PiperBackend
        assert PiperBackend.speed_to_length_scale(2.0) == 0.5

    def test_speed_0_5_gives_length_scale_2_0(self):
        from claude_speak.tts_piper import PiperBackend
        assert PiperBackend.speed_to_length_scale(0.5) == 2.0

    def test_speed_zero_gives_length_scale_1_0(self):
        """Speed of 0 should not cause division by zero; returns default 1.0."""
        from claude_speak.tts_piper import PiperBackend
        assert PiperBackend.speed_to_length_scale(0.0) == 1.0

    def test_speed_negative_gives_length_scale_1_0(self):
        """Negative speed should return default 1.0."""
        from claude_speak.tts_piper import PiperBackend
        assert PiperBackend.speed_to_length_scale(-1.0) == 1.0


# ---------------------------------------------------------------------------
# Tests: missing model handling
# ---------------------------------------------------------------------------

class TestMissingModelHandling:
    """Tests for error handling when models are missing."""

    def test_unknown_voice_raises_file_not_found(self):
        """Requesting an unknown voice that is not in PIPER_VOICES raises FileNotFoundError."""
        from claude_speak.tts_piper import PiperBackend

        cfg = _make_config(voice="nonexistent-voice-model")
        backend = PiperBackend(cfg)

        mock_piper_module, _ = _install_piper_mock()

        with patch.dict("sys.modules", {"piper": mock_piper_module}), \
             patch("claude_speak.tts_piper.PiperBackend._resolve_model_path", return_value=None):
            with pytest.raises(FileNotFoundError, match="Piper model not found"):
                backend.load()

    def test_resolve_model_path_returns_none_for_missing(self):
        """_resolve_model_path returns None when model file does not exist."""
        from claude_speak.tts_piper import PiperBackend

        with patch("claude_speak.tts_piper.PIPER_MODELS_DIR", Path("/nonexistent/dir")):
            result = PiperBackend._resolve_model_path("en_US-lessac-medium")
        assert result is None

    def test_resolve_model_path_finds_cached_model(self, tmp_path):
        """_resolve_model_path returns the path when model exists in cache."""
        from claude_speak.tts_piper import PiperBackend

        # Create a fake model file in a temp directory
        model_file = tmp_path / "en_US-lessac-medium.onnx"
        model_file.touch()

        with patch("claude_speak.tts_piper.PIPER_MODELS_DIR", tmp_path):
            result = PiperBackend._resolve_model_path("en_US-lessac-medium")
        assert result == model_file


# ---------------------------------------------------------------------------
# Tests: Piper voice registry in models.py
# ---------------------------------------------------------------------------

class TestPiperVoiceRegistry:
    """Tests for the PIPER_VOICES registry."""

    def test_piper_voices_has_three_entries(self):
        from claude_speak.models import PIPER_VOICES
        assert len(PIPER_VOICES) >= 3

    def test_lessac_voice_registered(self):
        from claude_speak.models import PIPER_VOICES
        assert "en_US-lessac-medium" in PIPER_VOICES
        info = PIPER_VOICES["en_US-lessac-medium"]
        assert "lessac" in info.onnx_url
        assert info.config_url.endswith(".onnx.json")

    def test_ryan_voice_registered(self):
        from claude_speak.models import PIPER_VOICES
        assert "en_US-ryan-medium" in PIPER_VOICES
        info = PIPER_VOICES["en_US-ryan-medium"]
        assert "ryan" in info.onnx_url

    def test_alan_voice_registered(self):
        from claude_speak.models import PIPER_VOICES
        assert "en_GB-alan-medium" in PIPER_VOICES
        info = PIPER_VOICES["en_GB-alan-medium"]
        assert "alan" in info.onnx_url
        assert "en_GB" in info.onnx_url

    def test_piper_voice_urls_use_huggingface(self):
        """All Piper voice URLs should point to HuggingFace."""
        from claude_speak.models import PIPER_VOICES
        for name, info in PIPER_VOICES.items():
            assert "huggingface.co" in info.onnx_url, f"{name} onnx_url missing huggingface.co"
            assert "huggingface.co" in info.config_url, f"{name} config_url missing huggingface.co"

    def test_piper_models_dir_exists(self):
        """PIPER_MODELS_DIR should be under ~/.claude-speak/models/piper/."""
        from claude_speak.models import PIPER_MODELS_DIR
        assert str(PIPER_MODELS_DIR).endswith("models/piper")

    def test_list_downloaded_piper_voices_empty_when_no_dir(self):
        """list_downloaded_piper_voices returns empty list when dir doesn't exist."""
        from claude_speak.models import list_downloaded_piper_voices
        result = list_downloaded_piper_voices(dest_dir=Path("/nonexistent/path"))
        assert result == []

    def test_list_downloaded_piper_voices_finds_models(self, tmp_path):
        """list_downloaded_piper_voices finds .onnx files with companion .json."""
        from claude_speak.models import list_downloaded_piper_voices
        # Create model files
        (tmp_path / "en_US-lessac-medium.onnx").touch()
        (tmp_path / "en_US-lessac-medium.onnx.json").touch()
        # This one has no companion json -- should not be listed
        (tmp_path / "en_US-ryan-medium.onnx").touch()

        result = list_downloaded_piper_voices(dest_dir=tmp_path)
        assert result == ["en_US-lessac-medium"]

    def test_download_piper_voice_rejects_unknown(self):
        """download_piper_voice raises ValueError for unknown voice names."""
        from claude_speak.models import download_piper_voice
        with pytest.raises(ValueError, match="Unknown Piper voice"):
            download_piper_voice("xx_XX-nonexistent-voice")
