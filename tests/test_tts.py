"""
Unit tests for src/tts.py — TTSEngine.

All external dependencies (kokoro_onnx, sounddevice) are mocked.
Tests run without audio hardware or model files.
"""

import asyncio
import threading
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from claude_speak.audio_devices import AudioDeviceManager
from claude_speak.config import Config, TTSConfig

# Import the sd module reference so we can patch it on the tts module
import claude_speak.tts as tts_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(voice="af_sarah", speed=1.0, device="auto", volume=1.0):
    """Create a TTSEngine with a given config (does NOT call load())."""
    cfg = Config(tts=TTSConfig(voice=voice, speed=speed, device=device, volume=volume))
    return tts_module.TTSEngine(cfg)


def _fake_kokoro():
    """Return a mock Kokoro instance with common stubs."""
    mock = MagicMock()
    mock.get_voice_style.return_value = np.ones(256, dtype=np.float32)
    mock.get_voices.return_value = ["af_sarah", "bm_fable", "bm_george"]
    return mock


def _mock_sd():
    """Return a MagicMock for sounddevice with common defaults."""
    mock = MagicMock()
    mock.default.device = (0, 1)
    mock.PortAudioError = type("PortAudioError", (Exception,), {})
    return mock


# ---------------------------------------------------------------------------
# Tests: voice resolution
# ---------------------------------------------------------------------------

class TestResolveVoice:
    """Tests for _resolve_voice (single voice vs blend)."""

    def test_single_voice_passthrough(self):
        engine = _make_engine(voice="af_sarah")
        engine.kokoro = _fake_kokoro()
        result = engine._resolve_voice()
        assert result == "af_sarah"

    def test_blend_two_voices_with_explicit_weights(self):
        engine = _make_engine(voice="bm_george:60+bm_fable:40")
        engine.kokoro = _fake_kokoro()
        result = engine._resolve_voice()
        # Should return a numpy array (blend), not a string
        assert isinstance(result, np.ndarray)
        engine.kokoro.get_voice_style.assert_any_call("bm_george")
        engine.kokoro.get_voice_style.assert_any_call("bm_fable")

    def test_blend_equal_shares_when_no_weights(self):
        engine = _make_engine(voice="bm_george+bm_fable")
        engine.kokoro = _fake_kokoro()
        result = engine._resolve_voice()
        assert isinstance(result, np.ndarray)
        # Each voice should contribute 50% -- total approx 1.0 * array
        np.testing.assert_allclose(result, np.ones(256, dtype=np.float32), atol=1e-5)

    def test_blend_mixed_explicit_and_equal(self):
        """One voice has explicit weight, the other gets the remainder."""
        engine = _make_engine(voice="bm_george:70+bm_fable")
        engine.kokoro = _fake_kokoro()
        result = engine._resolve_voice()
        assert isinstance(result, np.ndarray)
        # george=0.7, fable=0.3 -> total array = ones * (0.7 + 0.3) = ones
        np.testing.assert_allclose(result, np.ones(256, dtype=np.float32), atol=1e-5)

    def test_blend_three_voices(self):
        engine = _make_engine(voice="a:50+b:30+c:20")
        engine.kokoro = _fake_kokoro()
        result = engine._resolve_voice()
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Tests: device resolution
# ---------------------------------------------------------------------------

class TestResolveDevice:
    """Tests for device resolution via AudioDeviceManager (delegated from TTSEngine)."""

    def test_auto_device_uses_default(self):
        """device='auto' should use sounddevice default via device manager."""
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.resolve_output.return_value = 5
        mock_dm.get_device_name.return_value = "Built-in Speaker"
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="auto")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("auto")

    def test_device_by_numeric_id(self):
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.resolve_output.return_value = 3
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="3")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("3")

    def test_device_by_name_substring(self):
        """Match device by name substring via device manager."""
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.resolve_output.return_value = 1
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="airpods")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("airpods")

    def test_device_name_not_found_falls_back_to_default(self):
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.resolve_output.return_value = 0
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="nonexistent")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("nonexistent")


# ---------------------------------------------------------------------------
# Tests: stream management
# ---------------------------------------------------------------------------

class TestStreamManagement:
    """Tests for _ensure_stream and stream lifecycle."""

    def test_ensure_stream_creates_new_stream(self):
        mock_sd = _mock_sd()
        mock_stream = MagicMock()
        mock_sd.OutputStream.return_value = mock_stream
        with patch.object(tts_module, "sd", mock_sd):
            engine = _make_engine()
            engine._output_device = 0
            engine._ensure_stream(24000)
            mock_sd.OutputStream.assert_called_once()
            mock_stream.start.assert_called_once()
            assert engine._stream is mock_stream
            assert engine._sample_rate == 24000

    def test_ensure_stream_reuses_when_same_sample_rate(self):
        mock_sd = _mock_sd()
        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd.OutputStream.return_value = mock_stream
        with patch.object(tts_module, "sd", mock_sd):
            engine = _make_engine()
            engine._output_device = 0
            engine._ensure_stream(24000)
            # Second call -- same sample rate, stream still active
            engine._ensure_stream(24000)
            # OutputStream should only be called once (reused)
            assert mock_sd.OutputStream.call_count == 1

    def test_ensure_stream_recreates_on_sample_rate_change(self):
        mock_sd = _mock_sd()
        stream1 = MagicMock(active=True)
        stream2 = MagicMock(active=True)
        mock_sd.OutputStream.side_effect = [stream1, stream2]
        with patch.object(tts_module, "sd", mock_sd):
            engine = _make_engine()
            engine._output_device = 0
            engine._ensure_stream(24000)
            assert engine._sample_rate == 24000
            engine._ensure_stream(16000)
            assert engine._sample_rate == 16000
            stream1.close.assert_called_once()
            assert mock_sd.OutputStream.call_count == 2


# ---------------------------------------------------------------------------
# Tests: stop mechanism
# ---------------------------------------------------------------------------

class TestStop:
    """Tests for stop() and its interaction with _write_samples."""

    def test_stop_sets_event_and_aborts_stream(self):
        engine = _make_engine()
        mock_stream = MagicMock()
        engine._stream = mock_stream
        engine.stop()
        assert engine._stopped.is_set()
        mock_stream.abort.assert_called_once()
        mock_stream.close.assert_called_once()
        assert engine._stream is None

    def test_stop_when_no_stream(self):
        """stop() should not raise even when _stream is None."""
        engine = _make_engine()
        engine.stop()
        assert engine._stopped.is_set()

    def test_write_samples_respects_stopped_event(self):
        """_write_samples should bail immediately if _stopped is set."""
        engine = _make_engine()
        engine._stopped.set()
        engine._stream = MagicMock()
        samples = np.zeros(8192, dtype=np.float32)
        engine._write_samples(samples)
        # stream.write should never be called
        engine._stream.write.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: volume scaling
# ---------------------------------------------------------------------------

class TestVolume:
    """Tests for volume scaling in _write_samples."""

    def test_volume_scaling_below_one(self):
        engine = _make_engine(volume=0.5)
        mock_stream = MagicMock()
        engine._stream = mock_stream
        engine._stopped.clear()
        samples = np.ones(100, dtype=np.float32)
        engine._write_samples(samples)
        # The stream.write should have been called with scaled samples
        assert mock_stream.write.called
        written = mock_stream.write.call_args[0][0]
        # Samples should be ~0.5 (volume scaled), reshaped to (-1, 1)
        assert written.max() <= 0.6  # allow small tolerance

    def test_volume_at_one_no_scaling(self):
        engine = _make_engine(volume=1.0)
        mock_stream = MagicMock()
        engine._stream = mock_stream
        engine._stopped.clear()
        samples = np.ones(100, dtype=np.float32)
        engine._write_samples(samples)
        assert mock_stream.write.called
        written = mock_stream.write.call_args[0][0]
        # Volume=1.0 means no scaling -- samples should still be 1.0
        np.testing.assert_allclose(written.flatten(), np.ones(100, dtype=np.float32), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: generate_audio / speak
# ---------------------------------------------------------------------------

class TestGenerateAudio:
    """Tests for generate_audio and speak."""

    def test_generate_audio_returns_segments(self):
        engine = _make_engine()
        engine.kokoro = _fake_kokoro()
        engine._voice_style = "af_sarah"

        # Mock create_stream to yield two chunks
        async def fake_stream(text, voice, speed, lang):
            yield (np.zeros(1000, dtype=np.float32), 24000)
            yield (np.zeros(500, dtype=np.float32), 24000)

        engine.kokoro.create_stream = fake_stream
        result = asyncio.run(engine.generate_audio("Hello world"))
        assert len(result) == 2
        assert result[0][1] == 24000  # sample rate
        assert len(result[0][0]) == 1000

    def test_generate_audio_loads_model_if_needed(self):
        engine = _make_engine()
        # kokoro is None -- should trigger load()
        with patch.object(engine, "load") as mock_load:
            # After load, set up kokoro mock
            def do_load():
                engine.kokoro = _fake_kokoro()
                engine._voice_style = "af_sarah"

                async def fake_stream(text, voice, speed, lang):
                    yield (np.zeros(100, dtype=np.float32), 24000)

                engine.kokoro.create_stream = fake_stream

            mock_load.side_effect = do_load
            result = asyncio.run(engine.generate_audio("test"))
            mock_load.assert_called_once()
            assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: play_audio
# ---------------------------------------------------------------------------

class TestPlayAudio:
    """Tests for play_audio."""

    def test_play_audio_calls_ensure_stream_and_write(self):
        engine = _make_engine()
        engine._output_device = 0

        mock_sd = _mock_sd()
        mock_stream = MagicMock(active=True)
        mock_sd.OutputStream.return_value = mock_stream

        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.maybe_resolve_output.return_value = 0
        mock_dm.get_default_output.return_value = 0

        with patch.object(tts_module, "sd", mock_sd), \
             patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            segments = [
                (np.zeros(100, dtype=np.float32), 24000),
                (np.zeros(200, dtype=np.float32), 24000),
            ]
            engine.play_audio(segments)
            # Stream.write should have been called (samples were written)
            assert mock_stream.write.called


# ---------------------------------------------------------------------------
# Tests: list_voices
# ---------------------------------------------------------------------------

class TestListVoices:
    """Tests for list_voices."""

    def test_list_voices_returns_list(self):
        engine = _make_engine()
        engine.kokoro = _fake_kokoro()
        voices = engine.list_voices()
        assert voices == ["af_sarah", "bm_fable", "bm_george"]

    def test_list_voices_loads_model_if_needed(self):
        engine = _make_engine()
        with patch.object(engine, "load") as mock_load:
            def do_load():
                engine.kokoro = _fake_kokoro()
            mock_load.side_effect = do_load
            voices = engine.list_voices()
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _maybe_resolve_device (delegates to AudioDeviceManager)
# ---------------------------------------------------------------------------

class TestMaybeResolveDevice:
    """Tests for _maybe_resolve_device delegation to AudioDeviceManager."""

    def test_delegates_to_device_manager(self):
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.maybe_resolve_output.return_value = 7
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="auto")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("auto")
            assert engine._output_device == 7

    def test_passes_device_preference(self):
        mock_dm = MagicMock(spec=AudioDeviceManager)
        mock_dm.maybe_resolve_output.return_value = 3
        with patch("claude_speak.tts.get_device_manager", return_value=mock_dm):
            engine = _make_engine(device="airpods")
            engine._maybe_resolve_device()
            mock_dm.maybe_resolve_output.assert_called_once_with("airpods")
