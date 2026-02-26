"""
Unit tests for claude_speak/vad.py — Silero VAD voice activity detection.

All tests mock onnxruntime.InferenceSession so they run without the actual
ONNX model or onnxruntime installed.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_session(speech_prob: float = 0.0):
    """Create a mock ONNX InferenceSession that returns a fixed speech probability.

    Returns outputs in the same format as Silero VAD: [prob, h_out, c_out].
    """
    mock_session = MagicMock()
    h_out = np.zeros((2, 1, 64), dtype=np.float32)
    c_out = np.zeros((2, 1, 64), dtype=np.float32)
    mock_session.run.return_value = [
        np.array([[speech_prob]], dtype=np.float32),
        h_out,
        c_out,
    ]
    return mock_session


# ---------------------------------------------------------------------------
# Tests: SileroVAD initialization
# ---------------------------------------------------------------------------


class TestSileroVADInit:
    """Tests for SileroVAD.__init__."""

    @patch("claude_speak.vad.ort.InferenceSession")
    @patch("claude_speak.vad.MODELS_DIR", Path("/tmp/fake-models"))
    def test_init_sets_threshold_and_sample_rate(self, mock_ort_session):
        """SileroVAD stores the threshold and sample_rate on the instance."""
        mock_ort_session.return_value = _make_mock_session()

        # Ensure the model file "exists"
        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.6, sample_rate=16000)

        assert vad.threshold == 0.6
        assert vad.sample_rate == 16000

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_init_creates_zero_state_tensors(self, mock_ort_session):
        """Internal h and c state tensors should be zero-initialized."""
        mock_ort_session.return_value = _make_mock_session()

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD()

        assert vad._h.shape == (2, 1, 64)
        assert vad._c.shape == (2, 1, 64)
        np.testing.assert_array_equal(vad._h, np.zeros((2, 1, 64), dtype=np.float32))
        np.testing.assert_array_equal(vad._c, np.zeros((2, 1, 64), dtype=np.float32))

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_init_loads_onnx_session(self, mock_ort_session):
        """Constructor should create an ONNX InferenceSession."""
        mock_ort_session.return_value = _make_mock_session()

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD()

        mock_ort_session.assert_called_once()
        assert vad._session is mock_ort_session.return_value


# ---------------------------------------------------------------------------
# Tests: is_speech
# ---------------------------------------------------------------------------


class TestIsSpeech:
    """Tests for SileroVAD.is_speech."""

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_returns_bool(self, mock_ort_session):
        """is_speech should always return a bool."""
        mock_session = _make_mock_session(speech_prob=0.8)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        chunk = np.zeros(512, dtype=np.float32)
        result = vad.is_speech(chunk)

        assert isinstance(result, bool)

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_silence_returns_false(self, mock_ort_session):
        """All-zero audio with low probability should return False."""
        mock_session = _make_mock_session(speech_prob=0.1)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        silence = np.zeros(512, dtype=np.float32)
        assert vad.is_speech(silence) is False

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_speech_returns_true(self, mock_ort_session):
        """When model returns high probability, is_speech should return True."""
        mock_session = _make_mock_session(speech_prob=0.9)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        chunk = np.random.randn(512).astype(np.float32)
        assert vad.is_speech(chunk) is True

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_calls_onnx_model_with_correct_inputs(self, mock_ort_session):
        """is_speech should pass audio, sr, h, c to the ONNX session."""
        mock_session = _make_mock_session(speech_prob=0.3)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5, sample_rate=16000)

        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        ort_inputs = call_args[1] if call_args[1] else call_args[0][1]

        assert "input" in ort_inputs
        assert "sr" in ort_inputs
        assert "h" in ort_inputs
        assert "c" in ort_inputs

        # Check shapes
        assert ort_inputs["input"].shape == (1, 512)
        assert ort_inputs["sr"].dtype == np.int64
        assert ort_inputs["h"].shape == (2, 1, 64)
        assert ort_inputs["c"].shape == (2, 1, 64)

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_updates_state_after_call(self, mock_ort_session):
        """After is_speech, the h and c state should be updated from model outputs."""
        mock_session = MagicMock()
        h_out = np.ones((2, 1, 64), dtype=np.float32) * 0.42
        c_out = np.ones((2, 1, 64), dtype=np.float32) * 0.84
        mock_session.run.return_value = [
            np.array([[0.3]], dtype=np.float32),
            h_out,
            c_out,
        ]
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk)

        np.testing.assert_array_equal(vad._h, h_out)
        np.testing.assert_array_equal(vad._c, c_out)

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_handles_2d_input(self, mock_ort_session):
        """is_speech should handle 2D input (e.g., from sounddevice) by flattening."""
        mock_session = _make_mock_session(speech_prob=0.8)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        # sounddevice often returns shape (N, 1)
        chunk_2d = np.zeros((512, 1), dtype=np.float32)
        result = vad.is_speech(chunk_2d)
        assert isinstance(result, bool)

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_threshold_boundary_equal(self, mock_ort_session):
        """Probability exactly equal to threshold should return False (not >)."""
        mock_session = _make_mock_session(speech_prob=0.5)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        chunk = np.zeros(512, dtype=np.float32)
        assert vad.is_speech(chunk) is False


# ---------------------------------------------------------------------------
# Tests: reset
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for SileroVAD.reset."""

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_reset_clears_state(self, mock_ort_session):
        """reset() should zero out the h and c state tensors."""
        mock_session = MagicMock()
        # Return non-zero state from a call
        h_dirty = np.ones((2, 1, 64), dtype=np.float32) * 99.0
        c_dirty = np.ones((2, 1, 64), dtype=np.float32) * 99.0
        mock_session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            h_dirty,
            c_dirty,
        ]
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        # Run once to dirty the state
        chunk = np.zeros(512, dtype=np.float32)
        vad.is_speech(chunk)
        assert np.any(vad._h != 0)  # state should be dirty

        # Reset
        vad.reset()

        np.testing.assert_array_equal(vad._h, np.zeros((2, 1, 64), dtype=np.float32))
        np.testing.assert_array_equal(vad._c, np.zeros((2, 1, 64), dtype=np.float32))

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_reset_preserves_session(self, mock_ort_session):
        """reset() should not reload the ONNX session."""
        mock_session = _make_mock_session()
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD()

        session_before = vad._session
        vad.reset()

        assert vad._session is session_before
        # InferenceSession should only be created once (during __init__)
        mock_ort_session.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: get_vad singleton
# ---------------------------------------------------------------------------


class TestGetVad:
    """Tests for the get_vad singleton accessor."""

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_returns_singleton(self, mock_ort_session):
        """get_vad should return the same instance on multiple calls."""
        mock_ort_session.return_value = _make_mock_session()

        with patch.object(Path, "exists", return_value=True):
            import claude_speak.vad as vad_module
            vad_module._vad_instance = None  # ensure clean state

            instance1 = vad_module.get_vad(threshold=0.5)
            instance2 = vad_module.get_vad(threshold=0.7)  # threshold ignored on second call

        assert instance1 is instance2

        # Cleanup
        vad_module._vad_instance = None

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_returns_silero_vad_instance(self, mock_ort_session):
        """get_vad should return a SileroVAD instance."""
        mock_ort_session.return_value = _make_mock_session()

        with patch.object(Path, "exists", return_value=True):
            import claude_speak.vad as vad_module
            from claude_speak.vad import SileroVAD
            vad_module._vad_instance = None  # ensure clean state

            instance = vad_module.get_vad()

        assert isinstance(instance, SileroVAD)

        # Cleanup
        vad_module._vad_instance = None


# ---------------------------------------------------------------------------
# Tests: auto-download
# ---------------------------------------------------------------------------


class TestAutoDownload:
    """Tests for auto-downloading the model when missing."""

    @patch("claude_speak.vad.ort.InferenceSession")
    @patch("claude_speak.vad.download_model")
    def test_downloads_when_model_missing(self, mock_download, mock_ort_session):
        """When the model file does not exist, _load_model should trigger download."""
        mock_ort_session.return_value = _make_mock_session()
        mock_download.return_value = Path("/tmp/fake-models/silero_vad.onnx")

        with patch.object(Path, "exists", return_value=False):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD()

        mock_download.assert_called_once()
        # Verify it was called with the correct model info
        call_args = mock_download.call_args
        model_info = call_args[0][0]
        assert model_info.name == "silero_vad.onnx"

    @patch("claude_speak.vad.ort.InferenceSession")
    @patch("claude_speak.vad.download_model")
    def test_skips_download_when_model_exists(self, mock_download, mock_ort_session):
        """When the model file exists, _load_model should not download."""
        mock_ort_session.return_value = _make_mock_session()

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD()

        mock_download.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: synthetic audio scenarios
# ---------------------------------------------------------------------------


class TestSyntheticAudio:
    """Tests with synthetic audio signals to verify model invocation."""

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_silence_all_zeros(self, mock_ort_session):
        """All-zero audio (silence) should be passed to the model correctly."""
        mock_session = _make_mock_session(speech_prob=0.02)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        silence = np.zeros(512, dtype=np.float32)
        result = vad.is_speech(silence)

        assert result is False
        mock_session.run.assert_called_once()

    @patch("claude_speak.vad.ort.InferenceSession")
    def test_synthetic_speech_like_signal(self, mock_ort_session):
        """A non-zero audio signal should be passed to the model for inference."""
        mock_session = _make_mock_session(speech_prob=0.95)
        mock_ort_session.return_value = mock_session

        with patch.object(Path, "exists", return_value=True):
            from claude_speak.vad import SileroVAD
            vad = SileroVAD(threshold=0.5)

        # Simulate speech-like signal: sine wave at 440 Hz
        t = np.arange(512, dtype=np.float32) / 16000
        speech_like = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = vad.is_speech(speech_like)

        assert result is True
        mock_session.run.assert_called_once()

        # Verify the audio was passed as input
        call_args = mock_session.run.call_args
        ort_inputs = call_args[1] if call_args[1] else call_args[0][1]
        passed_audio = ort_inputs["input"]
        assert passed_audio.shape == (1, 512)
        # The audio content should match our sine wave (reshaped)
        np.testing.assert_array_almost_equal(passed_audio[0], speech_like)
