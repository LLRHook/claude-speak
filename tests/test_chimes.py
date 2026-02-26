"""
Unit tests for src/chimes.py — audio chime generation.

Mocks sounddevice.play to verify samples without playing audio.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from claude_speak.chimes import (
    _generate_tone,
    _play,
    play_ready_chime,
    play_error_chime,
    play_stop_chime,
    play_ack_chime,
)


# ---------------------------------------------------------------------------
# Tests: _generate_tone
# ---------------------------------------------------------------------------

class TestGenerateTone:
    """Tests for the _generate_tone helper."""

    def test_correct_sample_count(self):
        """Generated tone should have the right number of samples for the duration."""
        sr = 24000
        duration = 0.10
        tone = _generate_tone(440.0, duration, sr)
        expected = int(sr * duration)
        assert len(tone) == expected

    def test_sample_type_is_float32(self):
        tone = _generate_tone(440.0, 0.10, 24000)
        assert tone.dtype == np.float32

    def test_fade_envelope_first_sample_near_zero(self):
        """First sample should be near zero due to fade-in envelope."""
        tone = _generate_tone(440.0, 0.10, 24000)
        assert abs(tone[0]) < 0.05

    def test_fade_envelope_last_sample_near_zero(self):
        """Last sample should be near zero due to fade-out envelope."""
        tone = _generate_tone(440.0, 0.10, 24000)
        assert abs(tone[-1]) < 0.05

    def test_peak_amplitude_around_one(self):
        """Middle samples should reach close to amplitude 1.0."""
        tone = _generate_tone(440.0, 0.50, 24000)
        # Peak should be close to 1.0 (sine wave at max)
        assert tone.max() > 0.9

    def test_different_frequencies_produce_different_tones(self):
        tone_low = _generate_tone(261.0, 0.10, 24000)
        tone_high = _generate_tone(880.0, 0.10, 24000)
        # The tones should differ (not identical arrays)
        assert not np.array_equal(tone_low, tone_high)


# ---------------------------------------------------------------------------
# Tests: _play
# ---------------------------------------------------------------------------

class TestPlay:
    """Tests for the _play helper."""

    def test_play_applies_volume(self):
        mock_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            samples = np.ones(100, dtype=np.float32)
            _play(samples, volume=0.5, device=None, sample_rate=24000)
            mock_sd.play.assert_called_once()
            played = mock_sd.play.call_args[0][0]
            np.testing.assert_allclose(played, np.full(100, 0.5, dtype=np.float32), atol=1e-5)

    def test_play_catches_exceptions(self):
        mock_sd = MagicMock()
        mock_sd.play.side_effect = Exception("audio error")
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Should not raise
            _play(np.ones(10, dtype=np.float32), 0.5, None)


# ---------------------------------------------------------------------------
# Tests: chime composition functions
# ---------------------------------------------------------------------------

class TestChimeComposition:
    """Tests for the high-level chime functions.

    We mock _play to capture the samples that would be played,
    avoiding the need for a real sounddevice.
    """

    @patch("claude_speak.chimes._play")
    def test_play_ready_chime_generates_ascending_notes(self, mock_play):
        """Ready chime is C5 -> E5 (ascending frequencies)."""
        play_ready_chime(device=None, volume=0.3)
        mock_play.assert_called_once()
        samples = mock_play.call_args[0][0]
        assert len(samples) > 0
        assert samples.dtype == np.float32

    @patch("claude_speak.chimes._play")
    def test_play_error_chime_generates_descending_notes(self, mock_play):
        """Error chime is E5 -> C5 (descending)."""
        play_error_chime(device=None, volume=0.3)
        mock_play.assert_called_once()
        samples = mock_play.call_args[0][0]
        assert len(samples) > 0

    @patch("claude_speak.chimes._play")
    def test_play_stop_chime_single_tone(self, mock_play):
        """Stop chime is a single C4 tone."""
        play_stop_chime(device=None, volume=0.3)
        mock_play.assert_called_once()
        samples = mock_play.call_args[0][0]
        sr = 24000
        expected_len = int(sr * 0.10)  # 0.10s tone
        assert len(samples) == expected_len

    @patch("claude_speak.chimes._play")
    def test_play_ack_chime_fallback_when_asset_missing(self, mock_play):
        """When ack.wav doesn't exist, should fall back to a tone."""
        # Temporarily rename the asset so it doesn't exist
        ack_path = Path(__file__).resolve().parent.parent / "claude_speak" / "assets" / "ack.wav"
        tmp_path = ack_path.with_suffix(".wav.bak")
        renamed = False
        try:
            if ack_path.exists():
                ack_path.rename(tmp_path)
                renamed = True
            play_ack_chime(device=None, volume=0.3)
            mock_play.assert_called_once()
            # Fallback tone should be ~0.08s at 24kHz
            samples = mock_play.call_args[0][0]
            sr = 24000
            expected_len = int(sr * 0.08)
            assert len(samples) == expected_len
        finally:
            if renamed:
                tmp_path.rename(ack_path)

    @patch("claude_speak.chimes._play")
    def test_ready_chime_samples_longer_than_single_note(self, mock_play):
        """Ready chime has two notes + gap, so it should be longer than a single tone."""
        play_ready_chime(device=None, volume=0.3)
        samples = mock_play.call_args[0][0]
        sr = 24000
        single_tone_len = int(sr * 0.15)
        assert len(samples) > single_tone_len

    @patch("claude_speak.chimes._play")
    def test_ready_chime_passes_correct_device_and_sr(self, mock_play):
        """Ready chime should forward device and sample rate to _play."""
        play_ready_chime(device=7, volume=0.5)
        call_args = mock_play.call_args
        # _play(samples, volume, device, sr)
        assert call_args[0][1] == 0.5  # volume
        assert call_args[0][2] == 7    # device
        assert call_args[0][3] == 24000  # sample_rate

    @patch("claude_speak.chimes._play")
    def test_error_chime_has_two_notes_and_gap(self, mock_play):
        """Error chime should be note1 + gap + note2."""
        play_error_chime(device=None, volume=0.3)
        samples = mock_play.call_args[0][0]
        sr = 24000
        # note1 = 0.08s, gap = 0.03s, note2 = 0.12s = 0.23s total
        expected_len = int(sr * 0.08) + int(sr * 0.03) + int(sr * 0.12)
        assert len(samples) == expected_len
