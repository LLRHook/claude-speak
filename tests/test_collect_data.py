"""
Unit tests for train/collect_data.py -- real-world training data collection.

Mocks sounddevice to avoid audio hardware. Tests cover filename generation,
audio format, label validation, batch configuration, and recording logic.
"""

import re
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

import sys

# Add project root so we can import from train/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from collect_data import (
    CHANNELS,
    MAX_DURATION,
    MIN_DURATION,
    SAMPLE_RATE,
    VALID_LABELS,
    BatchConfig,
    generate_filename,
    get_output_dir,
    record_clip,
    sanitize_word,
    save_wav,
    validate_clip,
)


# ---------------------------------------------------------------------------
# Tests: sanitize_word
# ---------------------------------------------------------------------------


class TestSanitizeWord:
    """Tests for sanitize_word — cleaning wake word phrases for filenames."""

    def test_lowercase(self):
        assert sanitize_word("STOP") == "stop"

    def test_spaces_replaced(self):
        assert sanitize_word("hey jarvis") == "hey_jarvis"

    def test_special_characters_replaced(self):
        assert sanitize_word("stop!") == "stop"

    def test_multiple_spaces_collapsed(self):
        assert sanitize_word("hey   jarvis") == "hey_jarvis"

    def test_leading_trailing_stripped(self):
        assert sanitize_word("  stop  ") == "stop"

    def test_mixed_special_chars(self):
        assert sanitize_word("hey! jarvis?") == "hey_jarvis"

    def test_empty_string_returns_unknown(self):
        assert sanitize_word("") == "unknown"

    def test_only_special_chars_returns_unknown(self):
        assert sanitize_word("!!!") == "unknown"

    def test_numbers_preserved(self):
        assert sanitize_word("word123") == "word123"

    def test_underscores_preserved(self):
        assert sanitize_word("hey_jarvis") == "hey_jarvis"


# ---------------------------------------------------------------------------
# Tests: generate_filename
# ---------------------------------------------------------------------------


class TestGenerateFilename:
    """Tests for generate_filename — structured filename creation."""

    def test_positive_label_format(self):
        name = generate_filename("positive", "stop", 0, "20260226_143000")
        assert name == "positive_stop_20260226_143000_0000.wav"

    def test_negative_label_format(self):
        name = generate_filename("negative", "stop", 5, "20260226_143000")
        assert name == "negative_stop_20260226_143000_0005.wav"

    def test_multi_word_phrase(self):
        name = generate_filename("positive", "hey jarvis", 12, "20260226_150000")
        assert name == "positive_hey_jarvis_20260226_150000_0012.wav"

    def test_index_zero_padded(self):
        name = generate_filename("positive", "stop", 999, "20260226_143000")
        assert name == "positive_stop_20260226_143000_0999.wav"

    def test_wav_extension(self):
        name = generate_filename("positive", "stop", 0, "20260226_143000")
        assert name.endswith(".wav")

    def test_invalid_label_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid label"):
            generate_filename("invalid", "stop", 0)

    def test_empty_label_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid label"):
            generate_filename("", "stop", 0)

    def test_auto_generated_timestamp(self):
        """When timestamp is None, one is generated automatically."""
        name = generate_filename("positive", "stop", 0)
        # Should match pattern: positive_stop_YYYYMMDD_HHMMSS_0000.wav
        pattern = r"^positive_stop_\d{8}_\d{6}_0000\.wav$"
        assert re.match(pattern, name), f"Filename '{name}' does not match expected pattern"

    def test_word_is_sanitized(self):
        name = generate_filename("positive", "HEY JARVIS!", 0, "20260226_143000")
        assert "hey_jarvis" in name
        assert "HEY" not in name
        assert "!" not in name

    def test_filename_components_order(self):
        """Verify the order: label, word, timestamp, index."""
        name = generate_filename("negative", "hey jarvis", 3, "20260226_143000")
        parts = name.replace(".wav", "").split("_")
        # negative_hey_jarvis_20260226_143000_0003
        assert parts[0] == "negative"
        assert "hey" in parts
        assert "jarvis" in parts
        assert parts[-1] == "0003"


# ---------------------------------------------------------------------------
# Tests: get_output_dir
# ---------------------------------------------------------------------------


class TestGetOutputDir:
    """Tests for get_output_dir — mapping labels to directories."""

    def test_positive_dir(self):
        d = get_output_dir("positive")
        assert d.name == "positive"
        assert d.parent.name == "collected"

    def test_negative_dir(self):
        d = get_output_dir("negative")
        assert d.name == "negative"
        assert d.parent.name == "collected"

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid label"):
            get_output_dir("unknown")


# ---------------------------------------------------------------------------
# Tests: validate_clip
# ---------------------------------------------------------------------------


class TestValidateClip:
    """Tests for validate_clip — checking recorded clip quality."""

    def test_valid_clip(self):
        """A clip with audio content that meets minimum duration passes."""
        samples = np.random.randint(-5000, 5000, size=int(SAMPLE_RATE * 1.0), dtype=np.int16)
        assert validate_clip(samples) is True

    def test_too_short_clip(self):
        """Clips shorter than MIN_DURATION are rejected."""
        short = np.random.randint(-5000, 5000, size=int(SAMPLE_RATE * 0.1), dtype=np.int16)
        assert validate_clip(short) is False

    def test_silence_clip(self):
        """All-zero (silent) clips are rejected."""
        silence = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16)
        assert validate_clip(silence) is False

    def test_near_silence_clip(self):
        """Very quiet clips (RMS < threshold) are rejected."""
        quiet = np.ones(int(SAMPLE_RATE * 1.0), dtype=np.int16) * 5
        assert validate_clip(quiet) is False

    def test_exactly_min_duration(self):
        """A clip exactly at MIN_DURATION with content should pass."""
        n_samples = int(SAMPLE_RATE * MIN_DURATION)
        samples = np.random.randint(-5000, 5000, size=n_samples, dtype=np.int16)
        assert validate_clip(samples) is True

    def test_three_second_clip(self):
        """A full 3-second clip with content passes."""
        n_samples = int(SAMPLE_RATE * MAX_DURATION)
        samples = np.random.randint(-5000, 5000, size=n_samples, dtype=np.int16)
        assert validate_clip(samples) is True


# ---------------------------------------------------------------------------
# Tests: save_wav
# ---------------------------------------------------------------------------


class TestSaveWav:
    """Tests for save_wav — writing 16kHz mono WAV files."""

    def test_creates_wav_file(self, tmp_path):
        filepath = tmp_path / "test.wav"
        samples = np.random.randint(-5000, 5000, size=SAMPLE_RATE, dtype=np.int16)
        save_wav(filepath, samples)
        assert filepath.exists()

    def test_wav_format_16khz_mono(self, tmp_path):
        """Written WAV must be 16kHz, mono, 16-bit."""
        filepath = tmp_path / "test.wav"
        samples = np.random.randint(-5000, 5000, size=SAMPLE_RATE, dtype=np.int16)
        save_wav(filepath, samples)

        with wave.open(str(filepath), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2  # 16-bit

    def test_wav_frame_count(self, tmp_path):
        """Number of frames in the WAV file matches input samples."""
        filepath = tmp_path / "test.wav"
        n_samples = 8000  # 0.5 seconds
        samples = np.random.randint(-5000, 5000, size=n_samples, dtype=np.int16)
        save_wav(filepath, samples)

        with wave.open(str(filepath), "rb") as wf:
            assert wf.getnframes() == n_samples

    def test_wav_data_roundtrip(self, tmp_path):
        """Audio data can be read back and matches the original."""
        filepath = tmp_path / "test.wav"
        samples = np.array([100, -200, 300, -400, 500], dtype=np.int16)
        save_wav(filepath, samples)

        with wave.open(str(filepath), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            read_back = np.frombuffer(raw, dtype=np.int16)

        np.testing.assert_array_equal(read_back, samples)

    def test_creates_parent_directories(self, tmp_path):
        """save_wav creates intermediate directories as needed."""
        filepath = tmp_path / "a" / "b" / "c" / "test.wav"
        samples = np.random.randint(-5000, 5000, size=SAMPLE_RATE, dtype=np.int16)
        save_wav(filepath, samples)
        assert filepath.exists()


# ---------------------------------------------------------------------------
# Tests: record_clip (mocked sounddevice)
# ---------------------------------------------------------------------------


class TestRecordClip:
    """Tests for record_clip — recording via sounddevice (mocked)."""

    @patch("collect_data.sd")
    def test_records_at_16khz_mono(self, mock_sd):
        """record_clip should call sd.rec with correct parameters."""
        fake_audio = np.zeros((int(SAMPLE_RATE * MAX_DURATION), 1), dtype=np.int16)
        mock_sd.rec.return_value = fake_audio

        record_clip()

        mock_sd.rec.assert_called_once()
        call_kwargs = mock_sd.rec.call_args
        assert call_kwargs[1]["samplerate"] == SAMPLE_RATE
        assert call_kwargs[1]["channels"] == 1
        assert call_kwargs[1]["dtype"] == "int16"

    @patch("collect_data.sd")
    def test_returns_flattened_array(self, mock_sd):
        """record_clip should return a 1D array (flattened from Nx1)."""
        n_frames = int(SAMPLE_RATE * MAX_DURATION)
        fake_audio = np.random.randint(-5000, 5000, size=(n_frames, 1), dtype=np.int16)
        mock_sd.rec.return_value = fake_audio

        result = record_clip()

        assert result.ndim == 1
        assert len(result) == n_frames

    @patch("collect_data.sd")
    def test_waits_for_recording(self, mock_sd):
        """record_clip should call sd.wait() after starting recording."""
        fake_audio = np.zeros((int(SAMPLE_RATE * MAX_DURATION), 1), dtype=np.int16)
        mock_sd.rec.return_value = fake_audio

        record_clip()

        mock_sd.wait.assert_called_once()

    @patch("collect_data.sd")
    def test_custom_duration(self, mock_sd):
        """record_clip with custom duration records the right number of frames."""
        duration = 1.5
        n_frames = int(SAMPLE_RATE * duration)
        fake_audio = np.zeros((n_frames, 1), dtype=np.int16)
        mock_sd.rec.return_value = fake_audio

        record_clip(duration=duration)

        call_args = mock_sd.rec.call_args
        assert call_args[0][0] == n_frames

    @patch("collect_data.sd", None)
    def test_raises_without_sounddevice(self):
        """record_clip raises RuntimeError when sounddevice is not available."""
        with pytest.raises(RuntimeError, match="sounddevice is not installed"):
            record_clip()


# ---------------------------------------------------------------------------
# Tests: BatchConfig
# ---------------------------------------------------------------------------


class TestBatchConfig:
    """Tests for BatchConfig — batch recording session parameters."""

    def test_default_counts(self):
        config = BatchConfig(word="stop")
        assert config.positive_count == 10
        assert config.negative_count == 10

    def test_custom_counts(self):
        config = BatchConfig(word="stop", positive_count=5, negative_count=20)
        assert config.positive_count == 5
        assert config.negative_count == 20

    def test_total_count(self):
        config = BatchConfig(word="stop", positive_count=5, negative_count=15)
        assert config.total_count == 20

    def test_word_stored_stripped(self):
        config = BatchConfig(word="  stop  ")
        assert config.word == "stop"

    def test_negative_positive_count_raises(self):
        with pytest.raises(ValueError, match="positive_count must be >= 0"):
            BatchConfig(word="stop", positive_count=-1)

    def test_negative_negative_count_raises(self):
        with pytest.raises(ValueError, match="negative_count must be >= 0"):
            BatchConfig(word="stop", negative_count=-1)

    def test_empty_word_raises(self):
        with pytest.raises(ValueError, match="word must not be empty"):
            BatchConfig(word="")

    def test_whitespace_only_word_raises(self):
        with pytest.raises(ValueError, match="word must not be empty"):
            BatchConfig(word="   ")

    def test_zero_counts_allowed(self):
        config = BatchConfig(word="stop", positive_count=0, negative_count=0)
        assert config.total_count == 0


# ---------------------------------------------------------------------------
# Tests: label validation
# ---------------------------------------------------------------------------


class TestLabelValidation:
    """Tests that labels are validated across the module."""

    def test_valid_labels_constant(self):
        assert VALID_LABELS == ("positive", "negative")

    def test_generate_filename_rejects_bad_label(self):
        with pytest.raises(ValueError):
            generate_filename("neutral", "stop", 0)

    def test_get_output_dir_rejects_bad_label(self):
        with pytest.raises(ValueError):
            get_output_dir("neutral")

    def test_generate_filename_accepts_positive(self):
        name = generate_filename("positive", "stop", 0, "20260226_143000")
        assert name.startswith("positive_")

    def test_generate_filename_accepts_negative(self):
        name = generate_filename("negative", "stop", 0, "20260226_143000")
        assert name.startswith("negative_")


# ---------------------------------------------------------------------------
# Tests: audio format constants
# ---------------------------------------------------------------------------


class TestAudioConstants:
    """Tests that audio format constants are openwakeword-compatible."""

    def test_sample_rate_16khz(self):
        assert SAMPLE_RATE == 16000

    def test_mono_channel(self):
        assert CHANNELS == 1

    def test_max_duration_3_seconds(self):
        assert MAX_DURATION == 3.0

    def test_min_duration_positive(self):
        assert MIN_DURATION > 0
        assert MIN_DURATION < MAX_DURATION


# ---------------------------------------------------------------------------
# Tests: parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_no_args_defaults(self):
        from collect_data import parse_args
        args = parse_args([])
        assert args.word is None
        assert args.label is None
        assert args.count is None
        assert args.batch is None

    def test_word_arg(self):
        from collect_data import parse_args
        args = parse_args(["--word", "stop"])
        assert args.word == "stop"

    def test_batch_arg(self):
        from collect_data import parse_args
        args = parse_args(["--word", "stop", "--batch", "20"])
        assert args.batch == 20

    def test_label_positive(self):
        from collect_data import parse_args
        args = parse_args(["--word", "stop", "--label", "positive", "--count", "5"])
        assert args.label == "positive"
        assert args.count == 5

    def test_label_negative(self):
        from collect_data import parse_args
        args = parse_args(["--word", "stop", "--label", "negative"])
        assert args.label == "negative"

    def test_invalid_label_exits(self):
        from collect_data import parse_args
        with pytest.raises(SystemExit):
            parse_args(["--word", "stop", "--label", "invalid"])

    def test_negative_count_arg(self):
        from collect_data import parse_args
        args = parse_args(["--word", "stop", "--batch", "10", "--negative-count", "30"])
        assert args.negative_count == 30


# ---------------------------------------------------------------------------
# Tests: end-to-end save and verify
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration-style tests combining multiple functions."""

    def test_generate_and_save_clip(self, tmp_path):
        """Generate a filename, create synthetic audio, save, and verify."""
        filename = generate_filename("positive", "stop", 0, "20260226_143000")
        filepath = tmp_path / filename

        samples = np.random.randint(-5000, 5000, size=int(SAMPLE_RATE * 1.5), dtype=np.int16)
        save_wav(filepath, samples)

        assert filepath.exists()
        assert filepath.name == "positive_stop_20260226_143000_0000.wav"

        with wave.open(str(filepath), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2

    def test_filename_uniqueness_across_indices(self):
        """Different indices produce different filenames."""
        names = set()
        for i in range(100):
            name = generate_filename("positive", "stop", i, "20260226_143000")
            names.add(name)
        assert len(names) == 100

    def test_validate_then_save(self, tmp_path):
        """Only valid clips should be saved."""
        good = np.random.randint(-5000, 5000, size=int(SAMPLE_RATE * 1.0), dtype=np.int16)
        bad_short = np.random.randint(-5000, 5000, size=int(SAMPLE_RATE * 0.1), dtype=np.int16)
        bad_silent = np.zeros(int(SAMPLE_RATE * 1.0), dtype=np.int16)

        assert validate_clip(good) is True
        assert validate_clip(bad_short) is False
        assert validate_clip(bad_silent) is False

        # Only save the good clip
        filepath = tmp_path / "good.wav"
        save_wav(filepath, good)
        assert filepath.exists()
