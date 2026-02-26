"""
Unit tests for train/benchmark.py -- wake word accuracy benchmarking suite.

Mocks openwakeword to avoid model dependencies. Tests cover metric computation,
table formatting, JSON output, CLI argument parsing, threshold sweeps, and
edge cases like empty directories.
"""

import json
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root so we can import from train/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from benchmark import (
    DEFAULT_THRESHOLDS,
    BenchmarkResults,
    ThresholdMetrics,
    WakeWordBenchmark,
    compute_metrics,
    format_table,
    load_wav,
    parse_args,
    scan_wav_files,
    score_clip,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(filepath: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a mono 16-bit WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype(np.int16).tobytes())


def _synthetic_samples(duration_s: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate synthetic int16 audio samples."""
    n = int(duration_s * sample_rate)
    return np.random.randint(-5000, 5000, size=n, dtype=np.int16)


def _make_mock_model(score: float = 0.5) -> MagicMock:
    """Create a mock openwakeword model that returns a fixed score."""
    model = MagicMock()
    model.predict.return_value = {"test_model": score}
    model.reset.return_value = None
    return model


# ---------------------------------------------------------------------------
# Tests: ThresholdMetrics
# ---------------------------------------------------------------------------


class TestThresholdMetrics:
    """Tests for ThresholdMetrics dataclass properties."""

    def test_recall_perfect(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=10, false_negatives=0)
        assert m.recall == 1.0

    def test_recall_zero(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=0, false_negatives=10)
        assert m.recall == 0.0

    def test_recall_partial(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=7, false_negatives=3)
        assert m.recall == pytest.approx(0.7)

    def test_recall_empty(self):
        """Recall is 0.0 when there are no positive samples."""
        m = ThresholdMetrics(threshold=0.5)
        assert m.recall == 0.0

    def test_fpr_zero(self):
        m = ThresholdMetrics(threshold=0.5, false_positives=0, true_negatives=20)
        assert m.false_positive_rate == 0.0

    def test_fpr_perfect_bad(self):
        m = ThresholdMetrics(threshold=0.5, false_positives=20, true_negatives=0)
        assert m.false_positive_rate == 1.0

    def test_fpr_partial(self):
        m = ThresholdMetrics(threshold=0.5, false_positives=2, true_negatives=18)
        assert m.false_positive_rate == pytest.approx(0.1)

    def test_fpr_empty(self):
        """FPR is 0.0 when there are no negative samples."""
        m = ThresholdMetrics(threshold=0.5)
        assert m.false_positive_rate == 0.0

    def test_precision_perfect(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=10, false_positives=0)
        assert m.precision == 1.0

    def test_precision_zero(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=0, false_positives=10)
        assert m.precision == 0.0

    def test_precision_partial(self):
        m = ThresholdMetrics(threshold=0.5, true_positives=8, false_positives=2)
        assert m.precision == pytest.approx(0.8)

    def test_precision_empty(self):
        """Precision is 0.0 when no predictions are positive."""
        m = ThresholdMetrics(threshold=0.5)
        assert m.precision == 0.0

    def test_f1_perfect(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=10, false_positives=0,
            true_negatives=10, false_negatives=0,
        )
        assert m.f1_score == 1.0

    def test_f1_zero(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=0, false_positives=0,
            true_negatives=10, false_negatives=10,
        )
        assert m.f1_score == 0.0

    def test_f1_balanced(self):
        """F1 for precision=0.8, recall=0.8 should be 0.8."""
        m = ThresholdMetrics(
            threshold=0.5, true_positives=8, false_positives=2,
            true_negatives=18, false_negatives=2,
        )
        assert m.f1_score == pytest.approx(0.8)

    def test_accuracy_perfect(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=10, false_positives=0,
            true_negatives=10, false_negatives=0,
        )
        assert m.accuracy == 1.0

    def test_accuracy_half(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=5, false_positives=5,
            true_negatives=5, false_negatives=5,
        )
        assert m.accuracy == pytest.approx(0.5)

    def test_accuracy_empty(self):
        m = ThresholdMetrics(threshold=0.5)
        assert m.accuracy == 0.0

    def test_to_dict_keys(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=8, false_positives=2,
            true_negatives=18, false_negatives=2,
        )
        d = m.to_dict()
        expected_keys = {
            "threshold", "true_positives", "false_positives",
            "true_negatives", "false_negatives",
            "recall", "false_positive_rate", "precision",
            "f1_score", "accuracy",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_rounded(self):
        m = ThresholdMetrics(
            threshold=0.5, true_positives=1, false_positives=1,
            true_negatives=1, false_negatives=1,
        )
        d = m.to_dict()
        # All computed metrics should be 4-decimal rounded
        assert d["recall"] == 0.5
        assert d["precision"] == 0.5
        assert d["accuracy"] == 0.5


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for compute_metrics — threshold sweep over scores."""

    def test_all_positive_above_threshold(self):
        positive_scores = [0.9, 0.8, 0.7]
        negative_scores = [0.1, 0.2, 0.3]
        results = compute_metrics(positive_scores, negative_scores, [0.5])
        m = results[0]
        assert m.true_positives == 3
        assert m.false_negatives == 0
        assert m.false_positives == 0
        assert m.true_negatives == 3

    def test_all_below_threshold(self):
        positive_scores = [0.1, 0.2]
        negative_scores = [0.1, 0.2]
        results = compute_metrics(positive_scores, negative_scores, [0.5])
        m = results[0]
        assert m.true_positives == 0
        assert m.false_negatives == 2
        assert m.false_positives == 0
        assert m.true_negatives == 2

    def test_mixed_scores(self):
        positive_scores = [0.9, 0.4, 0.6]
        negative_scores = [0.1, 0.5, 0.3]
        results = compute_metrics(positive_scores, negative_scores, [0.5])
        m = results[0]
        assert m.true_positives == 2  # 0.9, 0.6
        assert m.false_negatives == 1  # 0.4
        assert m.false_positives == 1  # 0.5
        assert m.true_negatives == 2  # 0.1, 0.3

    def test_multiple_thresholds(self):
        positive_scores = [0.9, 0.5, 0.3]
        negative_scores = [0.8, 0.2, 0.1]
        thresholds = [0.3, 0.5, 0.8]
        results = compute_metrics(positive_scores, negative_scores, thresholds)

        assert len(results) == 3

        # Threshold 0.3: pos>=0.3 -> all 3 TP; neg>=0.3 -> 0.8 FP
        assert results[0].true_positives == 3
        assert results[0].false_positives == 1

        # Threshold 0.5: pos>=0.5 -> 0.9, 0.5 TP; neg>=0.5 -> 0.8 FP
        assert results[1].true_positives == 2
        assert results[1].false_positives == 1

        # Threshold 0.8: pos>=0.8 -> 0.9 TP; neg>=0.8 -> 0.8 FP
        assert results[2].true_positives == 1
        assert results[2].false_positives == 1

    def test_empty_positive(self):
        results = compute_metrics([], [0.5, 0.3], [0.5])
        m = results[0]
        assert m.true_positives == 0
        assert m.false_negatives == 0
        assert m.false_positives == 1
        assert m.true_negatives == 1

    def test_empty_negative(self):
        results = compute_metrics([0.9, 0.3], [], [0.5])
        m = results[0]
        assert m.true_positives == 1
        assert m.false_negatives == 1
        assert m.false_positives == 0
        assert m.true_negatives == 0

    def test_empty_both(self):
        results = compute_metrics([], [], [0.5])
        m = results[0]
        assert m.true_positives == 0
        assert m.false_negatives == 0
        assert m.false_positives == 0
        assert m.true_negatives == 0
        assert m.accuracy == 0.0

    def test_threshold_at_exact_score(self):
        """Score exactly equal to threshold is a positive detection."""
        results = compute_metrics([0.5], [0.5], [0.5])
        m = results[0]
        assert m.true_positives == 1
        assert m.false_positives == 1

    def test_decreasing_recall_with_higher_threshold(self):
        """Higher thresholds should produce equal or lower recall."""
        positive_scores = [0.9, 0.7, 0.5, 0.3, 0.1]
        negative_scores = [0.0] * 5
        thresholds = [0.2, 0.4, 0.6, 0.8]
        results = compute_metrics(positive_scores, negative_scores, thresholds)
        recalls = [m.recall for m in results]
        for i in range(len(recalls) - 1):
            assert recalls[i] >= recalls[i + 1]


# ---------------------------------------------------------------------------
# Tests: format_table
# ---------------------------------------------------------------------------


class TestFormatTable:
    """Tests for format_table — console output formatting."""

    def test_header_present(self):
        metrics = [ThresholdMetrics(threshold=0.5)]
        table = format_table(metrics)
        assert "Threshold" in table
        assert "Recall" in table
        assert "FPR" in table
        assert "Precision" in table
        assert "F1" in table
        assert "Accuracy" in table

    def test_separator_line(self):
        metrics = [ThresholdMetrics(threshold=0.5)]
        table = format_table(metrics)
        lines = table.split("\n")
        # Second line should be a separator
        assert set(lines[1].strip()) == {"-"}

    def test_data_row_count(self):
        metrics = [ThresholdMetrics(threshold=t) for t in [0.3, 0.5, 0.7]]
        table = format_table(metrics)
        lines = table.split("\n")
        # header + separator + 3 data rows
        assert len(lines) == 5

    def test_threshold_values_in_output(self):
        metrics = [
            ThresholdMetrics(threshold=0.3, true_positives=10, false_negatives=0,
                             false_positives=0, true_negatives=10),
            ThresholdMetrics(threshold=0.7, true_positives=5, false_negatives=5,
                             false_positives=0, true_negatives=10),
        ]
        table = format_table(metrics)
        assert "0.30" in table
        assert "0.70" in table

    def test_empty_metrics_list(self):
        table = format_table([])
        lines = table.split("\n")
        # header + separator only
        assert len(lines) == 2

    def test_metric_values_present(self):
        metrics = [ThresholdMetrics(
            threshold=0.5, true_positives=8, false_positives=2,
            true_negatives=18, false_negatives=2,
        )]
        table = format_table(metrics)
        # Check recall = 8/10 = 0.8
        assert "0.8000" in table


# ---------------------------------------------------------------------------
# Tests: BenchmarkResults serialization
# ---------------------------------------------------------------------------


class TestBenchmarkResults:
    """Tests for BenchmarkResults.to_dict — JSON output format."""

    def test_to_dict_keys(self):
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=10,
            num_negative=20,
        )
        d = results.to_dict()
        expected_keys = {
            "model_path", "positive_dir", "negative_dir",
            "num_positive", "num_negative",
            "thresholds", "positive_scores", "negative_scores",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_thresholds_are_dicts(self):
        metrics = ThresholdMetrics(
            threshold=0.5, true_positives=5, false_positives=1,
            true_negatives=9, false_negatives=5,
        )
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=10,
            num_negative=10,
            thresholds=[metrics],
        )
        d = results.to_dict()
        assert len(d["thresholds"]) == 1
        assert isinstance(d["thresholds"][0], dict)
        assert d["thresholds"][0]["threshold"] == 0.5

    def test_to_dict_scores_rounded(self):
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=2,
            num_negative=1,
            positive_scores=[0.123456789, 0.987654321],
            negative_scores=[0.555555555],
        )
        d = results.to_dict()
        assert d["positive_scores"] == [0.1235, 0.9877]
        assert d["negative_scores"] == [0.5556]

    def test_to_dict_json_serializable(self):
        metrics = ThresholdMetrics(
            threshold=0.5, true_positives=5, false_positives=1,
            true_negatives=9, false_negatives=5,
        )
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=10,
            num_negative=10,
            thresholds=[metrics],
            positive_scores=[0.9, 0.8],
            negative_scores=[0.1],
        )
        # Should not raise
        json_str = json.dumps(results.to_dict())
        parsed = json.loads(json_str)
        assert parsed["num_positive"] == 10


# ---------------------------------------------------------------------------
# Tests: load_wav
# ---------------------------------------------------------------------------


class TestLoadWav:
    """Tests for load_wav — reading WAV files."""

    def test_loads_16khz_mono(self, tmp_path):
        samples = _synthetic_samples(1.0)
        filepath = tmp_path / "test.wav"
        _make_wav(filepath, samples)

        loaded = load_wav(filepath)
        assert loaded.dtype == np.int16
        assert len(loaded) == len(samples)
        np.testing.assert_array_equal(loaded, samples)

    def test_resamples_non_16khz(self, tmp_path):
        """WAV files at other sample rates are resampled to 16kHz."""
        original_sr = 44100
        duration = 1.0
        n_samples = int(duration * original_sr)
        samples = np.random.randint(-5000, 5000, size=n_samples, dtype=np.int16)
        filepath = tmp_path / "test_44k.wav"
        _make_wav(filepath, samples, sample_rate=original_sr)

        loaded = load_wav(filepath)
        expected_length = int(n_samples * 16000 / original_sr)
        assert len(loaded) == expected_length

    def test_raises_on_invalid_file(self, tmp_path):
        filepath = tmp_path / "bad.wav"
        filepath.write_bytes(b"not a wav file")
        with pytest.raises(ValueError, match="Could not read WAV"):
            load_wav(filepath)


# ---------------------------------------------------------------------------
# Tests: scan_wav_files
# ---------------------------------------------------------------------------


class TestScanWavFiles:
    """Tests for scan_wav_files — directory scanning."""

    def test_finds_wav_files(self, tmp_path):
        for i in range(3):
            _make_wav(tmp_path / f"clip_{i}.wav", _synthetic_samples(0.5))
        files = scan_wav_files(tmp_path)
        assert len(files) == 3

    def test_ignores_non_wav_files(self, tmp_path):
        _make_wav(tmp_path / "clip.wav", _synthetic_samples(0.5))
        (tmp_path / "notes.txt").write_text("not audio")
        (tmp_path / "data.json").write_text("{}")
        files = scan_wav_files(tmp_path)
        assert len(files) == 1

    def test_empty_directory(self, tmp_path):
        files = scan_wav_files(tmp_path)
        assert files == []

    def test_nonexistent_directory(self, tmp_path):
        files = scan_wav_files(tmp_path / "does_not_exist")
        assert files == []

    def test_files_are_sorted(self, tmp_path):
        for name in ["c.wav", "a.wav", "b.wav"]:
            _make_wav(tmp_path / name, _synthetic_samples(0.5))
        files = scan_wav_files(tmp_path)
        names = [f.name for f in files]
        assert names == ["a.wav", "b.wav", "c.wav"]


# ---------------------------------------------------------------------------
# Tests: score_clip (with mock model)
# ---------------------------------------------------------------------------


class TestScoreClip:
    """Tests for score_clip — running audio through a model."""

    def test_returns_max_score_across_chunks(self):
        """score_clip should return the peak prediction across all chunks."""
        model = MagicMock()
        # Simulate varying scores across chunks
        model.predict.side_effect = [
            {"test": 0.1},
            {"test": 0.8},
            {"test": 0.3},
        ]
        model.reset.return_value = None

        # 3 chunks * 1280 samples = 3840 samples
        samples = np.zeros(3 * 1280, dtype=np.int16)
        score = score_clip(model, samples)
        assert score == pytest.approx(0.8)

    def test_calls_predict_per_chunk(self):
        model = _make_mock_model(0.5)
        samples = np.zeros(5 * 1280, dtype=np.int16)
        score_clip(model, samples)
        assert model.predict.call_count == 5

    def test_resets_model_after_clip(self):
        model = _make_mock_model(0.5)
        samples = np.zeros(2 * 1280, dtype=np.int16)
        score_clip(model, samples)
        model.reset.assert_called_once()

    def test_handles_short_clip(self):
        """Clips shorter than one chunk produce score 0.0."""
        model = _make_mock_model(0.5)
        samples = np.zeros(100, dtype=np.int16)  # less than 1280
        score = score_clip(model, samples)
        assert score == 0.0
        model.predict.assert_not_called()

    def test_multiple_model_names(self):
        """score_clip returns the max across all model names."""
        model = MagicMock()
        model.predict.return_value = {"model_a": 0.3, "model_b": 0.7}
        model.reset.return_value = None

        samples = np.zeros(1280, dtype=np.int16)
        score = score_clip(model, samples)
        assert score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Tests: WakeWordBenchmark.run (integration with mocks)
# ---------------------------------------------------------------------------


class TestWakeWordBenchmarkRun:
    """Integration tests for WakeWordBenchmark.run with mocked model."""

    def _setup_dirs(self, tmp_path, n_pos=3, n_neg=3):
        """Create temporary directories with synthetic WAV files."""
        pos_dir = tmp_path / "positive"
        neg_dir = tmp_path / "negative"
        pos_dir.mkdir()
        neg_dir.mkdir()

        for i in range(n_pos):
            _make_wav(pos_dir / f"pos_{i}.wav", _synthetic_samples(1.0))
        for i in range(n_neg):
            _make_wav(neg_dir / f"neg_{i}.wav", _synthetic_samples(1.0))

        return pos_dir, neg_dir

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_returns_results(self, mock_load, tmp_path):
        pos_dir, neg_dir = self._setup_dirs(tmp_path)
        mock_model = _make_mock_model(0.5)
        mock_load.return_value = mock_model

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
        )

        assert isinstance(results, BenchmarkResults)
        assert results.num_positive == 3
        assert results.num_negative == 3
        assert len(results.positive_scores) == 3
        assert len(results.negative_scores) == 3

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_uses_default_thresholds(self, mock_load, tmp_path):
        pos_dir, neg_dir = self._setup_dirs(tmp_path)
        mock_load.return_value = _make_mock_model(0.5)

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
        )

        assert len(results.thresholds) == len(DEFAULT_THRESHOLDS)
        assert [t.threshold for t in results.thresholds] == DEFAULT_THRESHOLDS

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_custom_thresholds(self, mock_load, tmp_path):
        pos_dir, neg_dir = self._setup_dirs(tmp_path)
        mock_load.return_value = _make_mock_model(0.5)

        benchmark = WakeWordBenchmark()
        custom_thresholds = [0.1, 0.9]
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
            thresholds=custom_thresholds,
        )

        assert len(results.thresholds) == 2
        assert results.thresholds[0].threshold == 0.1
        assert results.thresholds[1].threshold == 0.9

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_saves_json(self, mock_load, tmp_path):
        pos_dir, neg_dir = self._setup_dirs(tmp_path)
        mock_load.return_value = _make_mock_model(0.5)
        output_json = str(tmp_path / "results.json")

        benchmark = WakeWordBenchmark()
        benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
            output_json=output_json,
        )

        assert Path(output_json).exists()
        with open(output_json) as f:
            data = json.load(f)
        assert "thresholds" in data
        assert "positive_scores" in data
        assert data["num_positive"] == 3

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_empty_directories(self, mock_load, tmp_path):
        pos_dir = tmp_path / "empty_pos"
        neg_dir = tmp_path / "empty_neg"
        pos_dir.mkdir()
        neg_dir.mkdir()
        mock_load.return_value = _make_mock_model(0.5)

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
        )

        assert results.num_positive == 0
        assert results.num_negative == 0
        assert results.positive_scores == []
        assert results.negative_scores == []

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_run_with_synthetic_predictions(self, mock_load, tmp_path):
        """Mock model returns high scores for positive clips, low for negative."""
        pos_dir, neg_dir = self._setup_dirs(tmp_path, n_pos=5, n_neg=5)

        # Model returns 0.9 for first 5 calls (positive), 0.1 for next 5 (negative)
        # Each clip is ~1s at 16kHz = 12.5 chunks, so ~12 predict calls per clip
        model = MagicMock()
        model.reset.return_value = None

        # We need to track call count across all predict calls
        call_count = {"n": 0, "clip": 0}
        files_scored = {"n": 0}
        high_score = 0.9
        low_score = 0.1

        def predict_side_effect(chunk):
            # Each reset marks a new clip
            return {"test": high_score if files_scored["n"] < 5 else low_score}

        original_reset = model.reset

        def reset_side_effect():
            files_scored["n"] += 1

        model.predict.side_effect = predict_side_effect
        model.reset.side_effect = reset_side_effect
        mock_load.return_value = model

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
            thresholds=[0.5],
        )

        m = results.thresholds[0]
        assert m.true_positives == 5
        assert m.false_positives == 0
        assert m.true_negatives == 5
        assert m.false_negatives == 0
        assert m.recall == 1.0
        assert m.false_positive_rate == 0.0
        assert m.accuracy == 1.0

    @patch("benchmark.WakeWordBenchmark._load_model")
    def test_threshold_sweep_metrics(self, mock_load, tmp_path):
        """Test that threshold sweep correctly partitions detections."""
        pos_dir, neg_dir = self._setup_dirs(tmp_path, n_pos=2, n_neg=2)

        model = MagicMock()
        model.reset.return_value = None
        clip_number = {"n": 0}

        # Clip 0 (pos): 0.8, Clip 1 (pos): 0.4, Clip 2 (neg): 0.6, Clip 3 (neg): 0.2
        scores_per_clip = [0.8, 0.4, 0.6, 0.2]

        def predict_side_effect(chunk):
            return {"test": scores_per_clip[clip_number["n"]]}

        def reset_side_effect():
            clip_number["n"] += 1

        model.predict.side_effect = predict_side_effect
        model.reset.side_effect = reset_side_effect
        mock_load.return_value = model

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="fake.onnx",
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
            thresholds=[0.3, 0.5, 0.7],
        )

        # At threshold 0.3: pos scores [0.8, 0.4] -> both >= 0.3 -> TP=2
        #                    neg scores [0.6, 0.2] -> 0.6 >= 0.3 -> FP=1, TN=1
        assert results.thresholds[0].true_positives == 2
        assert results.thresholds[0].false_positives == 1

        # At threshold 0.5: pos scores [0.8, 0.4] -> 0.8 >= 0.5 -> TP=1, FN=1
        #                    neg scores [0.6, 0.2] -> 0.6 >= 0.5 -> FP=1, TN=1
        assert results.thresholds[1].true_positives == 1
        assert results.thresholds[1].false_negatives == 1
        assert results.thresholds[1].false_positives == 1

        # At threshold 0.7: pos scores [0.8, 0.4] -> 0.8 >= 0.7 -> TP=1, FN=1
        #                    neg scores [0.6, 0.2] -> none >= 0.7 -> FP=0, TN=2
        assert results.thresholds[2].true_positives == 1
        assert results.thresholds[2].false_positives == 0


# ---------------------------------------------------------------------------
# Tests: CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self):
        args = parse_args([
            "--model", "model.onnx",
            "--positive", "pos/",
            "--negative", "neg/",
        ])
        assert args.model == "model.onnx"
        assert args.positive == "pos/"
        assert args.negative == "neg/"

    def test_missing_model_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--positive", "pos/", "--negative", "neg/"])

    def test_missing_positive_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "m.onnx", "--negative", "neg/"])

    def test_missing_negative_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--model", "m.onnx", "--positive", "pos/"])

    def test_output_arg(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
            "--output", "results.json",
        ])
        assert args.output == "results.json"

    def test_output_default_none(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
        ])
        assert args.output is None

    def test_roc_arg(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
            "--roc", "roc.png",
        ])
        assert args.roc == "roc.png"

    def test_roc_default_none(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
        ])
        assert args.roc is None

    def test_custom_thresholds(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
            "--thresholds", "0.1", "0.2", "0.9",
        ])
        assert args.thresholds == [0.1, 0.2, 0.9]

    def test_thresholds_default_none(self):
        args = parse_args([
            "--model", "m.onnx", "--positive", "p/", "--negative", "n/",
        ])
        assert args.thresholds is None

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# Tests: JSON output file
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Tests for JSON file output correctness."""

    def test_json_roundtrip(self, tmp_path):
        """Results serialized to JSON can be read back correctly."""
        metrics = ThresholdMetrics(
            threshold=0.5, true_positives=8, false_positives=2,
            true_negatives=18, false_negatives=2,
        )
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=10,
            num_negative=20,
            thresholds=[metrics],
            positive_scores=[0.9, 0.8, 0.7],
            negative_scores=[0.1, 0.2],
        )

        output_path = tmp_path / "results.json"
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["model_path"] == "model.onnx"
        assert loaded["num_positive"] == 10
        assert loaded["num_negative"] == 20
        assert len(loaded["thresholds"]) == 1
        assert loaded["thresholds"][0]["recall"] == 0.8
        assert loaded["thresholds"][0]["accuracy"] == pytest.approx(0.8667, abs=0.001)

    def test_json_multiple_thresholds(self, tmp_path):
        """JSON output contains all thresholds."""
        thresholds_data = [
            ThresholdMetrics(threshold=t, true_positives=5, true_negatives=5)
            for t in DEFAULT_THRESHOLDS
        ]
        results = BenchmarkResults(
            model_path="model.onnx",
            positive_dir="pos/",
            negative_dir="neg/",
            num_positive=5,
            num_negative=5,
            thresholds=thresholds_data,
        )

        output_path = tmp_path / "results.json"
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f)

        with open(output_path) as f:
            loaded = json.load(f)

        assert len(loaded["thresholds"]) == len(DEFAULT_THRESHOLDS)
        for i, t in enumerate(DEFAULT_THRESHOLDS):
            assert loaded["thresholds"][i]["threshold"] == t


# ---------------------------------------------------------------------------
# Tests: ROC curve generation
# ---------------------------------------------------------------------------


class TestROCCurve:
    """Tests for ROC curve generation (optional matplotlib)."""

    def test_roc_without_matplotlib(self):
        """generate_roc_curve returns False when matplotlib is not available."""
        from benchmark import generate_roc_curve

        with patch.dict("sys.modules", {"matplotlib": None}):
            result = generate_roc_curve([0.9], [0.1], Path("/tmp/roc.png"))
            assert result is False

    def test_roc_with_matplotlib(self, tmp_path):
        """generate_roc_curve saves an image when matplotlib is available."""
        from benchmark import generate_roc_curve

        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")

        output_path = tmp_path / "roc.png"
        result = generate_roc_curve(
            [0.9, 0.8, 0.7, 0.6, 0.3],
            [0.1, 0.2, 0.4, 0.5, 0.8],
            output_path,
        )
        assert result is True
        assert output_path.exists()
        assert output_path.stat().st_size > 0
