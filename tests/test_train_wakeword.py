"""
Unit tests for claude_speak/train_wakeword.py — custom wake word training pipeline.

All audio operations use synthetic numpy data to avoid requiring a microphone.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.train_wakeword import (
    DEFAULT_SAMPLES,
    FEATURE_WINDOW_SEC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    _evaluate,
    _mel_filter_bank,
    _train_classifier,
    add_noise,
    augment_clip,
    compute_feature_dim,
    export_to_onnx,
    extract_mel_features,
    features_from_clip,
    pitch_shift,
    predict_onnx,
    speed_variation,
    train_wakeword,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clip(duration_sec: float = 1.5, freq_hz: float = 440.0) -> np.ndarray:
    """Create a synthetic int16 audio clip (sine wave)."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.arange(n) / SAMPLE_RATE
    signal = (np.sin(2 * np.pi * freq_hz * t) * 10000).astype(np.int16)
    return signal


def _make_random_clip(duration_sec: float = 1.5) -> np.ndarray:
    """Create a random int16 audio clip."""
    n = int(SAMPLE_RATE * duration_sec)
    return np.random.randint(-5000, 5000, size=n, dtype=np.int16)


# ---------------------------------------------------------------------------
# Tests: Augmentation — pitch_shift
# ---------------------------------------------------------------------------

class TestPitchShift:
    """Tests for the pitch_shift augmentation function."""

    def test_returns_same_length(self):
        clip = _make_clip()
        shifted = pitch_shift(clip, 2)
        assert len(shifted) == len(clip)

    def test_returns_same_dtype(self):
        clip = _make_clip()
        shifted = pitch_shift(clip, -1)
        assert shifted.dtype == clip.dtype

    def test_positive_shift_changes_content(self):
        clip = _make_clip()
        shifted = pitch_shift(clip, 3)
        # Should not be identical to original
        assert not np.array_equal(clip, shifted)

    def test_negative_shift_changes_content(self):
        clip = _make_clip()
        shifted = pitch_shift(clip, -3)
        assert not np.array_equal(clip, shifted)

    def test_zero_shift_preserves_signal(self):
        clip = _make_clip()
        shifted = pitch_shift(clip, 0)
        np.testing.assert_array_equal(clip, shifted)

    def test_preserves_int16_range(self):
        clip = np.array([32000, -32000, 0, 16000], dtype=np.int16)
        shifted = pitch_shift(clip, 5)
        assert shifted.min() >= -32768
        assert shifted.max() <= 32767


# ---------------------------------------------------------------------------
# Tests: Augmentation — add_noise
# ---------------------------------------------------------------------------

class TestAddNoise:
    """Tests for the add_noise augmentation function."""

    def test_returns_same_length(self):
        clip = _make_clip()
        noisy = add_noise(clip, 0.01)
        assert len(noisy) == len(clip)

    def test_returns_same_dtype(self):
        clip = _make_clip()
        noisy = add_noise(clip, 0.01)
        assert noisy.dtype == clip.dtype

    def test_noise_changes_content(self):
        clip = _make_clip()
        noisy = add_noise(clip, 0.01)
        assert not np.array_equal(clip, noisy)

    def test_zero_noise_level(self):
        clip = _make_clip()
        noisy = add_noise(clip, 0.0)
        # With zero noise, the output should be very close to the original
        np.testing.assert_array_almost_equal(
            noisy.astype(np.float64), clip.astype(np.float64), decimal=0
        )

    def test_output_clipped_to_int16_range(self):
        clip = np.full(1000, 32000, dtype=np.int16)
        noisy = add_noise(clip, 0.5)  # very high noise
        assert noisy.max() <= 32767
        assert noisy.min() >= -32768


# ---------------------------------------------------------------------------
# Tests: Augmentation — speed_variation
# ---------------------------------------------------------------------------

class TestSpeedVariation:
    """Tests for the speed_variation augmentation function."""

    def test_returns_same_length(self):
        clip = _make_clip()
        varied = speed_variation(clip, 1.1)
        assert len(varied) == len(clip)

    def test_returns_same_dtype(self):
        clip = _make_clip()
        varied = speed_variation(clip, 0.9)
        assert varied.dtype == clip.dtype

    def test_faster_changes_content(self):
        clip = _make_clip()
        varied = speed_variation(clip, 1.2)
        assert not np.array_equal(clip, varied)

    def test_slower_changes_content(self):
        clip = _make_clip()
        varied = speed_variation(clip, 0.8)
        assert not np.array_equal(clip, varied)

    def test_factor_one_preserves_signal(self):
        clip = _make_clip()
        varied = speed_variation(clip, 1.0)
        np.testing.assert_array_equal(clip, varied)

    def test_very_fast_does_not_crash(self):
        clip = _make_clip(duration_sec=0.1)
        varied = speed_variation(clip, 5.0)
        assert len(varied) == len(clip)


# ---------------------------------------------------------------------------
# Tests: Augmentation — augment_clip
# ---------------------------------------------------------------------------

class TestAugmentClip:
    """Tests for the augment_clip convenience function."""

    def test_returns_list_of_arrays(self):
        clip = _make_clip()
        variants = augment_clip(clip)
        assert isinstance(variants, list)
        assert all(isinstance(v, np.ndarray) for v in variants)

    def test_expected_number_of_variants(self):
        clip = _make_clip()
        variants = augment_clip(clip)
        # 4 pitch shifts + 3 noise levels + 2 speed variations = 9
        assert len(variants) == 9

    def test_all_variants_same_length(self):
        clip = _make_clip()
        variants = augment_clip(clip)
        for v in variants:
            assert len(v) == len(clip)

    def test_all_variants_same_dtype(self):
        clip = _make_clip()
        variants = augment_clip(clip)
        for v in variants:
            assert v.dtype == clip.dtype


# ---------------------------------------------------------------------------
# Tests: Mel filter bank
# ---------------------------------------------------------------------------

class TestMelFilterBank:
    """Tests for the mel filter bank construction."""

    def test_shape(self):
        bank = _mel_filter_bank()
        n_freqs = N_FFT // 2 + 1
        assert bank.shape == (N_MELS, n_freqs)

    def test_non_negative(self):
        bank = _mel_filter_bank()
        assert np.all(bank >= 0)

    def test_each_filter_peaks(self):
        """Each mel filter should have at least one nonzero value."""
        bank = _mel_filter_bank()
        for i in range(N_MELS):
            assert bank[i].max() > 0, f"Mel filter {i} is all zeros"


# ---------------------------------------------------------------------------
# Tests: Feature extraction
# ---------------------------------------------------------------------------

class TestExtractMelFeatures:
    """Tests for extract_mel_features — mel spectrogram computation."""

    def test_output_shape(self):
        clip = _make_clip(duration_sec=1.0)
        mel = extract_mel_features(clip)
        n_frames = 1 + (len(clip) - N_FFT) // HOP_LENGTH
        assert mel.shape == (N_MELS, n_frames)

    def test_output_dtype(self):
        clip = _make_clip()
        mel = extract_mel_features(clip)
        assert mel.dtype == np.float32

    def test_different_audio_produces_different_features(self):
        clip_a = _make_clip(freq_hz=440)
        clip_b = _make_clip(freq_hz=880)
        mel_a = extract_mel_features(clip_a)
        mel_b = extract_mel_features(clip_b)
        assert not np.allclose(mel_a, mel_b)

    def test_short_audio_padded(self):
        """Audio shorter than one FFT window should still produce output."""
        short_clip = np.array([100, -100, 200, -200], dtype=np.int16)
        mel = extract_mel_features(short_clip)
        assert mel.shape[0] == N_MELS
        assert mel.shape[1] >= 1


class TestFeaturesFromClip:
    """Tests for features_from_clip — fixed-size feature vectors."""

    def test_output_is_1d(self):
        clip = _make_clip()
        features = features_from_clip(clip)
        assert features.ndim == 1

    def test_output_shape_matches_compute_feature_dim(self):
        clip = _make_clip()
        features = features_from_clip(clip)
        expected_dim = compute_feature_dim()
        assert len(features) == expected_dim

    def test_output_dtype_float32(self):
        clip = _make_clip()
        features = features_from_clip(clip)
        assert features.dtype == np.float32

    def test_short_clip_same_dim(self):
        """Short clips should be padded to produce the same feature dimension."""
        short = _make_clip(duration_sec=0.5)
        long = _make_clip(duration_sec=2.0)
        feat_short = features_from_clip(short)
        feat_long = features_from_clip(long)
        assert len(feat_short) == len(feat_long)

    def test_different_clips_different_features(self):
        clip_a = _make_clip(freq_hz=300)
        clip_b = _make_clip(freq_hz=1000)
        feat_a = features_from_clip(clip_a)
        feat_b = features_from_clip(clip_b)
        assert not np.allclose(feat_a, feat_b)


class TestComputeFeatureDim:
    """Tests for compute_feature_dim."""

    def test_positive_integer(self):
        dim = compute_feature_dim()
        assert dim > 0
        assert isinstance(dim, int)

    def test_consistent_with_features_from_clip(self):
        clip = _make_clip()
        features = features_from_clip(clip)
        assert len(features) == compute_feature_dim()


# ---------------------------------------------------------------------------
# Tests: Training
# ---------------------------------------------------------------------------

class TestTrainClassifier:
    """Tests for the numpy logistic regression classifier."""

    def _make_training_data(self, n_per_class: int = 50, n_features: int = 20):
        """Create linearly separable synthetic training data."""
        rng = np.random.RandomState(42)
        X_pos = rng.randn(n_per_class, n_features) + 2.0
        X_neg = rng.randn(n_per_class, n_features) - 2.0
        X = np.vstack([X_pos, X_neg]).astype(np.float32)
        y = np.array([1.0] * n_per_class + [0.0] * n_per_class)
        return X, y

    def test_returns_weights_and_bias(self):
        X, y = self._make_training_data()
        w, b = _train_classifier(X, y)
        assert w.shape == (X.shape[1],)
        assert b.shape == (1,)

    def test_weights_dtype(self):
        X, y = self._make_training_data()
        w, b = _train_classifier(X, y)
        assert w.dtype == np.float32
        assert b.dtype == np.float32

    def test_high_accuracy_on_separable_data(self):
        X, y = self._make_training_data()
        w, b = _train_classifier(X, y)
        accuracy = _evaluate(X, y, w, b)
        assert accuracy >= 0.9, f"Expected high accuracy, got {accuracy:.2%}"

    def test_training_with_single_feature(self):
        rng = np.random.RandomState(42)
        X = np.concatenate([rng.randn(30, 1) + 3, rng.randn(30, 1) - 3]).astype(np.float32)
        y = np.array([1.0] * 30 + [0.0] * 30)
        w, b = _train_classifier(X, y)
        accuracy = _evaluate(X, y, w, b)
        assert accuracy >= 0.85


class TestEvaluate:
    """Tests for the _evaluate function."""

    def test_perfect_accuracy(self):
        X = np.array([[10.0], [-10.0]], dtype=np.float32)
        y = np.array([1.0, 0.0])
        w = np.array([1.0], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        assert _evaluate(X, y, w, b) == 1.0

    def test_zero_accuracy(self):
        X = np.array([[10.0], [-10.0]], dtype=np.float32)
        y = np.array([0.0, 1.0])  # flipped labels
        w = np.array([1.0], dtype=np.float32)
        b = np.array([0.0], dtype=np.float32)
        assert _evaluate(X, y, w, b) == 0.0


# ---------------------------------------------------------------------------
# Tests: ONNX export
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not importlib.util.find_spec("onnx"),
    reason="onnx package not installed",
)
class TestOnnxExport:
    """Tests for ONNX model export and inference."""

    def test_export_creates_file(self, tmp_path):
        n_features = 20
        w = np.random.randn(n_features).astype(np.float32)
        b = np.array([0.5], dtype=np.float32)
        out = tmp_path / "test_model.onnx"
        result = export_to_onnx(w, b, out, feature_dim=n_features)
        assert result.exists()
        assert result.suffix == ".onnx"

    def test_export_valid_onnx(self, tmp_path):
        """Exported model should pass ONNX validation."""
        import onnx

        n_features = 10
        w = np.random.randn(n_features).astype(np.float32)
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "valid_model.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)

        model = onnx.load(str(out))
        onnx.checker.check_model(model)

    def test_inference_produces_probability(self, tmp_path):
        n_features = 15
        w = np.random.randn(n_features).astype(np.float32)
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "inference_model.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)

        features = np.random.randn(1, n_features).astype(np.float32)
        prob = predict_onnx(out, features)
        assert prob.shape == (1, 1)
        assert 0.0 <= prob[0, 0] <= 1.0

    def test_inference_1d_input(self, tmp_path):
        """predict_onnx should accept 1-D input and reshape automatically."""
        n_features = 10
        w = np.random.randn(n_features).astype(np.float32)
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "model_1d.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)

        features = np.random.randn(n_features).astype(np.float32)
        prob = predict_onnx(out, features)
        assert prob.shape == (1, 1)

    def test_positive_weights_give_high_probability(self, tmp_path):
        """With large positive weights and positive input, probability should be high."""
        n_features = 5
        w = np.ones(n_features, dtype=np.float32) * 5.0
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "pos_model.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)

        features = np.ones((1, n_features), dtype=np.float32) * 2.0
        prob = predict_onnx(out, features)
        assert prob[0, 0] > 0.9

    def test_negative_weights_give_low_probability(self, tmp_path):
        """With large negative weights and positive input, probability should be low."""
        n_features = 5
        w = np.ones(n_features, dtype=np.float32) * -5.0
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "neg_model.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)

        features = np.ones((1, n_features), dtype=np.float32) * 2.0
        prob = predict_onnx(out, features)
        assert prob[0, 0] < 0.1

    def test_creates_parent_directories(self, tmp_path):
        n_features = 5
        w = np.random.randn(n_features).astype(np.float32)
        b = np.array([0.0], dtype=np.float32)
        out = tmp_path / "a" / "b" / "c" / "model.onnx"
        export_to_onnx(w, b, out, feature_dim=n_features)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: End-to-end training with synthetic data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not importlib.util.find_spec("onnx"),
    reason="onnx package not installed",
)
class TestTrainWithSyntheticData:
    """Test the full pipeline from features to ONNX using synthetic audio."""

    def test_features_to_onnx_pipeline(self, tmp_path):
        """Synthetic features -> train -> export -> verify inference."""
        rng = np.random.RandomState(123)
        feature_dim = compute_feature_dim()

        # Generate synthetic training data
        n_pos = 30
        n_neg = 30
        X_pos = rng.randn(n_pos, feature_dim).astype(np.float32) + 1.0
        X_neg = rng.randn(n_neg, feature_dim).astype(np.float32) - 1.0
        X = np.vstack([X_pos, X_neg])
        y = np.array([1.0] * n_pos + [0.0] * n_neg)

        # Shuffle
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        # Train
        w, b = _train_classifier(X, y)
        accuracy = _evaluate(X, y, w, b)
        assert accuracy >= 0.8

        # Export
        model_path = tmp_path / "test_wakeword.onnx"
        export_to_onnx(w, b, model_path, feature_dim=feature_dim)
        assert model_path.exists()

        # Verify inference
        prob = predict_onnx(model_path, X_pos[:1])
        assert prob.shape == (1, 1)
        assert 0.0 <= prob[0, 0] <= 1.0

    def test_audio_to_onnx_pipeline(self, tmp_path):
        """Synthetic audio clips -> features -> train -> export -> inference."""
        rng = np.random.RandomState(456)

        # Create synthetic positive clips (higher frequency)
        pos_clips = [_make_clip(freq_hz=800 + rng.randint(-50, 50)) for _ in range(10)]
        # Create synthetic negative clips (lower frequency)
        neg_clips = [_make_clip(freq_hz=200 + rng.randint(-50, 50)) for _ in range(10)]

        # Extract features
        feature_dim = compute_feature_dim()
        X_pos = np.array([features_from_clip(c) for c in pos_clips])
        X_neg = np.array([features_from_clip(c) for c in neg_clips])
        X = np.vstack([X_pos, X_neg])
        y = np.array([1.0] * len(pos_clips) + [0.0] * len(neg_clips))

        # Shuffle
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        # Train
        w, b = _train_classifier(X, y)

        # Export
        model_path = tmp_path / "audio_wakeword.onnx"
        export_to_onnx(w, b, model_path, feature_dim=feature_dim)
        assert model_path.exists()

        # Inference on a new positive clip
        new_pos = _make_clip(freq_hz=800)
        feat = features_from_clip(new_pos).reshape(1, -1)
        prob = predict_onnx(model_path, feat)
        assert prob.shape == (1, 1)


# ---------------------------------------------------------------------------
# Tests: Full pipeline with mocked recording
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not importlib.util.find_spec("onnx"),
    reason="onnx package not installed",
)
class TestTrainWakewordPipeline:
    """Test the train_wakeword function with mocked recording."""

    def _mock_record_fn(self, positive_freq: float = 800, negative_freq: float = 200):
        """Return a recording function that generates synthetic clips.

        Alternates between positive-like and negative-like frequencies
        based on call count.
        """
        call_count = {"n": 0}

        def mock_record(duration: float = 3.0) -> np.ndarray:
            call_count["n"] += 1
            n = int(SAMPLE_RATE * min(duration, 1.5))
            t = np.arange(n) / SAMPLE_RATE
            # Use different frequencies for first N (positive) vs rest (negative)
            freq = positive_freq if call_count["n"] <= 10 else negative_freq
            return (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)

        return mock_record

    def test_full_pipeline_produces_onnx_file(self, tmp_path):
        """Full pipeline with mocked recording should produce an ONNX model."""
        record_fn = self._mock_record_fn()
        with patch("builtins.input", return_value=""):
            model_path = train_wakeword(
                "hey test",
                output_dir=tmp_path,
                num_samples=5,
                record_fn=record_fn,
            )

        assert model_path.exists()
        assert model_path.suffix == ".onnx"
        assert "wakeword_hey_test" in model_path.name

    def test_pipeline_model_is_valid_onnx(self, tmp_path):
        """Output model should pass ONNX validation."""
        import onnx

        record_fn = self._mock_record_fn()
        with patch("builtins.input", return_value=""):
            model_path = train_wakeword(
                "hey claude",
                output_dir=tmp_path,
                num_samples=5,
                record_fn=record_fn,
            )

        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

    def test_pipeline_model_can_run_inference(self, tmp_path):
        """Output model should be usable for inference."""
        record_fn = self._mock_record_fn()
        with patch("builtins.input", return_value=""):
            model_path = train_wakeword(
                "hey claude",
                output_dir=tmp_path,
                num_samples=5,
                record_fn=record_fn,
            )

        clip = _make_clip(freq_hz=800)
        features = features_from_clip(clip).reshape(1, -1)
        prob = predict_onnx(model_path, features)
        assert prob.shape == (1, 1)
        assert 0.0 <= prob[0, 0] <= 1.0

    def test_pipeline_insufficient_clips_raises(self, tmp_path):
        """Pipeline should raise if too few clips are recorded."""
        call_count = {"n": 0}

        def failing_record(duration: float = 3.0) -> np.ndarray:
            call_count["n"] += 1
            # Return silence (which will fail validation)
            return np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)

        with patch("builtins.input", return_value=""):
            with pytest.raises(ValueError, match="Insufficient positive clips"):
                train_wakeword(
                    "test",
                    output_dir=tmp_path,
                    num_samples=5,
                    record_fn=failing_record,
                )

    def test_pipeline_custom_samples_count(self, tmp_path):
        """Pipeline should respect the num_samples parameter."""
        call_tracker = {"count": 0}

        def counting_record(duration: float = 3.0) -> np.ndarray:
            call_tracker["count"] += 1
            return _make_random_clip(duration_sec=min(duration, 1.5))

        with patch("builtins.input", return_value=""):
            train_wakeword(
                "test",
                output_dir=tmp_path,
                num_samples=3,
                record_fn=counting_record,
            )

        # Should have recorded 3 positive + 3 negative = 6 clips
        assert call_tracker["count"] == 6

    def test_pipeline_output_directory_created(self, tmp_path):
        """Output directory should be created if it doesn't exist."""
        out = tmp_path / "deep" / "nested" / "dir"
        record_fn = self._mock_record_fn()
        with patch("builtins.input", return_value=""):
            model_path = train_wakeword(
                "hey test",
                output_dir=out,
                num_samples=3,
                record_fn=record_fn,
            )
        assert model_path.parent.exists()


# ---------------------------------------------------------------------------
# Tests: CLI command wiring
# ---------------------------------------------------------------------------

class TestCliWiring:
    """Tests that the CLI train-wakeword command is properly wired."""

    def test_train_wakeword_command_calls_pipeline(self):
        """The train-wakeword CLI command should call train_wakeword."""
        with patch("claude_speak.cli.sys") as mock_sys:
            mock_sys.argv = ["claude-speak", "train-wakeword", "hey claude"]
            with patch("claude_speak.train_wakeword.train_wakeword") as mock_train:
                from claude_speak.cli import cmd_train_wakeword

                mock_train.return_value = Path("/tmp/model.onnx")
                cmd_train_wakeword()
                mock_train.assert_called_once()

    def test_train_wakeword_with_samples_flag(self):
        """The --samples flag should be parsed and passed."""
        with patch("claude_speak.cli.sys") as mock_sys:
            mock_sys.argv = ["claude-speak", "train-wakeword", "hey claude", "--samples", "15"]
            with patch("claude_speak.train_wakeword.train_wakeword") as mock_train:
                from claude_speak.cli import cmd_train_wakeword

                mock_train.return_value = Path("/tmp/model.onnx")
                cmd_train_wakeword()
                _, kwargs = mock_train.call_args
                assert kwargs["num_samples"] == 15

    def test_train_wakeword_with_output_flag(self):
        """The --output flag should be parsed and passed."""
        with patch("claude_speak.cli.sys") as mock_sys:
            mock_sys.argv = ["claude-speak", "train-wakeword", "hey claude", "--output", "/tmp/mymodels"]
            with patch("claude_speak.train_wakeword.train_wakeword") as mock_train:
                from claude_speak.cli import cmd_train_wakeword

                mock_train.return_value = Path("/tmp/model.onnx")
                cmd_train_wakeword()
                _, kwargs = mock_train.call_args
                assert kwargs["output_dir"] == Path("/tmp/mymodels")

    def test_train_wakeword_missing_phrase_exits(self):
        """Missing wake phrase should cause sys.exit(1)."""
        with patch("claude_speak.cli.sys") as mock_sys:
            mock_sys.argv = ["claude-speak", "train-wakeword"]
            mock_sys.exit = MagicMock(side_effect=SystemExit(1))
            from claude_speak.cli import cmd_train_wakeword

            with pytest.raises(SystemExit):
                cmd_train_wakeword()

    def test_train_wakeword_in_main_dispatch(self):
        """The main() dispatcher should route train-wakeword to cmd_train_wakeword."""
        with patch("claude_speak.cli.sys") as mock_sys, \
             patch("claude_speak.cli.cmd_train_wakeword") as mock_cmd:
            mock_sys.argv = ["claude-speak", "train-wakeword", "hey test"]
            mock_sys.exit = MagicMock(side_effect=SystemExit)
            # Need platform check to pass
            with patch("claude_speak.cli.platform") as mock_platform:
                mock_platform.system.return_value = "Darwin"
                from claude_speak.cli import main

                main()
                mock_cmd.assert_called_once()

    def test_train_wakeword_in_docstring(self):
        """The CLI docstring should mention train-wakeword."""
        import claude_speak.cli as cli_module

        assert "train-wakeword" in cli_module.__doc__
