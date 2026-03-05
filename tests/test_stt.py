"""
Unit tests for claude_speak/stt.py — Speech-to-text interface and MLX Whisper backend.

All tests mock mlx_whisper so they run without the actual package installed.
"""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.stt import MLXWhisperRecognizer, SpeechRecognizer, get_recognizer


# ---------------------------------------------------------------------------
# Tests: SpeechRecognizer abstract base class
# ---------------------------------------------------------------------------


class TestSpeechRecognizerABC:
    """Tests for the abstract SpeechRecognizer base class."""

    def test_cannot_instantiate_directly(self):
        """SpeechRecognizer is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            SpeechRecognizer()

    def test_subclass_must_implement_transcribe(self):
        """A subclass missing transcribe() cannot be instantiated."""

        class Incomplete(SpeechRecognizer):
            def is_available(self) -> bool:
                return True

            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_is_available(self):
        """A subclass missing is_available() cannot be instantiated."""

        class Incomplete(SpeechRecognizer):
            def transcribe(self, audio, sample_rate=16000):
                return ""

            @property
            def name(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_must_implement_name(self):
        """A subclass missing name property cannot be instantiated."""

        class Incomplete(SpeechRecognizer):
            def transcribe(self, audio, sample_rate=16000):
                return ""

            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_can_be_instantiated(self):
        """A subclass implementing all abstract methods can be instantiated."""

        class Complete(SpeechRecognizer):
            def transcribe(self, audio, sample_rate=16000):
                return "hello"

            def is_available(self) -> bool:
                return True

            @property
            def name(self) -> str:
                return "complete"

        instance = Complete()
        assert instance.name == "complete"
        assert instance.is_available() is True
        assert instance.transcribe(np.zeros(100)) == "hello"


# ---------------------------------------------------------------------------
# Tests: MLXWhisperRecognizer properties
# ---------------------------------------------------------------------------


class TestMLXWhisperRecognizerProperties:
    """Tests for MLXWhisperRecognizer name, model mapping, and init."""

    def test_name_includes_model_size(self):
        recognizer = MLXWhisperRecognizer(model="base")
        assert recognizer.name == "MLX Whisper (base)"

    def test_name_with_tiny_model(self):
        recognizer = MLXWhisperRecognizer(model="tiny")
        assert recognizer.name == "MLX Whisper (tiny)"

    def test_name_with_small_model(self):
        recognizer = MLXWhisperRecognizer(model="small")
        assert recognizer.name == "MLX Whisper (small)"

    def test_name_with_medium_model(self):
        recognizer = MLXWhisperRecognizer(model="medium")
        assert recognizer.name == "MLX Whisper (medium)"

    def test_name_with_custom_model_id(self):
        """Custom HF repo ID is stored as-is for the model size in the name."""
        recognizer = MLXWhisperRecognizer(model="mlx-community/whisper-large-v3-turbo")
        assert "mlx-community/whisper-large-v3-turbo" in recognizer.name

    def test_model_map_has_expected_entries(self):
        assert "tiny" in MLXWhisperRecognizer.MODEL_MAP
        assert "base" in MLXWhisperRecognizer.MODEL_MAP
        assert "small" in MLXWhisperRecognizer.MODEL_MAP
        assert "medium" in MLXWhisperRecognizer.MODEL_MAP

    def test_model_map_values_are_hf_repos(self):
        for size, repo in MLXWhisperRecognizer.MODEL_MAP.items():
            assert repo.startswith("mlx-community/whisper-"), f"Unexpected repo for {size}: {repo}"

    def test_default_model_is_base(self):
        recognizer = MLXWhisperRecognizer()
        assert recognizer._model_size == "base"
        assert recognizer._model_id == "mlx-community/whisper-base"

    def test_known_model_size_resolves_to_hf_repo(self):
        recognizer = MLXWhisperRecognizer(model="small")
        assert recognizer._model_id == "mlx-community/whisper-small"

    def test_unknown_model_used_as_hf_repo_directly(self):
        """If the model name is not in MODEL_MAP, it is used as-is (custom HF repo)."""
        recognizer = MLXWhisperRecognizer(model="mlx-community/whisper-large-v3-turbo")
        assert recognizer._model_id == "mlx-community/whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# Tests: MLXWhisperRecognizer.is_available
# ---------------------------------------------------------------------------


class TestMLXWhisperIsAvailable:
    """Tests for is_available with and without mlx_whisper installed."""

    def test_is_available_returns_false_when_not_installed(self):
        """When mlx_whisper is not importable, is_available returns False."""
        recognizer = MLXWhisperRecognizer(model="base")
        # Ensure mlx_whisper is not in sys.modules (it shouldn't be in test env)
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            assert recognizer.is_available() is False

    def test_is_available_returns_true_when_installed(self):
        """When mlx_whisper can be imported, is_available returns True."""
        mock_module = ModuleType("mlx_whisper")
        recognizer = MLXWhisperRecognizer(model="base")
        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            assert recognizer.is_available() is True


# ---------------------------------------------------------------------------
# Tests: MLXWhisperRecognizer.transcribe
# ---------------------------------------------------------------------------


class TestMLXWhisperTranscribe:
    """Tests for the transcribe method with mocked mlx_whisper."""

    def _make_mock_mlx_whisper(self, text="Hello world"):
        """Create a mock mlx_whisper module with a transcribe function."""
        mock_module = ModuleType("mlx_whisper")
        mock_transcribe = MagicMock(return_value={"text": text, "segments": [], "language": "en"})
        mock_module.transcribe = mock_transcribe
        return mock_module, mock_transcribe

    def test_transcribe_calls_mlx_whisper_with_correct_args(self):
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("Hello world")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            audio = np.zeros(16000, dtype=np.float32)
            result = recognizer.transcribe(audio, sample_rate=16000)

        assert result == "Hello world"
        mock_transcribe.assert_called_once()
        call_kwargs = mock_transcribe.call_args
        # First positional arg is the audio array
        passed_audio = call_kwargs[0][0]
        assert isinstance(passed_audio, np.ndarray)
        assert passed_audio.dtype == np.float32
        # Keyword args
        assert call_kwargs[1]["path_or_hf_repo"] == "mlx-community/whisper-base"
        assert call_kwargs[1]["language"] == "en"

    def test_transcribe_strips_whitespace(self):
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("  Hello world  ")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            audio = np.zeros(16000, dtype=np.float32)
            result = recognizer.transcribe(audio)

        assert result == "Hello world"

    def test_transcribe_returns_empty_string_when_no_text(self):
        mock_module = ModuleType("mlx_whisper")
        mock_module.transcribe = MagicMock(return_value={"segments": [], "language": "en"})

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            audio = np.zeros(16000, dtype=np.float32)
            result = recognizer.transcribe(audio)

        assert result == ""

    def test_transcribe_converts_non_float32_input(self):
        """Audio with dtype other than float32 should be converted."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("converted")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            audio_int16 = np.zeros(16000, dtype=np.int16)
            result = recognizer.transcribe(audio_int16, sample_rate=16000)

        assert result == "converted"
        passed_audio = mock_transcribe.call_args[0][0]
        assert passed_audio.dtype == np.float32

    def test_transcribe_resamples_non_16khz_input(self):
        """Audio at a sample rate other than 16kHz should be resampled."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("resampled")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            # 1 second of audio at 48kHz = 48000 samples
            audio_48k = np.zeros(48000, dtype=np.float32)
            result = recognizer.transcribe(audio_48k, sample_rate=48000)

        assert result == "resampled"
        # After resampling 48kHz -> 16kHz: 1 second = 16000 samples
        passed_audio = mock_transcribe.call_args[0][0]
        assert len(passed_audio) == 16000

    def test_transcribe_resamples_8khz_to_16khz(self):
        """Audio at 8kHz should be upsampled to 16kHz."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("upsampled")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            # 1 second of audio at 8kHz = 8000 samples
            audio_8k = np.ones(8000, dtype=np.float32) * 0.5
            result = recognizer.transcribe(audio_8k, sample_rate=8000)

        assert result == "upsampled"
        passed_audio = mock_transcribe.call_args[0][0]
        assert len(passed_audio) == 16000
        assert passed_audio.dtype == np.float32

    def test_transcribe_no_resample_at_16khz(self):
        """Audio already at 16kHz should not be resampled."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("no resample")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="base")
            audio = np.zeros(16000, dtype=np.float32)
            result = recognizer.transcribe(audio, sample_rate=16000)

        assert result == "no resample"
        passed_audio = mock_transcribe.call_args[0][0]
        assert len(passed_audio) == 16000

    def test_transcribe_uses_correct_model_id(self):
        """The model ID passed to mlx_whisper.transcribe should match the configured model."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("test")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="small")
            audio = np.zeros(16000, dtype=np.float32)
            recognizer.transcribe(audio)

        assert mock_transcribe.call_args[1]["path_or_hf_repo"] == "mlx-community/whisper-small"

    def test_transcribe_with_custom_hf_repo(self):
        """Custom HF repo IDs should be passed through to mlx_whisper."""
        mock_module, mock_transcribe = self._make_mock_mlx_whisper("custom")

        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = MLXWhisperRecognizer(model="mlx-community/whisper-large-v3-turbo")
            audio = np.zeros(16000, dtype=np.float32)
            recognizer.transcribe(audio)

        assert mock_transcribe.call_args[1]["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


# ---------------------------------------------------------------------------
# Tests: get_recognizer factory function
# ---------------------------------------------------------------------------


class TestGetRecognizer:
    """Tests for the get_recognizer factory function."""

    def test_auto_backend_returns_mlx_when_available(self):
        mock_module = ModuleType("mlx_whisper")
        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = get_recognizer(backend="auto", model="base")
        assert isinstance(recognizer, MLXWhisperRecognizer)
        assert recognizer._model_size == "base"

    def test_mlx_backend_returns_mlx_when_available(self):
        mock_module = ModuleType("mlx_whisper")
        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = get_recognizer(backend="mlx", model="tiny")
        assert isinstance(recognizer, MLXWhisperRecognizer)
        assert recognizer._model_size == "tiny"

    def test_auto_backend_raises_when_nothing_available(self):
        with patch.dict(sys.modules, {"mlx_whisper": None, "faster_whisper": None}):
            with pytest.raises(RuntimeError, match="No STT backend available"):
                get_recognizer(backend="auto")

    def test_mlx_backend_raises_when_not_available(self):
        with patch.dict(sys.modules, {"mlx_whisper": None}):
            with pytest.raises(RuntimeError, match="MLX Whisper requested but not available"):
                get_recognizer(backend="mlx")

    def test_unknown_backend_raises(self):
        with pytest.raises(RuntimeError, match="No STT backend available"):
            get_recognizer(backend="nonexistent")

    def test_auto_backend_passes_model_to_recognizer(self):
        mock_module = ModuleType("mlx_whisper")
        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = get_recognizer(backend="auto", model="small")
        assert recognizer._model_id == "mlx-community/whisper-small"

    def test_factory_default_args(self):
        """Default args should be backend='auto', model='base'."""
        mock_module = ModuleType("mlx_whisper")
        with patch.dict(sys.modules, {"mlx_whisper": mock_module}):
            recognizer = get_recognizer()
        assert isinstance(recognizer, MLXWhisperRecognizer)
        assert recognizer._model_size == "base"


# ---------------------------------------------------------------------------
# Tests: MLXWhisperRecognizer is a valid SpeechRecognizer
# ---------------------------------------------------------------------------


class TestMLXWhisperIsASpeechRecognizer:
    """Verify MLXWhisperRecognizer satisfies the SpeechRecognizer interface."""

    def test_is_subclass_of_speech_recognizer(self):
        assert issubclass(MLXWhisperRecognizer, SpeechRecognizer)

    def test_instance_passes_isinstance_check(self):
        recognizer = MLXWhisperRecognizer(model="base")
        assert isinstance(recognizer, SpeechRecognizer)
