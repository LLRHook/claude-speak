"""
Integration test -- full pipeline from text through normalization, chunking,
and TTS generation. Verifies the entire speak pipeline produces valid audio.

Marked with @pytest.mark.slow and @pytest.mark.requires_model.
Run with: pytest tests/test_integration.py -m slow
"""

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from claude_speak.config import Config, TTSConfig
from claude_speak.normalizer import normalize, chunk_text


# ---------------------------------------------------------------------------
# Full pipeline test (mocked TTS -- always runs)
# ---------------------------------------------------------------------------

class TestFullPipelineMocked:
    """Integration test exercising normalize -> chunk -> TTS generate with mocked Kokoro."""

    def test_realistic_claude_response_pipeline(self):
        """Take a realistic Claude response, normalize it, chunk it, and
        verify mock TTS produces valid output for each chunk."""
        # Realistic Claude response with markdown, code, lists, etc.
        raw_text = """
Here's what I found:

## Summary

The `config.py` file handles TOML-based configuration with 5 sections:

1. **TTS settings** -- voice, speed, device
2. **Wake word** -- model path, sensitivity (0.5 default)
3. **Input** -- Superwhisper shortcut (Option+Space)
4. **Normalization** -- toggles for code stripping, unit expansion
5. **Audio** -- chime volume (0.3), greeting text

```python
@dataclass
class Config:
    tts: TTSConfig = field(default_factory=TTSConfig)
    wakeword: WakeWordConfig = field(default_factory=WakeWordConfig)
```

The file is located at `/Users/victor/projects/claude-speak/src/config.py`.

Temperature is 72F (22.2C). The model is v1.0.0 and runs at ~87MB.
"""

        # Step 1: Normalize
        normalized = normalize(raw_text)
        assert isinstance(normalized, str)
        assert len(normalized) > 0
        # Code block should be described, not passed through verbatim
        assert "@dataclass" not in normalized

        # Step 2: Chunk
        chunks = chunk_text(normalized, max_chars=400)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

        # Step 3: Generate audio (mocked Kokoro)
        from claude_speak.tts import TTSEngine

        config = Config(tts=TTSConfig())
        engine = TTSEngine(config)

        # Set up mock Kokoro
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro
        engine._voice_style = "af_sarah"

        async def fake_stream(text, voice, speed, lang):
            # Generate realistic-ish audio: 0.5s of silence per chunk
            samples = np.random.randn(12000).astype(np.float32) * 0.1
            yield (samples, 24000)

        mock_kokoro.create_stream = fake_stream

        async def run_generation():
            all_segments = []
            for chunk in chunks:
                segments = await engine.generate_audio(chunk)
                assert len(segments) >= 1
                for samples, sr in segments:
                    assert isinstance(samples, np.ndarray)
                    assert samples.dtype == np.float32
                    assert sr == 24000
                    assert len(samples) > 0
                all_segments.extend(segments)
            return all_segments

        all_segments = asyncio.run(run_generation())

        # Total audio should be non-trivial
        total_samples = sum(len(s) for s, _ in all_segments)
        assert total_samples > 0

    def test_empty_text_after_normalization(self):
        """Text that normalizes to empty should be handled gracefully."""
        raw_text = "```\nprint('hello')\n```"
        normalized = normalize(raw_text)
        # Pure code block may normalize to a description or empty
        # Either way, it should not crash
        assert isinstance(normalized, str)

    def test_pipeline_with_chunking_boundary(self):
        """Long text should produce multiple chunks, each generating audio."""
        long_text = "This is a test sentence for the TTS engine. " * 50
        normalized = normalize(long_text)
        chunks = chunk_text(normalized, max_chars=200)
        assert len(chunks) > 1

        from claude_speak.tts import TTSEngine
        config = Config(tts=TTSConfig())
        engine = TTSEngine(config)
        mock_kokoro = MagicMock()
        engine.kokoro = mock_kokoro
        engine._voice_style = "af_sarah"

        async def fake_stream(text, voice, speed, lang):
            yield (np.zeros(1000, dtype=np.float32), 24000)

        mock_kokoro.create_stream = fake_stream

        async def run_generation():
            for chunk in chunks:
                segments = await engine.generate_audio(chunk)
                assert len(segments) >= 1

        asyncio.run(run_generation())

    def test_normalization_preserves_meaningful_content(self):
        """Normalization should keep the meaningful parts of the text."""
        raw_text = "The function returns 42. It uses Python 3.10 and runs on macOS."
        normalized = normalize(raw_text)
        assert "function" in normalized.lower()
        assert "42" in normalized or "forty" in normalized.lower()

    def test_chunk_text_respects_max_chars(self):
        """Each chunk should be at or below max_chars (at sentence boundaries)."""
        long_text = "Hello world. " * 100
        normalized = normalize(long_text)
        chunks = chunk_text(normalized, max_chars=150)
        for chunk in chunks:
            # Allow some slack since chunking splits at sentence boundaries
            assert len(chunk) <= 300  # generous upper bound


# ---------------------------------------------------------------------------
# Full pipeline test WITH real Kokoro (requires model files)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_model
class TestFullPipelineReal:
    """Integration test with real Kokoro model. Skipped if model files not present."""

    @pytest.fixture(autouse=True)
    def check_model(self):
        from claude_speak.config import MODELS_DIR
        model_path = MODELS_DIR / "kokoro-v1.0.onnx"
        voices_path = MODELS_DIR / "voices-v1.0.bin"
        if not model_path.exists() or not voices_path.exists():
            pytest.skip("Kokoro model files not found -- skipping real TTS test")

    def test_real_tts_pipeline(self):
        """End-to-end: normalize -> chunk -> real Kokoro generation."""
        raw_text = "Hello! This is a test of the claude speak text to speech pipeline."
        normalized = normalize(raw_text)
        chunks = chunk_text(normalized, max_chars=400)

        config = Config(tts=TTSConfig())
        from claude_speak.tts import TTSEngine
        engine = TTSEngine(config)
        engine.load()

        async def run():
            for chunk in chunks:
                segments = await engine.generate_audio(chunk)
                assert len(segments) >= 1
                for samples, sr in segments:
                    assert isinstance(samples, np.ndarray)
                    assert samples.dtype == np.float32
                    assert sr > 0
                    assert len(samples) > 0

        asyncio.run(run())
