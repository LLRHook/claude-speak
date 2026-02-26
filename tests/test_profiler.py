"""
Unit tests for claude_speak/profiler.py — TTS pipeline performance profiling.

All tests mock the TTS engine so they don't require a real model.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from claude_speak.profiler import (
    BENCHMARK_TEXTS,
    BenchmarkReport,
    PipelineProfiler,
    StageStats,
    TimingResult,
    _compute_percentiles,
    _fmt_ms,
    _percentile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_engine(loaded: bool = True) -> MagicMock:
    """Create a mock TTSEngine whose generate_audio returns a small numpy array.

    The mock introduces a tiny sleep to produce measurable timing.
    """
    engine = MagicMock()
    engine._backend = MagicMock()
    engine._backend.is_loaded.return_value = loaded
    engine._backend.name = "mock"

    async def _fake_generate(text: str) -> list[tuple[np.ndarray, int]]:
        # Tiny sleep to simulate generation time
        await asyncio.sleep(0.001)
        samples = np.zeros(1000, dtype=np.float32)
        return [(samples, 24000)]

    engine.generate_audio = AsyncMock(side_effect=_fake_generate)
    return engine


# ---------------------------------------------------------------------------
# Tests: TimingResult dataclass
# ---------------------------------------------------------------------------


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_fields(self):
        tr = TimingResult(text_index=0, text_length=42, stage="normalize", duration_ms=1.5)
        assert tr.text_index == 0
        assert tr.text_length == 42
        assert tr.stage == "normalize"
        assert tr.duration_ms == 1.5

    def test_different_stages(self):
        stages = ["normalize", "chunk", "tts_generate", "total"]
        for stage in stages:
            tr = TimingResult(text_index=0, text_length=10, stage=stage, duration_ms=0.5)
            assert tr.stage == stage


# ---------------------------------------------------------------------------
# Tests: StageStats dataclass
# ---------------------------------------------------------------------------


class TestStageStats:
    """Tests for StageStats dataclass."""

    def test_to_dict_keys(self):
        ss = StageStats(
            stage="normalize", count=10, min_ms=0.1, max_ms=5.0,
            mean_ms=2.0, p50_ms=1.5, p95_ms=4.0, p99_ms=4.8,
        )
        d = ss.to_dict()
        expected_keys = {"stage", "count", "min_ms", "max_ms", "mean_ms", "p50_ms", "p95_ms", "p99_ms"}
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_rounded(self):
        ss = StageStats(
            stage="chunk", count=5, min_ms=0.12345, max_ms=9.87654,
            mean_ms=3.33333, p50_ms=2.22222, p95_ms=8.88888, p99_ms=9.55555,
        )
        d = ss.to_dict()
        assert d["min_ms"] == 0.123
        assert d["max_ms"] == 9.877
        assert d["mean_ms"] == 3.333
        assert d["p50_ms"] == 2.222
        assert d["p95_ms"] == 8.889
        assert d["p99_ms"] == 9.556

    def test_to_dict_json_serializable(self):
        ss = StageStats(
            stage="total", count=3, min_ms=1.0, max_ms=10.0,
            mean_ms=5.0, p50_ms=4.0, p95_ms=9.0, p99_ms=9.8,
        )
        # Should not raise
        json_str = json.dumps(ss.to_dict())
        parsed = json.loads(json_str)
        assert parsed["stage"] == "total"
        assert parsed["count"] == 3


# ---------------------------------------------------------------------------
# Tests: Percentile computation
# ---------------------------------------------------------------------------


class TestPercentileComputation:
    """Tests for _percentile and _compute_percentiles."""

    def test_single_value(self):
        assert _percentile([5.0], 50) == 5.0
        assert _percentile([5.0], 95) == 5.0
        assert _percentile([5.0], 99) == 5.0

    def test_two_values_p50(self):
        result = _percentile([1.0, 3.0], 50)
        assert result == pytest.approx(2.0)

    def test_sorted_list_p50(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _percentile(values, 50)
        assert result == pytest.approx(3.0)

    def test_sorted_list_p95(self):
        # 100 values: 1..100
        values = [float(i) for i in range(1, 101)]
        result = _percentile(values, 95)
        assert result == pytest.approx(95.05, abs=0.1)

    def test_sorted_list_p99(self):
        values = [float(i) for i in range(1, 101)]
        result = _percentile(values, 99)
        assert result == pytest.approx(99.01, abs=0.1)

    def test_empty_list(self):
        assert _percentile([], 50) == 0.0

    def test_compute_percentiles_basic(self):
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _compute_percentiles("test", durations)
        assert stats.stage == "test"
        assert stats.count == 5
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0
        assert stats.mean_ms == pytest.approx(3.0)
        assert stats.p50_ms == pytest.approx(3.0)

    def test_compute_percentiles_single(self):
        stats = _compute_percentiles("x", [42.0])
        assert stats.count == 1
        assert stats.min_ms == 42.0
        assert stats.max_ms == 42.0
        assert stats.mean_ms == 42.0
        assert stats.p50_ms == 42.0
        assert stats.p95_ms == 42.0
        assert stats.p99_ms == 42.0


# ---------------------------------------------------------------------------
# Tests: BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    """Tests for BenchmarkReport dataclass."""

    def _make_report_with_timings(self) -> BenchmarkReport:
        """Create a report with known timings for testing."""
        report = BenchmarkReport(num_texts=3)
        for i in range(3):
            report.timings.append(TimingResult(
                text_index=i, text_length=50 + i * 10,
                stage="normalize", duration_ms=1.0 + i * 0.5,
            ))
            report.timings.append(TimingResult(
                text_index=i, text_length=50 + i * 10,
                stage="chunk", duration_ms=0.1 + i * 0.05,
            ))
            report.timings.append(TimingResult(
                text_index=i, text_length=50 + i * 10,
                stage="total", duration_ms=1.1 + i * 0.55,
            ))
        report.compute_stats()
        return report

    def test_compute_stats_creates_stages(self):
        report = self._make_report_with_timings()
        assert "normalize" in report.stage_stats
        assert "chunk" in report.stage_stats
        assert "total" in report.stage_stats

    def test_compute_stats_counts(self):
        report = self._make_report_with_timings()
        assert report.stage_stats["normalize"].count == 3
        assert report.stage_stats["chunk"].count == 3
        assert report.stage_stats["total"].count == 3

    def test_compute_stats_min_max(self):
        report = self._make_report_with_timings()
        norm = report.stage_stats["normalize"]
        assert norm.min_ms == pytest.approx(1.0)
        assert norm.max_ms == pytest.approx(2.0)

    def test_to_json_keys(self):
        report = self._make_report_with_timings()
        d = report.to_json()
        assert "num_texts" in d
        assert "tts_available" in d
        assert "notes" in d
        assert "stages" in d
        assert "timings" in d

    def test_to_json_serializable(self):
        report = self._make_report_with_timings()
        # Should not raise
        json_str = json.dumps(report.to_json())
        parsed = json.loads(json_str)
        assert parsed["num_texts"] == 3
        assert "normalize" in parsed["stages"]

    def test_to_json_stages_have_stats(self):
        report = self._make_report_with_timings()
        d = report.to_json()
        norm_stats = d["stages"]["normalize"]
        assert "p50_ms" in norm_stats
        assert "p95_ms" in norm_stats
        assert "p99_ms" in norm_stats
        assert "count" in norm_stats

    def test_to_json_timings_count(self):
        report = self._make_report_with_timings()
        d = report.to_json()
        # 3 texts * 3 stages (normalize, chunk, total)
        assert len(d["timings"]) == 9

    def test_to_json_with_notes(self):
        report = BenchmarkReport(num_texts=1, notes=["TTS skipped"])
        report.compute_stats()
        d = report.to_json()
        assert d["notes"] == ["TTS skipped"]

    def test_to_json_tts_available_flag(self):
        report = BenchmarkReport(num_texts=1, tts_available=False)
        report.compute_stats()
        d = report.to_json()
        assert d["tts_available"] is False

    def test_to_text_contains_header(self):
        report = self._make_report_with_timings()
        text = report.to_text()
        assert "Pipeline Benchmark Report" in text
        assert "3 texts" in text

    def test_to_text_contains_stages(self):
        report = self._make_report_with_timings()
        text = report.to_text()
        assert "normalize" in text
        assert "chunk" in text
        assert "total" in text

    def test_to_text_contains_columns(self):
        report = self._make_report_with_timings()
        text = report.to_text()
        assert "Stage" in text
        assert "P50" in text
        assert "P95" in text
        assert "P99" in text
        assert "Min" in text
        assert "Max" in text
        assert "Mean" in text

    def test_to_text_contains_separator(self):
        report = self._make_report_with_timings()
        text = report.to_text()
        lines = text.split("\n")
        # Should have separator lines (all dashes)
        separator_lines = [l for l in lines if l.strip() and set(l.strip()) == {"-"}]
        assert len(separator_lines) >= 1

    def test_to_text_contains_notes(self):
        report = BenchmarkReport(
            num_texts=1,
            notes=["TTS engine not available — tts_generate timings skipped."],
        )
        report.compute_stats()
        text = report.to_text()
        assert "Note:" in text
        assert "TTS engine not available" in text


# ---------------------------------------------------------------------------
# Tests: PipelineProfiler.run_benchmark with mock engine
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """Tests for PipelineProfiler.run_benchmark with mocked TTS engine."""

    def test_basic_run_without_engine(self):
        """Benchmark runs without TTS engine and records normalize/chunk/total."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:3], engine=None)

        assert report.num_texts == 3
        assert report.tts_available is False
        assert len(report.notes) > 0
        assert "normalize" in report.stage_stats
        assert "chunk" in report.stage_stats
        assert "total" in report.stage_stats

    def test_basic_run_with_mock_engine(self):
        """Benchmark runs with mock engine and includes tts_generate stage."""
        engine = _make_mock_engine(loaded=True)
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:2], engine=engine)

        assert report.num_texts == 2
        assert report.tts_available is True
        assert "tts_generate" in report.stage_stats
        assert report.stage_stats["tts_generate"].count == 2

    def test_run_with_unloaded_engine(self):
        """Engine that is not loaded results in TTS being skipped."""
        engine = _make_mock_engine(loaded=False)
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:2], engine=engine)

        assert report.tts_available is False
        assert "tts_generate" not in report.stage_stats

    def test_timing_values_are_positive(self):
        """All timing measurements should be positive."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:5], engine=None)

        for timing in report.timings:
            assert timing.duration_ms >= 0.0, (
                f"Negative timing: {timing.stage} = {timing.duration_ms}ms"
            )

    def test_total_includes_normalize_and_chunk(self):
        """Total timing should be >= normalize + chunk for each text."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:3], engine=None)

        for idx in range(3):
            idx_timings = [t for t in report.timings if t.text_index == idx]
            norm = next(t for t in idx_timings if t.stage == "normalize")
            chunk = next(t for t in idx_timings if t.stage == "chunk")
            total = next(t for t in idx_timings if t.stage == "total")
            assert total.duration_ms >= norm.duration_ms + chunk.duration_ms - 0.01

    def test_text_index_and_length_tracked(self):
        """TimingResult should track text index and length correctly."""
        texts = ["short", "a longer sentence with more words"]
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(texts, engine=None)

        idx0_timings = [t for t in report.timings if t.text_index == 0]
        idx1_timings = [t for t in report.timings if t.text_index == 1]

        assert all(t.text_length == len("short") for t in idx0_timings)
        assert all(
            t.text_length == len("a longer sentence with more words")
            for t in idx1_timings
        )

    def test_all_benchmark_texts_run(self):
        """All 10 built-in benchmark texts should run without error."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS, engine=None)

        assert report.num_texts == len(BENCHMARK_TEXTS)
        # Each text produces normalize + chunk + total = 3 timings (no TTS)
        assert len(report.timings) == len(BENCHMARK_TEXTS) * 3

    def test_tts_generate_called_for_each_text(self):
        """Mock engine's generate_audio is called for each text."""
        engine = _make_mock_engine(loaded=True)
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:3], engine=engine)

        # generate_audio is called at least once per text
        # (could be more if chunking produces multiple chunks)
        assert engine.generate_audio.call_count >= 3

    def test_report_json_roundtrip(self):
        """Report serialized to JSON and back preserves structure."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:3], engine=None)

        d = report.to_json()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)

        assert parsed["num_texts"] == 3
        assert "normalize" in parsed["stages"]
        assert "chunk" in parsed["stages"]
        assert "total" in parsed["stages"]
        assert len(parsed["timings"]) == 9  # 3 texts * 3 stages

    def test_report_text_output_not_empty(self):
        """Human-readable output should contain meaningful content."""
        profiler = PipelineProfiler()
        report = profiler.run_benchmark(BENCHMARK_TEXTS[:2], engine=None)

        text = report.to_text()
        assert len(text) > 100
        assert "ms" in text


# ---------------------------------------------------------------------------
# Tests: _fmt_ms helper
# ---------------------------------------------------------------------------


class TestFmtMs:
    """Tests for the _fmt_ms formatting helper."""

    def test_sub_millisecond(self):
        result = _fmt_ms(0.123)
        assert result == "0.123ms"

    def test_normal_range(self):
        result = _fmt_ms(12.34)
        assert result == "12.34ms"

    def test_hundred_range(self):
        result = _fmt_ms(123.4)
        assert result == "123.4ms"

    def test_large_value(self):
        result = _fmt_ms(1234.5)
        assert result == "1234ms" or result == "1235ms"
