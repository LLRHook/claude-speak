"""
Performance profiler for the claude-speak TTS pipeline.

Profiles each stage of the pipeline (normalize, chunk, TTS generate) and
produces latency statistics (P50, P95, P99) in human-readable and JSON formats.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .normalizer import chunk_text, normalize
from .tts import TTSEngine

# ---------------------------------------------------------------------------
# Representative benchmark texts
# ---------------------------------------------------------------------------

BENCHMARK_TEXTS = [
    "Sure, I can help with that.",
    "The function returns a list of integers sorted in ascending order.",
    (
        "I've made the following changes to your code:\n\n"
        "1. Added error handling for the API call\n"
        "2. Refactored the database query to use parameterized inputs\n"
        "3. Updated the test suite with three new test cases"
    ),
    (
        "Looking at `src/utils/helpers.py`, the `parse_config()` function on "
        "line 42 has a bug where it doesn't handle empty TOML sections."
    ),
    (
        "The error `TypeError: cannot unpack non-sequence NoneType` occurs "
        "because the function returns None when the API key is missing. You "
        "need to add a check before destructuring."
    ),
    (
        "Let me create a new file at "
        "`/Users/example/project/src/components/Dashboard.tsx` "
        "with the React component."
    ),
    (
        "The CI/CD pipeline uses GitHub Actions with three jobs: "
        "lint (ruff + mypy), test (pytest on Python 3.11 and 3.13), "
        "and build (wheel + sdist)."
    ),
    (
        "Here's a comparison:\n\n"
        "| Feature | Option A | Option B |\n"
        "|---------|----------|----------|\n"
        "| Speed | 100ms | 250ms |\n"
        "| Memory | 50MB | 120MB |\n"
        "| Cost | Free | $0.01/req |"
    ),
    (
        "Version 2.3.1 fixes CVE-2024-1234 by sanitizing user input "
        "in the /api/v2/users endpoint."
    ),
    (
        "The `async def fetch_data(url: str, timeout: float = 30.0) "
        "-> dict[str, Any]` function uses aiohttp to make HTTP GET "
        "requests with exponential backoff retry logic."
    ),
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Timing for a single text through one pipeline stage."""

    text_index: int
    text_length: int
    stage: str
    duration_ms: float


@dataclass
class StageStats:
    """Aggregate latency statistics for a single pipeline stage."""

    stage: str
    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


@dataclass
class BenchmarkReport:
    """Full benchmark report with per-stage statistics."""

    num_texts: int
    timings: list[TimingResult] = field(default_factory=list)
    stage_stats: dict[str, StageStats] = field(default_factory=dict)
    tts_available: bool = True
    notes: list[str] = field(default_factory=list)

    def compute_stats(self) -> None:
        """Compute P50/P95/P99 statistics from collected timings."""
        stages: dict[str, list[float]] = {}
        for t in self.timings:
            stages.setdefault(t.stage, []).append(t.duration_ms)

        self.stage_stats = {}
        for stage, durations in stages.items():
            self.stage_stats[stage] = _compute_percentiles(stage, durations)

    def to_json(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict."""
        return {
            "num_texts": self.num_texts,
            "tts_available": self.tts_available,
            "notes": self.notes,
            "stages": {
                name: stats.to_dict()
                for name, stats in self.stage_stats.items()
            },
            "timings": [
                {
                    "text_index": t.text_index,
                    "text_length": t.text_length,
                    "stage": t.stage,
                    "duration_ms": round(t.duration_ms, 3),
                }
                for t in self.timings
            ],
        }

    def to_text(self) -> str:
        """Format a human-readable summary table."""
        lines: list[str] = []
        lines.append(f"Pipeline Benchmark Report ({self.num_texts} texts)")
        lines.append("=" * 72)

        if self.notes:
            for note in self.notes:
                lines.append(f"  Note: {note}")
            lines.append("")

        # Table header
        header = (
            f"{'Stage':<16s} {'Count':>6s} {'Min':>10s} {'P50':>10s} "
            f"{'P95':>10s} {'P99':>10s} {'Max':>10s} {'Mean':>10s}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        # Sort stages in pipeline order
        stage_order = ["normalize", "chunk", "tts_generate", "total"]
        sorted_stages = sorted(
            self.stage_stats.items(),
            key=lambda kv: stage_order.index(kv[0]) if kv[0] in stage_order else 99,
        )

        for _name, stats in sorted_stages:
            row = (
                f"{stats.stage:<16s} {stats.count:>6d} "
                f"{_fmt_ms(stats.min_ms):>10s} {_fmt_ms(stats.p50_ms):>10s} "
                f"{_fmt_ms(stats.p95_ms):>10s} {_fmt_ms(stats.p99_ms):>10s} "
                f"{_fmt_ms(stats.max_ms):>10s} {_fmt_ms(stats.mean_ms):>10s}"
            )
            lines.append(row)

        lines.append("-" * len(header))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Profiles the claude-speak TTS pipeline stages."""

    def run_benchmark(
        self,
        texts: list[str],
        engine: TTSEngine | None = None,
    ) -> BenchmarkReport:
        """Run each text through normalize -> chunk -> generate and time each stage.

        If *engine* is None or the TTS backend is not loaded, the TTS stage
        is skipped and a note is added to the report.
        """
        import asyncio

        report = BenchmarkReport(num_texts=len(texts))

        # Determine TTS availability
        tts_ok = False
        if engine is not None:
            try:
                tts_ok = engine._backend.is_loaded()
            except Exception:
                tts_ok = False

        if not tts_ok:
            report.tts_available = False
            report.notes.append(
                "TTS engine not available — tts_generate timings skipped."
            )

        for idx, text in enumerate(texts):
            text_len = len(text)

            # --- normalize ---
            t0 = time.perf_counter()
            normalized = normalize(text)
            t1 = time.perf_counter()
            report.timings.append(TimingResult(
                text_index=idx,
                text_length=text_len,
                stage="normalize",
                duration_ms=(t1 - t0) * 1000,
            ))

            # --- chunk ---
            t2 = time.perf_counter()
            chunks = chunk_text(normalized)
            t3 = time.perf_counter()
            report.timings.append(TimingResult(
                text_index=idx,
                text_length=text_len,
                stage="chunk",
                duration_ms=(t3 - t2) * 1000,
            ))

            # --- tts_generate ---
            tts_duration = 0.0
            if tts_ok and engine is not None:
                t4 = time.perf_counter()
                for chunk in chunks:
                    asyncio.run(engine.generate_audio(chunk))
                t5 = time.perf_counter()
                tts_duration = (t5 - t4) * 1000
                report.timings.append(TimingResult(
                    text_index=idx,
                    text_length=text_len,
                    stage="tts_generate",
                    duration_ms=tts_duration,
                ))

            # --- total ---
            total_ms = (t1 - t0) * 1000 + (t3 - t2) * 1000 + tts_duration
            report.timings.append(TimingResult(
                text_index=idx,
                text_length=text_len,
                stage="total",
                duration_ms=total_ms,
            ))

        report.compute_stats()
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from a sorted list of values.

    Uses the 'nearest rank' method.
    """
    if not sorted_values:
        return 0.0
    k = (p / 100.0) * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    # Linear interpolation
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _compute_percentiles(stage: str, durations: list[float]) -> StageStats:
    """Compute aggregate statistics from a list of duration measurements."""
    s = sorted(durations)
    count = len(s)
    return StageStats(
        stage=stage,
        count=count,
        min_ms=s[0] if s else 0.0,
        max_ms=s[-1] if s else 0.0,
        mean_ms=sum(s) / count if count else 0.0,
        p50_ms=_percentile(s, 50),
        p95_ms=_percentile(s, 95),
        p99_ms=_percentile(s, 99),
    )


def _fmt_ms(value: float) -> str:
    """Format a millisecond value for display."""
    if value < 1.0:
        return f"{value:.3f}ms"
    elif value < 100.0:
        return f"{value:.2f}ms"
    elif value < 1000.0:
        return f"{value:.1f}ms"
    else:
        return f"{value:.0f}ms"
