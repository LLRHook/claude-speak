#!/usr/bin/env python3
"""
Wake word accuracy benchmarking suite.

Evaluates a wake word model against labeled audio datasets at multiple
sensitivity thresholds, computing standard classification metrics (recall,
precision, F1, FPR, accuracy) and optionally generating an ROC curve.

Usage:
    python -m train.benchmark --model models/stop.onnx \
        --positive train/collected/positive/ \
        --negative train/collected/negative/ \
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Audio parameters matching openwakeword expectations
_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1280  # 80ms at 16kHz

DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ThresholdMetrics:
    """Classification metrics at a single threshold."""

    threshold: float
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def recall(self) -> float:
        """True Positive Rate = TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / total."""
        total = (
            self.true_positives
            + self.false_positives
            + self.true_negatives
            + self.false_negatives
        )
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "recall": round(self.recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass
class BenchmarkResults:
    """Full benchmark results across all thresholds."""

    model_path: str
    positive_dir: str
    negative_dir: str
    num_positive: int
    num_negative: int
    thresholds: list[ThresholdMetrics] = field(default_factory=list)
    positive_scores: list[float] = field(default_factory=list)
    negative_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "model_path": self.model_path,
            "positive_dir": self.positive_dir,
            "negative_dir": self.negative_dir,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "positive_scores": [round(s, 4) for s in self.positive_scores],
            "negative_scores": [round(s, 4) for s in self.negative_scores],
        }


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_wav(filepath: Path) -> np.ndarray:
    """Load a WAV file as int16 samples at 16kHz mono.

    Args:
        filepath: path to a .wav file

    Returns:
        1D numpy array of int16 samples

    Raises:
        ValueError: if the WAV file cannot be read or is not mono
    """
    try:
        with wave.open(str(filepath), "rb") as wf:
            if wf.getnchannels() != 1:
                raise ValueError(
                    f"Expected mono audio, got {wf.getnchannels()} channels: {filepath}"
                )
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)
            samples = np.frombuffer(raw_bytes, dtype=np.int16)

            # Resample if not 16kHz
            if sample_rate != _SAMPLE_RATE:
                n_new = int(len(samples) * _SAMPLE_RATE / sample_rate)
                indices = np.linspace(0, len(samples) - 1, n_new)
                samples = np.interp(
                    indices, np.arange(len(samples)), samples.astype(np.float64)
                ).astype(np.int16)

            return samples
    except wave.Error as e:
        raise ValueError(f"Could not read WAV file {filepath}: {e}") from e


def scan_wav_files(directory: Path) -> list[Path]:
    """Find all .wav files in a directory (non-recursive).

    Args:
        directory: path to scan

    Returns:
        sorted list of Path objects for .wav files
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.wav"))


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def score_clip(model, samples: np.ndarray) -> float:
    """Run a full audio clip through an openwakeword model.

    Feeds the clip in 80ms chunks and returns the maximum prediction
    score across all chunks and all model names.

    Args:
        model: an openwakeword Model instance
        samples: int16 audio samples (1D array)

    Returns:
        peak prediction score (float in [0, 1])
    """
    max_score = 0.0

    # Feed in chunks of _CHUNK_SAMPLES
    n_chunks = len(samples) // _CHUNK_SAMPLES
    for i in range(n_chunks):
        chunk = samples[i * _CHUNK_SAMPLES : (i + 1) * _CHUNK_SAMPLES]
        prediction = model.predict(chunk)
        for score in prediction.values():
            if score > max_score:
                max_score = score

    # Reset model state between clips so predictions don't bleed
    model.reset()

    return max_score


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    positive_scores: list[float],
    negative_scores: list[float],
    thresholds: list[float],
) -> list[ThresholdMetrics]:
    """Compute classification metrics at each threshold.

    Args:
        positive_scores: model scores for positive (wake word) clips
        negative_scores: model scores for negative clips
        thresholds: list of decision thresholds to evaluate

    Returns:
        list of ThresholdMetrics, one per threshold
    """
    results = []
    for threshold in thresholds:
        metrics = ThresholdMetrics(threshold=threshold)
        for score in positive_scores:
            if score >= threshold:
                metrics.true_positives += 1
            else:
                metrics.false_negatives += 1
        for score in negative_scores:
            if score >= threshold:
                metrics.false_positives += 1
            else:
                metrics.true_negatives += 1
        results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def format_table(metrics_list: list[ThresholdMetrics]) -> str:
    """Format metrics as a console-friendly table.

    Args:
        metrics_list: list of ThresholdMetrics to display

    Returns:
        multi-line string with header and data rows
    """
    header = f"{'Threshold':<11}| {'Recall':<8}| {'FPR':<8}| {'Precision':<11}| {'F1':<8}| {'Accuracy':<8}"
    separator = "-" * len(header)
    lines = [header, separator]

    for m in metrics_list:
        line = (
            f"{m.threshold:<11.2f}| "
            f"{m.recall:<8.4f}| "
            f"{m.false_positive_rate:<8.4f}| "
            f"{m.precision:<11.4f}| "
            f"{m.f1_score:<8.4f}| "
            f"{m.accuracy:<8.4f}"
        )
        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ROC curve (optional)
# ---------------------------------------------------------------------------


def generate_roc_curve(
    positive_scores: list[float],
    negative_scores: list[float],
    output_path: Path,
) -> bool:
    """Generate and save an ROC curve image.

    Requires matplotlib. Returns True if the image was saved, False if
    matplotlib is not available.

    Args:
        positive_scores: model scores for positive clips
        negative_scores: model scores for negative clips
        output_path: path to save the PNG image

    Returns:
        True if image was saved, False if matplotlib unavailable
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, skipping ROC curve generation")
        return False

    # Generate fine-grained thresholds for smooth curve
    fine_thresholds = np.linspace(0.0, 1.0, 201).tolist()
    metrics = compute_metrics(positive_scores, negative_scores, fine_thresholds)

    fprs = [m.false_positive_rate for m in metrics]
    tprs = [m.recall for m in metrics]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fprs, tprs, "b-", linewidth=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Wake Word Detection ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC curve saved to %s", output_path)
    return True


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------


class WakeWordBenchmark:
    """Benchmarks a wake word model against labeled audio datasets.

    Usage::

        benchmark = WakeWordBenchmark()
        results = benchmark.run(
            model_path="models/stop.onnx",
            positive_dir="train/collected/positive/",
            negative_dir="train/collected/negative/",
        )
    """

    def run(
        self,
        model_path: str,
        positive_dir: str,
        negative_dir: str,
        thresholds: Optional[list[float]] = None,
        output_json: Optional[str] = None,
        roc_image: Optional[str] = None,
    ) -> BenchmarkResults:
        """Run the benchmark suite.

        Args:
            model_path: path to an openwakeword-compatible ONNX model
            positive_dir: directory containing positive (wake word) .wav files
            negative_dir: directory containing negative .wav files
            thresholds: sensitivity thresholds to evaluate (default: [0.3..0.7])
            output_json: optional path to save JSON results
            roc_image: optional path to save ROC curve PNG

        Returns:
            BenchmarkResults with all metrics and scores
        """
        if thresholds is None:
            thresholds = list(DEFAULT_THRESHOLDS)

        model_path_obj = Path(model_path)
        pos_dir = Path(positive_dir)
        neg_dir = Path(negative_dir)

        # Load model
        logger.info("Loading model: %s", model_path)
        model = self._load_model(str(model_path_obj))

        # Scan directories
        pos_files = scan_wav_files(pos_dir)
        neg_files = scan_wav_files(neg_dir)

        logger.info(
            "Found %d positive, %d negative clips", len(pos_files), len(neg_files)
        )

        if not pos_files and not neg_files:
            logger.warning("No audio files found in either directory")

        # Score all clips
        positive_scores = self._score_files(model, pos_files, "positive")
        negative_scores = self._score_files(model, neg_files, "negative")

        # Compute metrics
        metrics_list = compute_metrics(positive_scores, negative_scores, thresholds)

        # Build results
        results = BenchmarkResults(
            model_path=str(model_path_obj),
            positive_dir=str(pos_dir),
            negative_dir=str(neg_dir),
            num_positive=len(pos_files),
            num_negative=len(neg_files),
            thresholds=metrics_list,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )

        # Print table
        table = format_table(metrics_list)
        print("\n" + table + "\n")

        # Save JSON
        if output_json:
            self._save_json(results, output_json)

        # Generate ROC curve
        if roc_image and (positive_scores or negative_scores):
            generate_roc_curve(positive_scores, negative_scores, Path(roc_image))

        return results

    def _load_model(self, model_path: str):
        """Load an openwakeword model.

        Args:
            model_path: path to an ONNX model file

        Returns:
            openwakeword Model instance
        """
        from openwakeword.model import Model as OWWModel

        model = OWWModel(
            wakeword_models=[model_path],
            inference_framework="onnx",
        )
        logger.info("Loaded model keys: %s", list(model.models.keys()))
        return model

    def _score_files(
        self,
        model,
        files: list[Path],
        label: str,
    ) -> list[float]:
        """Score a list of audio files.

        Args:
            model: openwakeword Model instance
            files: list of .wav file paths
            label: label string for logging ("positive" or "negative")

        Returns:
            list of peak scores, one per file
        """
        scores = []
        for i, filepath in enumerate(files):
            try:
                samples = load_wav(filepath)
                score = score_clip(model, samples)
                scores.append(score)
                if (i + 1) % 50 == 0 or (i + 1) == len(files):
                    logger.info(
                        "  Scored %d/%d %s clips", i + 1, len(files), label
                    )
            except Exception as e:
                logger.warning("Error scoring %s: %s", filepath.name, e)
        return scores

    def _save_json(self, results: BenchmarkResults, output_path: str) -> None:
        """Save results to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info("Results saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark a wake word model against labeled audio datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m train.benchmark --model models/stop.onnx \\
      --positive train/collected/positive/ \\
      --negative train/collected/negative/

  python -m train.benchmark --model models/stop.onnx \\
      --positive data/pos/ --negative data/neg/ \\
      --output results.json --roc roc.png \\
      --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the ONNX wake word model",
    )
    parser.add_argument(
        "--positive",
        type=str,
        required=True,
        help="Directory containing positive (wake word) .wav files",
    )
    parser.add_argument(
        "--negative",
        type=str,
        required=True,
        help="Directory containing negative .wav files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    parser.add_argument(
        "--roc",
        type=str,
        default=None,
        help="Path to save ROC curve image (optional, requires matplotlib)",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help=f"Sensitivity thresholds to evaluate (default: {DEFAULT_THRESHOLDS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="[benchmark] %(message)s",
    )

    args = parse_args(argv)

    benchmark = WakeWordBenchmark()
    benchmark.run(
        model_path=args.model,
        positive_dir=args.positive,
        negative_dir=args.negative,
        thresholds=args.thresholds,
        output_json=args.output,
        roc_image=args.roc,
    )


if __name__ == "__main__":
    main()
