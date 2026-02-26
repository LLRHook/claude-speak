#!/usr/bin/env python3
"""
Real-world training data collection for wake word models.

Interactive CLI that records labeled audio clips from real speakers and
environments. Produces 16kHz mono WAV files compatible with openwakeword
training pipelines.

Usage:
    python train/collect_data.py                    # interactive mode
    python train/collect_data.py --batch 10         # record 10 positive + 10 negative
    python train/collect_data.py --word "hey jarvis" --label positive --count 5

Output: train/collected/{positive,negative}/*.wav
"""

import argparse
import re
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000       # openwakeword expects 16 kHz
CHANNELS = 1              # mono
DTYPE = "int16"           # 16-bit PCM
MAX_DURATION = 3.0        # auto-stop after 3 seconds
MIN_DURATION = 0.3        # discard clips shorter than 300 ms
VALID_LABELS = ("positive", "negative")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = Path(__file__).resolve().parent
COLLECTED_DIR = TRAIN_DIR / "collected"
POS_DIR = COLLECTED_DIR / "positive"
NEG_DIR = COLLECTED_DIR / "negative"


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def sanitize_word(word: str) -> str:
    """Sanitize a wake word phrase for use in filenames.

    Lowercases, replaces spaces/special characters with underscores,
    strips leading/trailing underscores.
    """
    word = word.lower().strip()
    word = re.sub(r"[^a-z0-9]+", "_", word)
    word = word.strip("_")
    return word or "unknown"


def generate_filename(label: str, word: str, index: int, timestamp: str | None = None) -> str:
    """Generate a structured filename for a recorded clip.

    Format: {label}_{word}_{timestamp}_{index}.wav

    Args:
        label: "positive" or "negative"
        word: sanitized wake word phrase
        index: clip index number
        timestamp: ISO-ish timestamp string; generated if None

    Returns:
        Filename string (not a full path).

    Raises:
        ValueError: if label is not "positive" or "negative"
    """
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label '{label}'. Must be one of: {VALID_LABELS}")

    sanitized = sanitize_word(word)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{label}_{sanitized}_{timestamp}_{index:04d}.wav"


def get_output_dir(label: str) -> Path:
    """Return the output directory for a given label."""
    if label == "positive":
        return POS_DIR
    elif label == "negative":
        return NEG_DIR
    else:
        raise ValueError(f"Invalid label '{label}'. Must be one of: {VALID_LABELS}")


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_clip(duration: float = MAX_DURATION, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record a single audio clip.

    Records for up to ``duration`` seconds at the given sample rate.

    Args:
        duration: maximum recording duration in seconds
        sample_rate: sample rate in Hz (default 16000)

    Returns:
        numpy array of int16 samples, shape (N,)

    Raises:
        RuntimeError: if sounddevice is not available
    """
    if sd is None:
        raise RuntimeError(
            "sounddevice is not installed. Install with: pip install sounddevice"
        )

    frames = int(duration * sample_rate)
    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype=DTYPE,
    )
    sd.wait()

    # Flatten from (N, 1) to (N,)
    return recording.flatten()


def save_wav(filepath: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Save audio samples to a WAV file (16kHz mono int16).

    Args:
        filepath: path to write the WAV file
        samples: int16 audio samples, shape (N,)
        sample_rate: sample rate in Hz
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def validate_clip(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    """Check that a recorded clip meets minimum quality requirements.

    Returns True if the clip is long enough and not pure silence.
    """
    duration = len(samples) / sample_rate
    if duration < MIN_DURATION:
        return False

    # Check it's not pure silence (RMS > very low threshold)
    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    return bool(rms >= 10)  # ~10 out of 32767 for int16


# ---------------------------------------------------------------------------
# Batch configuration
# ---------------------------------------------------------------------------

class BatchConfig:
    """Configuration for a batch recording session.

    Attributes:
        word: the wake word phrase
        positive_count: number of positive examples to record
        negative_count: number of negative examples to record
    """

    def __init__(self, word: str, positive_count: int = 10, negative_count: int = 10):
        if positive_count < 0:
            raise ValueError("positive_count must be >= 0")
        if negative_count < 0:
            raise ValueError("negative_count must be >= 0")
        if not word or not word.strip():
            raise ValueError("word must not be empty")

        self.word = word.strip()
        self.positive_count = positive_count
        self.negative_count = negative_count

    @property
    def total_count(self) -> int:
        return self.positive_count + self.negative_count


# ---------------------------------------------------------------------------
# Interactive recording session
# ---------------------------------------------------------------------------

def run_batch_session(config: BatchConfig) -> dict:
    """Run a batch recording session.

    Records positive_count positive examples followed by negative_count
    negative examples, with progress indicators.

    Args:
        config: BatchConfig with word, counts

    Returns:
        dict with keys "positive" and "negative", each an int count of
        successfully saved clips.
    """
    results = {"positive": 0, "negative": 0}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label, count in [("positive", config.positive_count), ("negative", config.negative_count)]:
        if count == 0:
            continue

        out_dir = get_output_dir(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        if label == "positive":
            print(f"\n--- Recording {count} POSITIVE examples ---")
            print(f"    Say: \"{config.word}\"")
        else:
            print(f"\n--- Recording {count} NEGATIVE examples ---")
            print(f"    Say anything OTHER than \"{config.word}\" (random speech, noise, silence)")

        print(f"    Each clip auto-stops after {MAX_DURATION}s")
        print()

        for i in range(count):
            progress = f"[{i + 1}/{count}]"
            input(f"  {progress} Press Enter to start recording...")

            print(f"  {progress} Recording... (up to {MAX_DURATION}s)")
            try:
                samples = record_clip()
            except Exception as e:
                print(f"  {progress} Error recording: {e}")
                continue

            if not validate_clip(samples):
                print(f"  {progress} Clip too short or silent, skipping.")
                continue

            filename = generate_filename(label, config.word, i, timestamp)
            filepath = out_dir / filename
            save_wav(filepath, samples)
            results[label] += 1

            duration = len(samples) / SAMPLE_RATE
            print(f"  {progress} Saved: {filename} ({duration:.1f}s)")

    return results


def interactive_mode() -> None:
    """Run the interactive data collection CLI."""
    print("=" * 60)
    print("  Wake Word Training Data Collection")
    print("=" * 60)
    print()

    word = input("Enter wake word phrase (e.g., 'hey jarvis', 'stop'): ").strip()
    if not word:
        print("Error: wake word cannot be empty.")
        sys.exit(1)

    try:
        pos_count = int(input("Number of POSITIVE examples to record [10]: ").strip() or "10")
    except ValueError:
        pos_count = 10

    try:
        neg_count = int(input("Number of NEGATIVE examples to record [10]: ").strip() or "10")
    except ValueError:
        neg_count = 10

    config = BatchConfig(word=word, positive_count=pos_count, negative_count=neg_count)

    print(f"\nPlan: Record {config.positive_count} positive + {config.negative_count} negative clips")
    print(f"Wake word: \"{config.word}\"")
    print(f"Output: {COLLECTED_DIR}/")
    print(f"Format: 16kHz mono WAV, up to {MAX_DURATION}s each")
    print()
    input("Press Enter to begin...")

    results = run_batch_session(config)

    print()
    print("=" * 60)
    print("  Session complete!")
    print(f"  Positive clips saved: {results['positive']}")
    print(f"  Negative clips saved: {results['negative']}")
    print(f"  Output directory: {COLLECTED_DIR}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect real-world audio clips for wake word training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python train/collect_data.py                              # interactive mode
  python train/collect_data.py --batch 10                   # 10 positive + 10 negative
  python train/collect_data.py --word stop --label positive --count 5
  python train/collect_data.py --word "hey jarvis" --batch 20 --negative-count 30
""",
    )
    parser.add_argument(
        "--word", type=str, default=None,
        help="Wake word phrase (e.g., 'stop', 'hey jarvis')",
    )
    parser.add_argument(
        "--label", type=str, choices=VALID_LABELS, default=None,
        help="Record only this label type (positive or negative)",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help="Number of clips to record (used with --label)",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch mode: record N positive + N negative clips",
    )
    parser.add_argument(
        "--negative-count", type=int, default=None,
        help="Override negative count in batch mode (default: same as --batch)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)

    if sd is None:
        print("Error: sounddevice is not installed.")
        print("Install with: pip install sounddevice")
        sys.exit(1)

    # If no arguments provided, run interactive mode
    if args.word is None and args.batch is None and args.label is None:
        interactive_mode()
        return

    # Need a word for non-interactive modes
    if args.word is None:
        print("Error: --word is required in non-interactive mode.")
        sys.exit(1)

    # Single-label mode
    if args.label is not None:
        count = args.count or 10
        if args.label == "positive":
            config = BatchConfig(word=args.word, positive_count=count, negative_count=0)
        else:
            config = BatchConfig(word=args.word, positive_count=0, negative_count=count)
    # Batch mode
    elif args.batch is not None:
        neg_count = args.negative_count if args.negative_count is not None else args.batch
        config = BatchConfig(word=args.word, positive_count=args.batch, negative_count=neg_count)
    else:
        # Word provided but no mode — default to batch of 10
        config = BatchConfig(word=args.word, positive_count=10, negative_count=10)

    results = run_batch_session(config)

    print(f"\nDone! Positive: {results['positive']}, Negative: {results['negative']}")
    print(f"Output: {COLLECTED_DIR}")


if __name__ == "__main__":
    main()
