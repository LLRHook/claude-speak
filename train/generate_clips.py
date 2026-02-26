#!/usr/bin/env python3
"""
Phase 1: Generate synthetic wake word training clips using Kokoro TTS.

Generates positive clips ("stop", "quiet") and negative clips (similar-sounding
words, filler phrases) across all available Kokoro voices and speed variations.

Output: train/positive/*.wav, train/negative/*.wav
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = Path(__file__).resolve().parent
POS_DIR = OUTPUT_DIR / "positive"
NEG_DIR = OUTPUT_DIR / "negative"

# Positive phrases — what we want to detect
POSITIVE_PHRASES = [
    "stop",
    "Stop.",
    "stop!",
    "Stop!",
    "stop stop",
    "stop it",
    "please stop",
    "quiet",
    "Quiet.",
    "quiet!",
    "be quiet",
]

# Negative phrases — similar-sounding words the model should NOT trigger on
NEGATIVE_PHRASES = [
    "start", "step", "top", "stock", "stoke", "stuff", "shop", "drop",
    "pop", "slot", "snap", "spot", "stomp", "stump", "strap", "strip",
    "stuck", "stir", "stem", "stone", "store", "storm", "story",
    "quite", "quick", "quit", "quote", "quest",
    "hello", "yes", "no", "okay", "thanks", "please",
    "the", "and", "but", "for", "not", "you", "all", "can",
    "her", "was", "one", "our", "out", "are", "has", "his",
    "run the tests", "check the logs", "open the file",
    "what is this", "how does it work", "tell me more",
    "hey jarvis", "computer", "listen",
]

# Speed variations
SPEEDS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]

# Only use English voices (prefix a=American, b=British, e=? — skip zh/ja/etc)
ENGLISH_PREFIXES = ("af_", "am_", "bf_", "bm_", "ef_", "em_")


def generate_clips():
    from kokoro_onnx import Kokoro

    print(f"Loading Kokoro model...", flush=True)
    kokoro = Kokoro(
        str(MODELS_DIR / "kokoro-v1.0.onnx"),
        str(MODELS_DIR / "voices-v1.0.bin"),
    )

    all_voices = sorted(kokoro.get_voices())
    # Filter to English voices for training data quality
    english_voices = [v for v in all_voices if any(v.startswith(p) for p in ENGLISH_PREFIXES)]
    print(f"Using {len(english_voices)} English voices out of {len(all_voices)} total", flush=True)

    POS_DIR.mkdir(parents=True, exist_ok=True)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    # Generate positive clips
    pos_count = 0
    total_pos = len(POSITIVE_PHRASES) * len(english_voices) * len(SPEEDS)
    print(f"\n--- Generating ~{total_pos} positive clips ---", flush=True)

    for phrase in POSITIVE_PHRASES:
        for voice in english_voices:
            for speed in SPEEDS:
                tag = phrase.lower().replace(" ", "_").replace(".", "").replace("!", "")
                filename = f"{tag}_{voice}_s{speed:.1f}.wav"
                filepath = POS_DIR / filename

                if filepath.exists():
                    pos_count += 1
                    continue

                try:
                    samples, sr = kokoro.create(
                        phrase, voice=voice, speed=speed, lang="en-us",
                    )
                    # Trim silence: find first/last sample above threshold
                    samples = _trim_silence(samples)
                    sf.write(str(filepath), samples, sr)
                    pos_count += 1
                    if pos_count % 100 == 0:
                        print(f"  Positive: {pos_count}/{total_pos}", flush=True)
                except Exception as e:
                    print(f"  Error ({voice}, {speed}x, '{phrase}'): {e}", flush=True)

    print(f"  Positive clips generated: {pos_count}", flush=True)

    # Generate negative clips (fewer speeds to save time — diversity matters more)
    neg_speeds = [0.8, 1.0, 1.2]
    neg_count = 0
    total_neg = len(NEGATIVE_PHRASES) * len(english_voices) * len(neg_speeds)
    print(f"\n--- Generating ~{total_neg} negative clips ---", flush=True)

    for phrase in NEGATIVE_PHRASES:
        for voice in english_voices:
            for speed in neg_speeds:
                tag = phrase.lower().replace(" ", "_").replace(".", "").replace("!", "")
                filename = f"{tag}_{voice}_s{speed:.1f}.wav"
                filepath = NEG_DIR / filename

                if filepath.exists():
                    neg_count += 1
                    continue

                try:
                    samples, sr = kokoro.create(
                        phrase, voice=voice, speed=speed, lang="en-us",
                    )
                    samples = _trim_silence(samples)
                    sf.write(str(filepath), samples, sr)
                    neg_count += 1
                    if neg_count % 200 == 0:
                        print(f"  Negative: {neg_count}/{total_neg}", flush=True)
                except Exception as e:
                    print(f"  Error ({voice}, {speed}x, '{phrase}'): {e}", flush=True)

    print(f"  Negative clips generated: {neg_count}", flush=True)
    print(f"\nDone! Positive: {pos_count}, Negative: {neg_count}", flush=True)
    print(f"Output: {POS_DIR}, {NEG_DIR}", flush=True)


def _trim_silence(samples: np.ndarray, threshold: float = 0.01, margin: int = 1600) -> np.ndarray:
    """Trim leading/trailing silence from audio, keeping a small margin."""
    abs_samples = np.abs(samples)
    above = np.where(abs_samples > threshold)[0]
    if len(above) == 0:
        return samples
    start = max(0, above[0] - margin)
    end = min(len(samples), above[-1] + margin)
    return samples[start:end]


if __name__ == "__main__":
    generate_clips()
