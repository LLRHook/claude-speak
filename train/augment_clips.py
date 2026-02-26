#!/usr/bin/env python3
"""
Phase 2: Augment training clips with real-world variations.

Takes the clean Kokoro-generated clips and creates augmented versions with:
  - Volume variation (0.3x to 2.0x)
  - Background noise (white, pink) at various SNR levels
  - Speed jitter (±5% resampling)
  - Random silence padding before/after

Output: train/augmented_positive/*.wav, train/augmented_negative/*.wav
"""

import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
POS_DIR = OUTPUT_DIR / "positive"
NEG_DIR = OUTPUT_DIR / "negative"
AUG_POS_DIR = OUTPUT_DIR / "augmented_positive"
AUG_NEG_DIR = OUTPUT_DIR / "augmented_negative"

# How many augmented versions to create per clean clip
AUGMENTATIONS_PER_CLIP = 3

TARGET_SR = 16000  # openwakeword expects 16kHz


def add_noise(samples: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise at a given signal-to-noise ratio."""
    rms_signal = np.sqrt(np.mean(samples ** 2))
    if rms_signal == 0:
        return samples
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, len(samples)).astype(np.float32)
    return samples + noise


def add_pink_noise(samples: np.ndarray, snr_db: float) -> np.ndarray:
    """Add pink (1/f) noise at a given SNR."""
    rms_signal = np.sqrt(np.mean(samples ** 2))
    if rms_signal == 0:
        return samples
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    # Generate pink noise via spectral shaping
    n = len(samples)
    white = np.random.normal(0, 1, n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # avoid division by zero
    fft *= 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n).astype(np.float32)
    pink = pink / (np.sqrt(np.mean(pink ** 2)) + 1e-10) * rms_noise
    return samples + pink


def change_volume(samples: np.ndarray, factor: float) -> np.ndarray:
    """Scale amplitude."""
    return np.clip(samples * factor, -1.0, 1.0).astype(np.float32)


def speed_jitter(samples: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Resample to simulate speed change, then resample back to original sr."""
    # Simple linear interpolation resample
    n_new = int(len(samples) / factor)
    if n_new == 0:
        return samples
    indices = np.linspace(0, len(samples) - 1, n_new)
    resampled = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)
    return resampled


def pad_silence(samples: np.ndarray, sr: int, max_pad_ms: int = 300) -> np.ndarray:
    """Add random silence padding before and after."""
    pre_ms = np.random.randint(0, max_pad_ms)
    post_ms = np.random.randint(0, max_pad_ms)
    pre_samples = np.zeros(int(sr * pre_ms / 1000), dtype=np.float32)
    post_samples = np.zeros(int(sr * post_ms / 1000), dtype=np.float32)
    return np.concatenate([pre_samples, samples, post_samples])


def resample_to_16k(samples: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 16kHz using linear interpolation."""
    if orig_sr == TARGET_SR:
        return samples
    n_new = int(len(samples) * TARGET_SR / orig_sr)
    if n_new == 0:
        return samples
    indices = np.linspace(0, len(samples) - 1, n_new)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def augment_clip(samples: np.ndarray, sr: int) -> np.ndarray:
    """Apply a random combination of augmentations."""
    rng = np.random

    # Resample to 16kHz first
    samples = resample_to_16k(samples, sr)
    sr = TARGET_SR

    # Volume variation (always apply)
    vol = rng.uniform(0.4, 1.8)
    samples = change_volume(samples, vol)

    # Speed jitter (80% chance)
    if rng.random() < 0.8:
        speed = rng.uniform(0.93, 1.07)
        samples = speed_jitter(samples, sr, speed)

    # Noise (70% chance)
    if rng.random() < 0.7:
        snr = rng.uniform(5, 30)  # 5dB (very noisy) to 30dB (clean)
        samples = add_noise(samples, snr) if rng.random() < 0.5 else add_pink_noise(samples, snr)

    # Silence padding (always apply)
    samples = pad_silence(samples, sr)

    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def process_directory(src_dir: Path, dst_dir: Path, augments_per_clip: int):
    """Augment all WAV files in src_dir, write to dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(src_dir.glob("*.wav"))
    if not wav_files:
        print(f"  No WAV files in {src_dir}", flush=True)
        return 0

    total = len(wav_files) * augments_per_clip
    count = 0

    for wav_path in wav_files:
        try:
            samples, sr = sf.read(wav_path, dtype="float32")
        except Exception as e:
            print(f"  Error reading {wav_path.name}: {e}", flush=True)
            continue

        # Also copy the original (resampled to 16kHz)
        orig_16k = resample_to_16k(samples, sr)
        orig_path = dst_dir / wav_path.name
        if not orig_path.exists():
            sf.write(str(orig_path), orig_16k, TARGET_SR)

        # Generate augmented versions
        stem = wav_path.stem
        for aug_idx in range(augments_per_clip):
            aug_name = f"{stem}_aug{aug_idx}.wav"
            aug_path = dst_dir / aug_name

            if aug_path.exists():
                count += 1
                continue

            augmented = augment_clip(samples, sr)
            sf.write(str(aug_path), augmented, TARGET_SR)
            count += 1

        if count % 500 == 0:
            print(f"  {count}/{total}", flush=True)

    return count


def main():
    print("=== Phase 2: Augmenting training clips ===\n", flush=True)

    # Check that Phase 1 output exists
    if not POS_DIR.exists() or not list(POS_DIR.glob("*.wav")):
        print(f"Error: No positive clips found in {POS_DIR}", flush=True)
        print("Run generate_clips.py first (Phase 1).", flush=True)
        sys.exit(1)

    pos_count = len(list(POS_DIR.glob("*.wav")))
    neg_count = len(list(NEG_DIR.glob("*.wav"))) if NEG_DIR.exists() else 0
    print(f"Input: {pos_count} positive, {neg_count} negative clips", flush=True)
    print(f"Augmentations per clip: {AUGMENTATIONS_PER_CLIP}", flush=True)
    print(f"Expected output: ~{pos_count * (AUGMENTATIONS_PER_CLIP + 1)} positive, "
          f"~{neg_count * (AUGMENTATIONS_PER_CLIP + 1)} negative\n", flush=True)

    print("--- Augmenting positive clips ---", flush=True)
    aug_pos = process_directory(POS_DIR, AUG_POS_DIR, AUGMENTATIONS_PER_CLIP)
    print(f"  Done: {aug_pos} augmented positive clips\n", flush=True)

    print("--- Augmenting negative clips ---", flush=True)
    aug_neg = process_directory(NEG_DIR, AUG_NEG_DIR, AUGMENTATIONS_PER_CLIP)
    print(f"  Done: {aug_neg} augmented negative clips\n", flush=True)

    # Summary
    final_pos = len(list(AUG_POS_DIR.glob("*.wav")))
    final_neg = len(list(AUG_NEG_DIR.glob("*.wav")))
    print("=== Complete ===", flush=True)
    print(f"Augmented positive: {final_pos} clips in {AUG_POS_DIR}", flush=True)
    print(f"Augmented negative: {final_neg} clips in {AUG_NEG_DIR}", flush=True)


if __name__ == "__main__":
    main()
