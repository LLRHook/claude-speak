"""
Audio chimes — short programmatic tones for state-change feedback.
Uses numpy sine waves + sounddevice, no external audio files needed.
"""

import numpy as np


def _generate_tone(freq: float, duration: float, sample_rate: int = 24000) -> np.ndarray:
    """Generate a sine-wave tone with a fade-in/fade-out envelope to avoid clicks."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Quick fade-in / fade-out (5ms each) to eliminate pops
    fade_samples = int(sample_rate * 0.005)
    if fade_samples > 0 and len(tone) > 2 * fade_samples:
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)

    return tone


def _play(samples: np.ndarray, volume: float, device: int | None, sample_rate: int = 24000):
    """Play samples on the given device. Non-blocking fire-and-forget."""
    import sounddevice as sd

    samples = (samples * volume).astype(np.float32)
    try:
        sd.play(samples, samplerate=sample_rate, device=device, blocking=True)
    except Exception as e:
        print(f"[chimes] Playback error: {e}", flush=True)


def play_ready_chime(device: int | None = None, volume: float = 0.3):
    """Ascending two-note chime (C5 -> E5) — daemon is ready."""
    sr = 24000
    note1 = _generate_tone(523.25, 0.10, sr)  # C5
    gap = np.zeros(int(sr * 0.03), dtype=np.float32)
    note2 = _generate_tone(659.25, 0.15, sr)  # E5
    samples = np.concatenate([note1, gap, note2])
    _play(samples, volume, device, sr)


def play_error_chime(device: int | None = None, volume: float = 0.3):
    """Descending two-note chime (E5 -> C5) — something went wrong."""
    sr = 24000
    note1 = _generate_tone(659.25, 0.08, sr)  # E5
    gap = np.zeros(int(sr * 0.03), dtype=np.float32)
    note2 = _generate_tone(523.25, 0.12, sr)  # C5
    samples = np.concatenate([note1, gap, note2])
    _play(samples, volume, device, sr)


def play_stop_chime(device: int | None = None, volume: float = 0.3):
    """Single short low tone (C4) — playback stopped."""
    sr = 24000
    samples = _generate_tone(261.63, 0.10, sr)  # C4
    _play(samples, volume, device, sr)


def play_ack_chime(device: int | None = None, volume: float = 0.3):
    """Spoken 'Got it' — voice input received, processing."""
    import soundfile as sf
    from pathlib import Path
    ack_path = Path(__file__).resolve().parent / "assets" / "ack.wav"
    if not ack_path.exists():
        # Fallback to a quick tone if the asset is missing
        sr = 24000
        samples = _generate_tone(783.99, 0.08, sr)
        _play(samples, volume, device, sr)
        return
    samples, sr = sf.read(str(ack_path), dtype="float32")
    # Add 250ms silence tail so playback doesn't clip the end
    tail = np.zeros(int(sr * 0.25), dtype=np.float32)
    samples = np.concatenate([samples, tail])
    _play(samples, volume, device, sr)
