"""
Custom wake word training pipeline.

Guides the user through recording positive/negative examples, augments the
data, extracts mel spectrogram features, trains a binary classifier, and
exports it to ONNX format for use with the wake word detector.

Usage (programmatic):
    from claude_speak.train_wakeword import train_wakeword
    train_wakeword("hey claude")

Usage (CLI):
    claude-speak train-wakeword "hey claude"
    claude-speak train-wakeword "hey claude" --samples 15
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
N_MELS = 40             # mel filter bank size
N_FFT = 512             # FFT window size
HOP_LENGTH = 160        # 10ms hop at 16 kHz
FEATURE_WINDOW_SEC = 1.5  # seconds of audio per feature window
DEFAULT_SAMPLES = 10
MAX_RECORD_DURATION = 3.0  # seconds per clip


# ---------------------------------------------------------------------------
# Audio augmentation (numpy-only, no extra dependencies)
# ---------------------------------------------------------------------------

def pitch_shift(samples: np.ndarray, semitones: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Shift pitch by resampling with linear interpolation.

    Shifts by ``semitones`` (positive = higher, negative = lower).
    Returns an array of the same length and dtype as ``samples``.
    """
    factor = 2.0 ** (semitones / 12.0)
    n = len(samples)
    # Resample: stretch/compress the time axis, then truncate/pad to original length
    indices = np.arange(n) * factor
    # Clamp indices to valid range
    indices = np.clip(indices, 0, n - 1)
    # Linear interpolation
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, n - 1)
    frac = indices - idx_floor
    result = samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac
    return result.astype(samples.dtype)


def add_noise(samples: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add Gaussian noise to audio samples.

    ``noise_level`` is relative to the maximum int16 amplitude.
    """
    noise = np.random.randn(len(samples)) * noise_level * 32767
    noisy = samples.astype(np.float64) + noise
    return np.clip(noisy, -32768, 32767).astype(samples.dtype)


def speed_variation(samples: np.ndarray, factor: float) -> np.ndarray:
    """Change playback speed by resampling with linear interpolation.

    ``factor`` > 1.0 = faster (shorter), < 1.0 = slower (longer).
    Output is padded/trimmed to match the original length.
    """
    n = len(samples)
    new_len = int(n / factor)
    if new_len < 2:
        return samples.copy()
    indices = np.linspace(0, n - 1, new_len)
    idx_floor = np.floor(indices).astype(np.int64)
    idx_ceil = np.minimum(idx_floor + 1, n - 1)
    frac = indices - idx_floor
    resampled = samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac
    resampled = resampled.astype(samples.dtype)
    # Pad or trim to original length
    if len(resampled) >= n:
        return resampled[:n]
    padded = np.zeros(n, dtype=samples.dtype)
    padded[: len(resampled)] = resampled
    return padded


def augment_clip(samples: np.ndarray) -> list[np.ndarray]:
    """Generate augmented variants of a single audio clip.

    Returns a list of augmented clips (does NOT include the original).
    """
    variants = []
    # Pitch shifts
    for st in [-2, -1, 1, 2]:
        variants.append(pitch_shift(samples, st))
    # Noise levels
    for nl in [0.003, 0.008, 0.015]:
        variants.append(add_noise(samples, nl))
    # Speed variations
    for spd in [0.9, 1.1]:
        variants.append(speed_variation(samples, spd))
    return variants


# ---------------------------------------------------------------------------
# Mel spectrogram feature extraction (numpy-only)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filter_bank(
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build a mel-scale filter bank matrix.

    Returns shape ``(n_mels, n_fft // 2 + 1)``.
    """
    n_freqs = n_fft // 2 + 1
    mel_low = _hz_to_mel(0)
    mel_high = _hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filters[i, j] = (right - j) / (right - center)
    return filters


# Module-level cache for filter bank (same params throughout)
_CACHED_FILTER_BANK: np.ndarray | None = None


def _get_filter_bank() -> np.ndarray:
    global _CACHED_FILTER_BANK
    if _CACHED_FILTER_BANK is None:
        _CACHED_FILTER_BANK = _mel_filter_bank()
    return _CACHED_FILTER_BANK


def extract_mel_features(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute a log-mel spectrogram from int16 audio samples.

    Returns shape ``(n_mels, n_frames)`` where n_frames depends on audio
    length and hop size.
    """
    # Convert to float
    audio = samples.astype(np.float32) / 32768.0

    # Pad to at least one full frame
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    # STFT with Hann window
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = np.abs(spectrum) ** 2  # power spectrum

    # Apply mel filter bank
    mel_bank = _get_filter_bank()
    mel_spec = mel_bank @ stft  # (n_mels, n_frames)

    # Log scale with floor to avoid log(0)
    mel_spec = np.log(np.maximum(mel_spec, 1e-10)).astype(np.float32)

    return mel_spec


def features_from_clip(
    samples: np.ndarray,
    window_sec: float = FEATURE_WINDOW_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract a fixed-size feature vector from a variable-length audio clip.

    The clip is trimmed or zero-padded to ``window_sec`` seconds, then a mel
    spectrogram is computed and flattened into a 1-D vector.
    """
    target_len = int(window_sec * sample_rate)
    if len(samples) >= target_len:
        clip = samples[:target_len]
    else:
        clip = np.zeros(target_len, dtype=samples.dtype)
        clip[: len(samples)] = samples

    mel = extract_mel_features(clip, sample_rate=sample_rate)
    return mel.flatten().astype(np.float32)


def compute_feature_dim() -> int:
    """Return the feature vector dimensionality for the default parameters."""
    target_len = int(FEATURE_WINDOW_SEC * SAMPLE_RATE)
    n_frames = 1 + (target_len - N_FFT) // HOP_LENGTH
    return N_MELS * n_frames


# ---------------------------------------------------------------------------
# Recording helpers (reuse collect_data utilities)
# ---------------------------------------------------------------------------

def _record_examples(
    label: str,
    wake_phrase: str,
    count: int,
    record_fn=None,
) -> list[np.ndarray]:
    """Guide the user through recording ``count`` examples.

    If ``record_fn`` is provided, it is called instead of the real
    ``record_clip`` function (useful for testing).

    Returns a list of valid numpy int16 arrays.
    """
    # Import collect_data utilities
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from collect_data import record_clip as real_record_clip
    from collect_data import validate_clip

    record = record_fn or real_record_clip

    clips: list[np.ndarray] = []
    if label == "positive":
        print(f'\n  Say "{wake_phrase}" when prompted.')
    else:
        print('\n  Say anything OTHER than the wake word (random speech, noise, silence).')

    max_attempts = count * 3
    attempt = 0
    while len(clips) < count and attempt < max_attempts:
        idx = len(clips) + 1
        input(f"  [{idx}/{count}] Press Enter to start recording...")
        print(f"  [{idx}/{count}] Recording... (up to {MAX_RECORD_DURATION}s)")
        attempt += 1

        try:
            samples = record(duration=MAX_RECORD_DURATION)
        except Exception as e:
            print(f"  [{idx}/{count}] Error: {e}")
            continue

        if not validate_clip(samples):
            print(f"  [{idx}/{count}] Clip too short or silent, try again.")
            continue

        clips.append(samples)
        duration = len(samples) / SAMPLE_RATE
        print(f"  [{idx}/{count}] Recorded ({duration:.1f}s)")

    return clips


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_classifier(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a logistic regression classifier using numpy.

    Uses L2-regularised logistic regression via gradient descent.
    Returns ``(weights, bias)`` where weights has shape ``(n_features,)``
    and bias is a scalar array of shape ``(1,)``.

    This avoids requiring scikit-learn as a hard dependency.
    """
    n_samples, n_features = X.shape

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    # Initialize
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0
    lr = 0.1
    reg = 0.01
    n_epochs = 200

    for epoch in range(n_epochs):
        logits = X_norm @ w + b
        # Stable sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

        # Gradient
        error = probs - y
        grad_w = (X_norm.T @ error) / n_samples + reg * w
        grad_b = error.mean()

        w -= lr * grad_w
        b -= lr * grad_b

        # Decay learning rate
        if (epoch + 1) % 50 == 0:
            lr *= 0.5

    # Fold standardization into weights: w_raw = w_norm / std, b_raw = b_norm - (w_norm / std) @ mean
    w_raw = w / std
    b_raw = b - (w / std) @ mean

    return w_raw.astype(np.float32), np.array([b_raw], dtype=np.float32)


def _evaluate(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> float:
    """Compute accuracy of the logistic regression model."""
    logits = X @ weights + bias[0]
    preds = (logits > 0).astype(int)
    return float(np.mean(preds == y))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    weights: np.ndarray,
    bias: np.ndarray,
    output_path: Path,
    feature_dim: int | None = None,
) -> Path:
    """Export a trained logistic regression model to ONNX format.

    The ONNX model takes an input tensor ``features`` of shape
    ``(1, feature_dim)`` and outputs ``probability`` of shape ``(1, 1)``.
    """
    import onnx
    from onnx import TensorProto, helper

    if feature_dim is None:
        feature_dim = len(weights)

    # Input: (1, feature_dim)
    X = helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, feature_dim])
    # Output: (1, 1)
    Y = helper.make_tensor_value_info("probability", TensorProto.FLOAT, [1, 1])

    # Weights as initializer: shape (feature_dim, 1)
    w_init = helper.make_tensor(
        "weights",
        TensorProto.FLOAT,
        [feature_dim, 1],
        weights.reshape(-1).tolist(),
    )
    # Bias as initializer: shape (1,)
    b_init = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [1],
        bias.reshape(-1).tolist(),
    )

    # MatMul: features @ weights -> logits
    matmul_node = helper.make_node("MatMul", ["features", "weights"], ["logits"])
    # Add bias
    add_node = helper.make_node("Add", ["logits", "bias"], ["logits_biased"])
    # Sigmoid -> probability
    sigmoid_node = helper.make_node("Sigmoid", ["logits_biased"], ["probability"])

    graph = helper.make_graph(
        [matmul_node, add_node, sigmoid_node],
        "wake_word_classifier",
        [X],
        [Y],
        initializer=[w_init, b_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# ONNX inference (for verification)
# ---------------------------------------------------------------------------

def predict_onnx(model_path: Path, features: np.ndarray) -> np.ndarray:
    """Run inference on a trained ONNX wake word model.

    ``features`` should have shape ``(1, feature_dim)`` or ``(feature_dim,)``.
    Returns the probability array.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path))
    if features.ndim == 1:
        features = features.reshape(1, -1)
    features = features.astype(np.float32)
    result = session.run(None, {"features": features})
    return result[0]


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def train_wakeword(
    wake_phrase: str,
    output_dir: Path | None = None,
    num_samples: int = DEFAULT_SAMPLES,
    record_fn=None,
) -> Path:
    """Run the full custom wake word training pipeline.

    Steps:
        1. Record positive examples
        2. Record negative examples
        3. Augment positive and negative recordings
        4. Extract mel spectrogram features
        5. Train a logistic regression classifier
        6. Export to ONNX and save

    Args:
        wake_phrase: The wake word or phrase to train (e.g. "hey claude").
        output_dir: Directory to save the model. Defaults to
            ``~/.claude-speak/models/``.
        num_samples: Number of positive/negative recordings each (default 10).
        record_fn: Optional recording function override (for testing).

    Returns:
        Path to the saved ONNX model file.
    """
    if output_dir is None:
        output_dir = Path.home() / ".claude-speak" / "models"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize the wake phrase for filename
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from collect_data import sanitize_word

    safe_name = sanitize_word(wake_phrase)
    model_filename = f"wakeword_{safe_name}.onnx"
    model_path = output_dir / model_filename

    print("=" * 60)
    print("  Custom Wake Word Training Pipeline")
    print("=" * 60)
    print(f'  Wake phrase:  "{wake_phrase}"')
    print(f"  Samples:      {num_samples} positive + {num_samples} negative")
    print(f"  Output:       {model_path}")
    print()

    t_start = time.time()

    # --- Step 1: Record positive examples ---
    print("Step 1/6: Record POSITIVE examples")
    positive_clips = _record_examples("positive", wake_phrase, num_samples, record_fn=record_fn)
    if len(positive_clips) < 3:
        print(f"  Error: Need at least 3 positive clips, got {len(positive_clips)}.")
        raise ValueError(f"Insufficient positive clips: {len(positive_clips)}")
    print(f"  Collected {len(positive_clips)} positive clips.\n")

    # --- Step 2: Record negative examples ---
    print("Step 2/6: Record NEGATIVE examples")
    negative_clips = _record_examples("negative", wake_phrase, num_samples, record_fn=record_fn)
    if len(negative_clips) < 3:
        print(f"  Error: Need at least 3 negative clips, got {len(negative_clips)}.")
        raise ValueError(f"Insufficient negative clips: {len(negative_clips)}")
    print(f"  Collected {len(negative_clips)} negative clips.\n")

    # --- Step 3: Augment recordings ---
    print("Step 3/6: Augmenting recordings...")
    aug_positive = []
    for clip in positive_clips:
        aug_positive.extend(augment_clip(clip))
    aug_negative = []
    for clip in negative_clips:
        aug_negative.extend(augment_clip(clip))

    all_positive = positive_clips + aug_positive
    all_negative = negative_clips + aug_negative
    print(f"  Positive: {len(positive_clips)} original + {len(aug_positive)} augmented = {len(all_positive)}")
    print(f"  Negative: {len(negative_clips)} original + {len(aug_negative)} augmented = {len(all_negative)}\n")

    # --- Step 4: Extract features ---
    print("Step 4/6: Extracting mel spectrogram features...")
    feature_dim = compute_feature_dim()
    n_total = len(all_positive) + len(all_negative)
    X = np.zeros((n_total, feature_dim), dtype=np.float32)
    y = np.zeros(n_total, dtype=np.float64)

    for i, clip in enumerate(all_positive):
        X[i] = features_from_clip(clip)
        y[i] = 1.0
    offset = len(all_positive)
    for i, clip in enumerate(all_negative):
        X[offset + i] = features_from_clip(clip)
        y[offset + i] = 0.0

    print(f"  Feature matrix: {X.shape} (samples x features)")
    print(f"  Labels: {int(y.sum())} positive, {int(len(y) - y.sum())} negative\n")

    # --- Step 5: Train classifier ---
    print("Step 5/6: Training classifier...")
    # Shuffle data
    perm = np.random.permutation(n_total)
    X = X[perm]
    y = y[perm]

    weights, bias = _train_classifier(X, y)
    accuracy = _evaluate(X, y, weights, bias)
    print(f"  Training accuracy: {accuracy:.1%}\n")

    # --- Step 6: Export to ONNX ---
    print("Step 6/6: Exporting to ONNX...")
    export_to_onnx(weights, bias, model_path, feature_dim=feature_dim)

    # Verify the exported model
    test_input = X[:1].astype(np.float32)
    prob = predict_onnx(model_path, test_input)
    print(f"  Model saved: {model_path}")
    print(f"  Verification: probability={prob.flatten()[0]:.4f} (label={int(y[0])})")

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Model: {model_path}")
    print(f"  Accuracy: {accuracy:.1%}")
    print("=" * 60)

    return model_path
