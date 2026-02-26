#!/usr/bin/env python3
"""
Phase 3: Train a custom openWakeWord model for "stop" detection.

Pipeline:
  1. Load augmented WAV clips (16kHz, float32 -> int16)
  2. Pad/trim to uniform 2-second windows (wake word at END)
  3. Extract features via openWakeWord's frozen melspec + embedding models
  4. Train a small DNN classifier
  5. Export to ONNX

Output: models/stop.onnx
"""

import copy
import os
import sys
import logging
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TRAIN_DIR = Path(__file__).resolve().parent
AUG_POS_DIR = TRAIN_DIR / "augmented_positive"
AUG_NEG_DIR = TRAIN_DIR / "augmented_negative"
FEATURES_DIR = TRAIN_DIR / "features"
OUTPUT_DIR = PROJECT_ROOT / "models"

TOTAL_LENGTH = 32000  # 2 seconds at 16kHz — minimum for openWakeWord
TARGET_SR = 16000

logging.basicConfig(level=logging.INFO, format="[train] %(message)s")
log = logging.getLogger(__name__)


def load_and_pad(wav_path: Path, total_length: int = TOTAL_LENGTH) -> np.ndarray:
    """Load a WAV file and left-pad/right-trim to uniform length.

    Wake word must be at the END of the window (left-pad with silence).
    Returns int16 array as required by openWakeWord's feature extractor.
    """
    samples, sr = sf.read(str(wav_path), dtype="float32")

    # Resample if not 16kHz (shouldn't happen, but be safe)
    if sr != TARGET_SR:
        n_new = int(len(samples) * TARGET_SR / sr)
        indices = np.linspace(0, len(samples) - 1, n_new)
        samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    # Convert float32 [-1, 1] to int16
    samples_int16 = (samples * 32767).astype(np.int16)

    if len(samples_int16) >= total_length:
        # Keep the END (where the wake word is)
        samples_int16 = samples_int16[-total_length:]
    else:
        # Left-pad with silence
        pad = np.zeros(total_length - len(samples_int16), dtype=np.int16)
        samples_int16 = np.concatenate([pad, samples_int16])

    return samples_int16


def extract_features(clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Extract openWakeWord embeddings from int16 audio clips.

    Args:
        clips: shape (N, 32000) int16 audio
        batch_size: batch size for feature extraction

    Returns:
        features: shape (N, 16, 96) float32 embeddings
    """
    from openwakeword.utils import AudioFeatures

    log.info("Initializing feature extractor...")
    F = AudioFeatures(device="cpu")

    log.info(f"Extracting features from {clips.shape[0]} clips (batch_size={batch_size})...")
    features = F.embed_clips(clips, batch_size=batch_size, ncpu=4)
    log.info(f"Features shape: {features.shape}")
    return features


def load_clips_from_dir(directory: Path, max_clips: int = 0) -> np.ndarray:
    """Load all WAV files from a directory, pad to uniform length."""
    wav_files = sorted(directory.glob("*.wav"))
    if max_clips > 0:
        wav_files = wav_files[:max_clips]

    log.info(f"Loading {len(wav_files)} clips from {directory.name}/...")
    clips = []
    for i, wav_path in enumerate(tqdm(wav_files, desc=f"Loading {directory.name}")):
        try:
            clip = load_and_pad(wav_path)
            clips.append(clip)
        except Exception as e:
            log.warning(f"Error loading {wav_path.name}: {e}")

        if (i + 1) % 5000 == 0:
            log.info(f"  Loaded {i + 1}/{len(wav_files)}")

    return np.array(clips, dtype=np.int16)


def main():
    log.info("=== Phase 3: Training custom wake word model ===\n")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for pre-computed features
    pos_feat_path = FEATURES_DIR / "positive_features.npy"
    neg_feat_path = FEATURES_DIR / "negative_features.npy"

    if pos_feat_path.exists() and neg_feat_path.exists():
        log.info("Loading pre-computed features...")
        pos_features = np.load(str(pos_feat_path))
        neg_features = np.load(str(neg_feat_path))
        log.info(f"Positive features: {pos_features.shape}")
        log.info(f"Negative features: {neg_features.shape}")
    else:
        # Step 1: Load clips
        if not AUG_POS_DIR.exists() or not list(AUG_POS_DIR.glob("*.wav")):
            log.error(f"No augmented positive clips in {AUG_POS_DIR}")
            log.error("Run augment_clips.py first (Phase 2).")
            sys.exit(1)

        pos_clips = load_clips_from_dir(AUG_POS_DIR)
        neg_clips = load_clips_from_dir(AUG_NEG_DIR)

        log.info(f"\nLoaded {pos_clips.shape[0]} positive, {neg_clips.shape[0]} negative clips")
        log.info(f"Clip shape: {pos_clips.shape[1]} samples ({pos_clips.shape[1]/TARGET_SR:.1f}s)\n")

        # Step 2: Extract features
        pos_features = extract_features(pos_clips, batch_size=64)
        neg_features = extract_features(neg_clips, batch_size=64)

        # Save features for reuse
        log.info("Saving features to disk...")
        np.save(str(pos_feat_path), pos_features)
        np.save(str(neg_feat_path), neg_features)
        log.info(f"Saved to {FEATURES_DIR}/\n")

    # Step 3: Prepare training data
    log.info("Preparing training data...")
    n_pos = pos_features.shape[0]
    n_neg = neg_features.shape[0]

    # Create labels: 1 = positive (wake word), 0 = negative
    X = np.concatenate([pos_features, neg_features], axis=0)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split: 90% train, 10% val
    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    log.info(f"Train: {X_train.shape[0]} samples ({(y_train == 1).sum()} pos, {(y_train == 0).sum()} neg)")
    log.info(f"Val:   {X_val.shape[0]} samples ({(y_val == 1).sum()} pos, {(y_val == 0).sum()} neg)\n")

    # Create DataLoaders
    batch_size = 512
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 4: Build model (same architecture as openWakeWord's DNN)
    log.info("Building model...")

    input_shape = (pos_features.shape[1], pos_features.shape[2])  # (16, 96)
    layer_dim = 128

    class FCNBlock(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = torch.nn.Linear(dim, dim)
            self.norm = torch.nn.LayerNorm(dim)
            self.relu = torch.nn.ReLU()
        def forward(self, x):
            return self.relu(self.norm(self.linear(x)))

    class WakeWordNet(torch.nn.Module):
        def __init__(self, input_shape, layer_dim, n_blocks=1):
            super().__init__()
            self.flatten = torch.nn.Flatten()
            self.input_layer = torch.nn.Linear(input_shape[0] * input_shape[1], layer_dim)
            self.norm1 = torch.nn.LayerNorm(layer_dim)
            self.relu1 = torch.nn.ReLU()
            self.blocks = torch.nn.ModuleList([FCNBlock(layer_dim) for _ in range(n_blocks)])
            self.output_layer = torch.nn.Linear(layer_dim, 1)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            x = self.relu1(self.norm1(self.input_layer(self.flatten(x))))
            for block in self.blocks:
                x = block(x)
            return self.sigmoid(self.output_layer(x))

    model = WakeWordNet(input_shape, layer_dim, n_blocks=1)

    log.info(f"Model input shape: {input_shape}")
    log.info(f"Architecture: Flatten -> Linear({input_shape[0]*input_shape[1]}, {layer_dim}) -> "
             f"LayerNorm -> ReLU -> FCN({layer_dim}) -> Linear({layer_dim}, 1) -> Sigmoid")
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}\n")

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.functional.binary_cross_entropy

    n_epochs = 20
    best_val_acc = 0.0
    best_model_state = None
    max_neg_weight = 500  # heavy penalty for false positives (our neg data is thin)

    log.info(f"Training for {n_epochs} epochs, batch_size={batch_size}, max_neg_weight={max_neg_weight}")
    log.info("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tp, epoch_fp, epoch_tn, epoch_fn = 0, 0, 0, 0

        # Linearly increase negative weight over epochs
        neg_weight = 1.0 + (max_neg_weight - 1.0) * (epoch / max(n_epochs - 1, 1))

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x).squeeze()

            # Hard-loss mining: focus on difficult samples
            neg_mask = (batch_y == 0) & (preds >= 0.001)
            pos_mask = (batch_y == 1) & (preds < 0.999)
            mask = neg_mask | pos_mask

            if mask.sum() < 2:
                continue

            preds_hard = preds[mask]
            y_hard = batch_y[mask]

            # Weighted loss: penalize false positives more
            weights = torch.ones_like(y_hard) * neg_weight
            weights[y_hard == 1] = 1.0

            loss = loss_fn(preds_hard, y_hard, weight=weights)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Track metrics
            with torch.no_grad():
                pred_binary = (preds >= 0.5).float()
                epoch_tp += ((pred_binary == 1) & (batch_y == 1)).sum().item()
                epoch_fp += ((pred_binary == 1) & (batch_y == 0)).sum().item()
                epoch_tn += ((pred_binary == 0) & (batch_y == 0)).sum().item()
                epoch_fn += ((pred_binary == 0) & (batch_y == 1)).sum().item()

        # Validation
        model.eval()
        val_tp, val_fp, val_tn, val_fn = 0, 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).squeeze()
                pred_binary = (preds >= 0.5).float()
                val_tp += ((pred_binary == 1) & (batch_y == 1)).sum().item()
                val_fp += ((pred_binary == 1) & (batch_y == 0)).sum().item()
                val_tn += ((pred_binary == 0) & (batch_y == 0)).sum().item()
                val_fn += ((pred_binary == 0) & (batch_y == 1)).sum().item()

        val_total = val_tp + val_fp + val_tn + val_fn
        val_acc = (val_tp + val_tn) / max(val_total, 1)
        val_recall = val_tp / max(val_tp + val_fn, 1)
        val_precision = val_tp / max(val_tp + val_fp, 1)
        val_fpr = val_fp / max(val_fp + val_tn, 1)

        train_recall = epoch_tp / max(epoch_tp + epoch_fn, 1)

        log.info(
            f"Epoch {epoch+1:2d}/{n_epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"train_recall={train_recall:.3f} | "
            f"val_acc={val_acc:.3f} val_recall={val_recall:.3f} "
            f"val_prec={val_precision:.3f} val_FPR={val_fpr:.4f} "
            f"(neg_w={neg_weight:.0f})"
        )

        # Save best model by val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())

    log.info("-" * 60)
    log.info(f"Best val accuracy: {best_val_acc:.4f}\n")

    # Step 5: Export to ONNX
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    model.to("cpu")

    onnx_path = OUTPUT_DIR / "stop.onnx"
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    log.info(f"Exported model to {onnx_path}")
    log.info(f"Model size: {onnx_path.stat().st_size / 1024:.1f} KB")

    # Quick sanity check with ONNX runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, *input_shape).astype(np.float32)
    result = session.run(None, {"input": test_input})
    log.info(f"ONNX sanity check — output shape: {result[0].shape}, value: {result[0][0][0]:.4f}")

    log.info("\n=== Phase 3 complete! ===")
    log.info(f"Model saved to: {onnx_path}")
    log.info("Next: Add 'stop.onnx' to wakeword config and test detection.")


if __name__ == "__main__":
    main()
