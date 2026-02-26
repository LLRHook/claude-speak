# Wake Word Training Pipeline

This directory contains the full training pipeline for custom openwakeword models used by claude-speak.

## Directory Structure

```
train/
  collect_data.py        # Record real-world audio clips (Phase 0)
  generate_clips.py      # Generate synthetic clips via Kokoro TTS (Phase 1)
  augment_clips.py       # Augment clips with noise/volume/speed (Phase 2)
  train_model.py         # Train DNN and export to ONNX (Phase 3)
  collected/             # Real-world recordings (from collect_data.py)
    positive/            #   Clips of the wake word being spoken
    negative/            #   Clips of non-wake-word speech, noise, silence
  positive/              # Synthetic positive clips (from generate_clips.py)
  negative/              # Synthetic negative clips (from generate_clips.py)
  augmented_positive/    # Augmented positive clips (from augment_clips.py)
  augmented_negative/    # Augmented negative clips (from augment_clips.py)
  features/              # Extracted openWakeWord embeddings (from train_model.py)
```

## Collecting Real-World Data

Real-world recordings significantly improve wake word accuracy compared to synthetic-only training data. The `collect_data.py` script provides an interactive CLI for recording labeled audio clips.

### Quick Start

```bash
# Interactive mode (prompts for all settings)
python train/collect_data.py

# Batch mode: record 10 positive + 10 negative clips for "stop"
python train/collect_data.py --word stop --batch 10

# Record only positive examples
python train/collect_data.py --word "hey jarvis" --label positive --count 20

# Batch with different positive/negative counts
python train/collect_data.py --word stop --batch 15 --negative-count 30
```

### Recording Workflow

1. The script prompts for the wake word phrase and number of clips
2. For each clip:
   - Press **Enter** to start recording
   - Speak the wake word (positive) or other speech/noise (negative)
   - Recording auto-stops after 3 seconds
3. Clips are saved as 16kHz mono WAV files in `train/collected/`

### Audio Format

All recordings are saved in openwakeword-compatible format:

| Property    | Value           |
|-------------|-----------------|
| Sample rate | 16,000 Hz       |
| Channels    | 1 (mono)        |
| Bit depth   | 16-bit (int16)  |
| Format      | WAV (PCM)       |
| Duration    | Up to 3 seconds |

### Filename Convention

Files follow a structured naming pattern:

```
{label}_{word}_{timestamp}_{index}.wav
```

- **label**: `positive` or `negative`
- **word**: sanitized wake word (lowercase, underscores for spaces)
- **timestamp**: `YYYYMMDD_HHMMSS` of the recording session
- **index**: zero-padded clip number within the session (e.g., `0000`, `0001`)

Examples:
```
positive_stop_20260226_143000_0000.wav
negative_stop_20260226_143000_0005.wav
positive_hey_jarvis_20260226_150000_0012.wav
```

### Recording Quality Guidelines

For best training results:

1. **Environment**: Record in a quiet room with minimal background noise. If possible, also record some clips with typical background noise (fan, office chatter) to improve robustness.

2. **Microphone distance**: Keep a consistent distance from the microphone (30-60 cm / 1-2 feet). This should match typical usage distance.

3. **Natural speech**: Speak the wake word naturally, as you would in everyday use. Vary your tone and speed slightly across recordings. Avoid whispering or shouting unless those are realistic use cases.

4. **Diversity**: Collect recordings from multiple speakers if possible (different voices, accents, genders). Record in different rooms and at different times of day.

5. **Negative examples**: Include a mix of:
   - Similar-sounding words (e.g., "stock", "step" for "stop")
   - Common speech phrases ("hello", "yes", "no", "thanks")
   - Background noise and silence
   - Typing, mouse clicks, and other desk sounds

6. **Volume**: Speak at a normal conversational volume. The script will reject clips that are pure silence.

7. **Quantity**: Aim for at least 20-30 positive and 50-100 negative examples per speaker. More is better.

### Contributing Recordings

To contribute recordings for improving shared wake word models:

1. Run the collection script and record clips as described above
2. Verify your recordings sound correct by playing them back
3. The collected files are in `train/collected/{positive,negative}/`
4. Compress the directory:
   ```bash
   cd train/collected
   tar -czf recordings_$(whoami)_$(date +%Y%m%d).tar.gz positive/ negative/
   ```
5. Share the archive via the project's issue tracker or file-sharing link

**Privacy note**: Recordings contain your voice audio. Only submit recordings you are comfortable sharing publicly. Do not include any sensitive or private conversations in negative examples.

## Full Training Pipeline

The complete pipeline for training a custom wake word model:

### Phase 0: Collect Real-World Data (Optional but Recommended)

```bash
python train/collect_data.py --word stop --batch 20
```

### Phase 1: Generate Synthetic Clips

```bash
python train/generate_clips.py
```

Generates positive and negative clips using Kokoro TTS across multiple voices and speeds. Requires Kokoro ONNX models in `models/`.

### Phase 2: Augment Clips

```bash
python train/augment_clips.py
```

Creates augmented versions of all clips with volume variation, background noise, speed jitter, and silence padding.

### Phase 3: Train Model

```bash
python train/train_model.py
```

Extracts openWakeWord features, trains a small DNN classifier, and exports to ONNX format at `models/stop.onnx`.

## Requirements

Core recording dependencies (already included in the project):
- `sounddevice` (audio recording)
- `numpy` (array operations)

Training-specific dependencies:
```bash
pip install -e ".[train,wakeword]"
```
