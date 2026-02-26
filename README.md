# claude-speak

**Hands-free voice interface for Claude Code.**

Talk to Claude and hear it talk back -- entirely local, no cloud APIs, no subscriptions (aside from Superwhisper for voice input).

- **Reads Claude's responses aloud** using local TTS via [Kokoro](https://github.com/thewh1teagle/kokoro-onnx)
- **Wake word activation** -- say "Hey Jarvis" to start voice input
- **Voice input via Superwhisper** with automatic transcription and submission
- **Smart text normalization** -- 24-step pipeline handles code blocks, tables, URLs, currency, math, dates, units, abbreviations, and more
- **Zero cloud dependency** -- everything runs on your machine
- **Low latency** -- parallel chunk generation, SIGUSR1 instant queue notification, small chunks for fast time-to-first-audio

---

## How It Works

```
Claude Code --> Hook (PostToolUse/Stop) --> Queue (/tmp/) --> Daemon --> Normalizer --> Kokoro TTS --> Audio
                                                                           ^
Wake Word (openwakeword) --> Voice Controller --> Superwhisper --> Transcription --> Auto-submit to Claude
```

**Hook layer** (`hooks/speak-response.sh`): Claude Code fires hooks after every tool use and when it finishes responding. The hook script reads new lines from the transcript, strips markdown, writes the text to a file-based queue in `/tmp/claude-speak-queue/`, and sends SIGUSR1 to the daemon for instant pickup.

**Daemon** (`src/daemon.py`): A persistent background process that loads the Kokoro TTS model once at startup, then watches the queue directory. When a new file appears, it normalizes the text, chunks it into small pieces (default 150 chars), and streams audio. For multi-chunk responses, the next chunk is generated while the current one plays -- eliminating dead air between sentences.

**Normalizer** (`src/normalizer.py`): A 24-step regex pipeline that transforms Claude's markdown-heavy, code-heavy responses into natural spoken prose. Code blocks become brief descriptions ("Here is a Python code snippet"), tables become narrated prose, URLs become "a github link", `$99.99` becomes "99 dollars and 99 cents", and so on.

**TTS Engine** (`src/tts.py`): Wraps Kokoro ONNX for streaming audio generation. Maintains a persistent output stream for gapless playback between chunks, with automatic device resolution and throttled re-checking.

**Wake Word Listener** (`src/wakeword.py`): Uses [openwakeword](https://github.com/dscripka/openWakeWord) to continuously listen for "Hey Jarvis" via the microphone. Runs in a background thread with cooldown to prevent rapid-fire triggers.

**Voice Controller** (`src/voice_controller.py`): Orchestrates the full voice input cycle. On wake word detection, it pauses the wake word listener (to release the mic), triggers Superwhisper to record, monitors for speech-then-silence, stops recording, waits for transcription, and auto-submits to Claude Code.

**Voice Input** (`src/voice_input.py`): Handles Superwhisper interaction via macOS AppleScript. Sends keyboard shortcuts to start/stop recording, monitors the clipboard for transcription output, and presses Enter to submit.

**Queue** (`src/queue.py`): File-based FIFO queue using timestamped `.txt` files in `/tmp/claude-speak-queue/`. The hook script writes, the daemon reads. Simple, reliable, no IPC dependencies.

**Chimes** (`src/chimes.py`): Programmatically generated sine-wave tones for state feedback -- ascending chime on startup, descending on error, single low tone on stop. No external audio files needed.

---

## Prerequisites

- **macOS** (required -- uses `osascript`, `pbpaste`, and System Events for key simulation)
- **Python 3.10+**
- **Claude Code CLI** (the hooks integrate directly with Claude Code's hook system)
- **Superwhisper** (optional, for voice input) -- a paid macOS app for local speech-to-text. Only needed if you want to talk to Claude, not just listen.
- **Microphone access** (for wake word detection) -- macOS will prompt for permission on first use

---

## Quick Start

```bash
git clone https://github.com/your-username/claude-speak.git
cd claude-speak
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the Kokoro TTS model files into the `models/` directory:

```
models/
  kokoro-v1.0.onnx    (325 MB)
  voices-v1.0.bin     (28 MB)
```

See the [Kokoro ONNX releases](https://github.com/thewh1teagle/kokoro-onnx/releases) for download links.

Enable voice output:

```bash
touch ~/.claude-speak-enabled
```

Test that audio works:

```bash
python cli.py say "Hello, this is claude-speak."
```

Start the daemon in the background:

```bash
python cli.py start
```

### Hook Setup

To have Claude Code automatically trigger speech, copy the hook configuration into your Claude Code settings. The `claude-hooks.json` file in the project root shows the required hook definitions:

- **SessionStart**: starts the daemon automatically
- **PostToolUse**: sends new assistant text to the queue after each tool use
- **Stop**: sends final assistant text when Claude finishes responding
- **SessionEnd**: stops the daemon when the session ends

Update the `REPO_DIR` placeholder in `claude-hooks.json` to match your installation path, then add the hooks to your Claude Code configuration (see `HOOKS-SETUP.md` for detailed instructions).

---

## Configuration

All configuration lives in `claude-speak.toml` at the project root. Copy `claude-speak.toml.example` to get started. Every setting has a sensible default -- you only need to override what you want to change.

The daemon hot-reloads the config file every 30 seconds, so most changes take effect without a restart.

Environment variables override TOML values: `CLAUDE_SPEAK_VOICE`, `CLAUDE_SPEAK_SPEED`, `CLAUDE_SPEAK_DEVICE`.

```toml
[tts]
# Voice ID for Kokoro TTS.
# Options include: af_sarah, af_bella, af_nicole, am_adam, am_michael, etc.
voice = "af_sarah"

# Playback speed multiplier. 1.0 = natural, 1.3 = conversational, 1.5 = fast.
speed = 1.3

# Audio output device.
#   "auto"    -- system default output device
#   "AirPods" -- substring match against device names
#   "3"       -- device index number
device = "auto"

# Maximum characters per TTS chunk. Smaller = faster first audio.
# Parallel generation of subsequent chunks eliminates gaps between them.
# Range: 50-500.
max_chunk_chars = 150

[wakeword]
# Enable wake word detection.
enabled = true

# Detection engine. Currently only "openwakeword" is supported.
engine = "openwakeword"

# Wake word model name or path to a custom .onnx file.
# Built-in options: "hey_jarvis", "hey_mycroft", "alexa", etc.
model = "hey_jarvis"

# Detection sensitivity (0.0 to 1.0).
# Lower = fewer false positives, higher = fewer missed detections.
sensitivity = 0.5

# Phrases that immediately stop TTS playback when detected.
stop_phrases = ["stop", "quiet", "shut up"]

[input]
# Use Superwhisper for voice-to-text. Requires the Superwhisper app.
superwhisper = true

# Automatically press Enter to submit transcribed text to Claude.
auto_submit = true

[audio]
# Play short chime sounds on state changes (ready, error, stop).
chimes = true

# Text spoken when the daemon starts. Set to "" to disable.
greeting = "Ready."

# Chime volume (0.0 to 1.0). Does not affect TTS voice volume.
volume = 0.3

[normalization]
# Replace code blocks with brief spoken descriptions.
skip_code = true

# Expand unit abbreviations: "5ms" -> "5 milliseconds".
expand_units = true

# Expand tech abbreviations: "API" -> "A P I", "CLI" -> "C L I".
expand_abbreviations = true

# Shorten long file paths to just the filename.
shorten_paths = true
```

---

## CLI Commands

All commands are invoked through `cli.py`:

| Command | Description |
|---------|-------------|
| `python cli.py start` | Start the daemon in the background |
| `python cli.py stop` | Stop the daemon gracefully |
| `python cli.py restart` | Kill all instances and start fresh |
| `python cli.py status` | Show daemon PID, uptime, queue depth, config summary |
| `python cli.py say "text"` | Speak text immediately (bypasses daemon, loads model inline) |
| `python cli.py queue "text"` | Normalize, chunk, and queue text for the daemon to speak |
| `python cli.py clear` | Clear the TTS queue |
| `python cli.py log` | Show the last 20 lines of `daemon.log` |
| `python cli.py config` | Print all current configuration values |
| `python cli.py voices` | List all available Kokoro voices |
| `python cli.py enable` | Enable voice output (creates `~/.claude-speak-enabled`) |
| `python cli.py disable` | Disable voice output (daemon stays loaded but silent) |
| `python cli.py listen` | Start the voice controller standalone (wake word + voice input) |
| `python cli.py voice-input` | Trigger a single voice input cycle manually |
| `python cli.py kill-all` | Nuclear option -- kill all claude-speak processes |

---

## Voice Commands

### Wake Word

Say **"Hey Jarvis"** to activate voice input. The system will:

1. Detect the wake word via openwakeword (running continuously in a background thread).
2. Pause the wake word listener to release the microphone.
3. Trigger Superwhisper to start recording (sends Option+Space).
4. Monitor the microphone for speech, then wait for 3 seconds of silence.
5. Trigger Superwhisper again to stop recording.
6. Wait for Superwhisper to transcribe and paste the text.
7. Press Enter to submit the transcription to Claude Code.
8. Resume the wake word listener.

### Stop Phrases

Say **"stop"**, **"quiet"**, or **"shut up"** to immediately:

- Abort current TTS playback
- Clear the entire TTS queue
- Play a short confirmation chime

Stop phrases are detected when the hook sends them through the queue. The daemon recognizes them and halts instead of speaking.

---

## Text Normalization

The normalizer is a 24-step pipeline that transforms Claude's raw markdown output into natural spoken text. Each step uses pre-compiled regex patterns for performance.

### Pipeline Order and Examples

| Step | Input | Output |
|------|-------|--------|
| Code blocks | ` ```python\ndef foo(): ... ``` ` | "Here is a Python code snippet." |
| Tables | `\| Name \| Type \| ...` | "A table with columns Name and Type. Row 1: ..." |
| Numbered lists | `1. Install\n2. Run` | "First, Install. Second, Run." |
| Bullet lists | `- Fast\n- Simple` | "Fast. Simple." |
| Raw code lines | `$ pip install foo` | *(removed)* |
| URLs | `https://github.com/user/repo` | "a github link" |
| Emails | `user@example.com` | "user at example dot com" |
| File paths | `/Users/victor/projects/src/tts.py` | "tts dot py" |
| Currency | `$99.99` | "99 dollars and 99 cents" |
| Currency (magnitude) | `$1.5M` | "1.5 million dollars" |
| Percentages | `75%` | "75 percent" |
| Ordinals | `1st`, `21st` | "first", "twenty first" |
| Number commas | `1,234,567` | "1234567" |
| Time (24h) | `14:30` | "2:30 PM" |
| Fractions | `3/4` | "three quarters" |
| Ratios | `16:9` | "16 to 9" |
| Dates (ISO) | `2024-02-26` | "February 26, 2024" |
| Temperature | `72°F` | "72 degrees Fahrenheit" |
| Math operators | `x != y` | "x not equals y" |
| Units | `27MB` | "27 megabytes" |
| Versions | `v1.0.3` | "version 1 point 0 point 3" |
| Decimals | `1.5` | "one point five" |
| File extensions | `script.py` | "script dot py" |
| Abbreviations | `API`, `CLI` | "A P I", "C L I" |
| Stop words | `e.g.`, `etc.` | "for example", "etcetera" |
| Slash pairs | `true/false` | "true or false" |
| Technical punctuation | backticks, arrows, braces, snake_case | cleaned/spoken form |

The pipeline handles over 50 tech abbreviations, 20+ unit types, 30+ file extensions, and 4 currency symbols with correct singular/plural forms.

---

## Performance

The system is designed for low latency from the moment Claude finishes writing to the moment audio starts playing.

- **SIGUSR1 instant notification**: The hook script signals the daemon immediately after writing to the queue. No polling delay -- the daemon wakes up within milliseconds.
- **Parallel chunk generation**: For multi-chunk responses, the next chunk is generated by Kokoro while the current one plays. This uses `asyncio.create_task` + `asyncio.to_thread` for true concurrency.
- **Small default chunk size** (150 chars): Kokoro generates audio for a short chunk in ~200-400ms, so the first audio starts quickly. Parallel generation ensures subsequent chunks are ready before the current one finishes.
- **Pre-compiled regex patterns**: All 30+ regex patterns in the normalizer are compiled at module import time, not per-call.
- **Persistent audio stream**: The TTS engine maintains a single `sounddevice.OutputStream` across chunks for gapless playback -- no stream creation overhead between sentences.
- **Device resolution throttling**: Audio device lookup happens at most once every 30 seconds, not per chunk.
- **Hot-reload config**: The daemon checks the config file mtime every 30 seconds and reloads without restart.
- **Performance logging**: Every stage is timed and logged with `[perf]` prefixes in `daemon.log` for easy profiling.

---

## Troubleshooting

**No audio output**
- Check that the daemon is running: `python cli.py status`
- Verify the toggle file exists: `ls ~/.claude-speak-enabled` (create it with `python cli.py enable`)
- Test audio directly: `python cli.py say "test"` -- this bypasses the daemon and loads the model inline
- Check the device setting in `claude-speak.toml` -- try `device = "auto"` first
- Ensure the Kokoro model files exist in `models/`

**Wake word not detecting**
- Grant microphone permission in System Settings > Privacy & Security > Microphone
- Check that `[wakeword] enabled = true` in the config
- Adjust `sensitivity` -- try 0.3 for fewer false positives or 0.7 for more responsive detection
- Run `python cli.py listen` to test wake word detection standalone

**Superwhisper not activating**
- Verify Superwhisper is running (check the menu bar icon)
- Confirm the keyboard shortcut is Option+Space (the default)
- Check that `[input] superwhisper = true` in the config
- Run `python cli.py voice-input` for a single test cycle

**Daemon not starting**
- Check the log: `python cli.py log`
- Look for lock file issues: `ls /tmp/claude-speak-daemon.lock`
- Nuclear option: `python cli.py kill-all` then `python cli.py start`

**High latency**
- Check `daemon.log` for `[perf]` lines to identify the bottleneck
- Reduce `max_chunk_chars` (try 100) for faster time-to-first-audio
- Ensure no other heavy processes are competing for CPU during TTS generation

---

## Known Limitations

- **macOS only** -- depends on `osascript`, `pbpaste`, and System Events for keyboard simulation
- **Kokoro TTS latency** -- first audio segment takes ~200-800ms to generate depending on chunk size and CPU load
- **Superwhisper is paid** -- a third-party macOS app required only for voice input (TTS output works without it)
- **Wake word models are pre-trained** -- limited to models bundled with openwakeword (`hey_jarvis`, `hey_mycroft`, `alexa`, etc.) or custom ONNX models
- **No streaming from Claude Code** -- hooks fire after tool completion or when Claude stops responding, not during generation. You hear the response after it is written, not as it streams.

---

## Project Structure

```
claude-speak/
  cli.py                    # CLI entry point
  claude-speak.toml         # Configuration (user-editable)
  claude-speak.toml.example # Annotated example config
  claude-hooks.json         # Claude Code hook definitions
  pyproject.toml            # Python package metadata
  requirements.txt          # Pip dependencies
  src/
    daemon.py               # Main daemon loop and lifecycle
    tts.py                  # Kokoro TTS engine wrapper
    normalizer.py           # 24-step text normalization pipeline
    wakeword.py             # Wake word detection (openwakeword)
    voice_input.py          # Superwhisper integration
    voice_controller.py     # Orchestrates wake word -> voice input
    config.py               # Configuration loading and defaults
    queue.py                # File-based FIFO queue
    chimes.py               # Programmatic audio feedback tones
  hooks/
    speak-response.sh       # PostToolUse/Stop hook -- queue new text
    daemon-start.sh         # SessionStart hook -- launch daemon
    daemon-stop.sh          # SessionEnd hook -- stop daemon
  models/
    kokoro-v1.0.onnx        # Kokoro TTS model (325 MB, not in git)
    voices-v1.0.bin         # Voice data (28 MB, not in git)
  tests/
    ...
```

---

## License

MIT
