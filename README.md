# claude-speak

**Hands-free voice interface for [Claude Code](https://docs.anthropic.com/en/docs/claude-code).**

claude-speak reads Claude's responses aloud and lets you talk back using wake word detection and local speech-to-text. It hooks into Claude Code's event system so every response is automatically spoken, and you can reply by saying "Hey Jarvis" instead of typing. The entire TTS and STT pipeline runs locally on your Mac -- no cloud APIs, no subscriptions, no data leaving your machine.

The project supports three TTS engines (Kokoro, Piper, ElevenLabs), built-in speech-to-text via MLX Whisper on Apple Silicon, custom wake word training, voice blending, global hotkeys, media key integration, and a 24-step text normalization pipeline that turns Claude's markdown-heavy output into natural spoken prose.

<!-- Demo video will go here -->

---

## Quick Start

**1. Install**

```bash
pip install claude-speak
```

Or install from source:

```bash
git clone https://github.com/vnicivanov/claude-speak.git
cd claude-speak
pip install -e .
```

**2. Setup**

```bash
claude-speak setup
```

This downloads the Kokoro TTS model (~350 MB), installs Claude Code hooks, creates a default config, and verifies audio output. Everything is stored in `~/.claude-speak/`.

**3. Use**

Start a Claude Code session -- claude-speak launches automatically via hooks. Claude's responses are spoken aloud. Say "Hey Jarvis" to reply by voice (requires the `wakeword` extra).

To test manually:

```bash
claude-speak say "Hello from claude-speak."
```

---

## Features

| Feature | Description |
|---------|-------------|
| **TTS output** | Every Claude Code response is automatically spoken aloud via local TTS |
| **Multiple TTS engines** | Kokoro (default, local), Piper (local, lightweight), ElevenLabs (cloud) |
| **Voice blending** | Mix voices with weighted ratios, e.g. `"bm_george:60+bm_fable:40"` |
| **Wake word detection** | Say "Hey Jarvis" (or a custom wake word) to start voice input |
| **Built-in STT** | Local speech-to-text via MLX Whisper on Apple Silicon (no Superwhisper needed) |
| **Custom wake words** | Train your own wake word model from recorded samples |
| **Text normalization** | 24-step pipeline converts code blocks, URLs, currency, math, dates, units, and more into natural speech |
| **SSML-like markup** | Control speech with `<pause>`, `<slow>`, `<fast>`, and `<spell>` tags |
| **Global hotkeys** | System-wide keyboard shortcuts for toggle, stop, and voice input |
| **Media key support** | Use hardware play/pause and volume keys to control TTS playback |
| **Voice commands** | Say "pause", "louder", "faster", "repeat", etc. during playback |
| **Stop phrases** | Say "stop", "quiet", or "shut up" to halt speech immediately |
| **Low latency** | Parallel chunk generation, SIGUSR1 instant queue notification, persistent audio stream |
| **Hot-reload config** | Change settings without restarting -- the daemon reloads config automatically |
| **Audio device routing** | Route output to any device by name (e.g. "AirPods") or index, with automatic switching |
| **Bluetooth mic workaround** | Automatically uses built-in mic when output is Bluetooth (avoids codec downgrade) |
| **Chimes** | Programmatic audio feedback tones for state changes (ready, error, stop) |

---

## Configuration Reference

All configuration lives in `claude-speak.toml` at the project root (or `~/.claude-speak/config.toml`). Copy `claude-speak.toml.example` to get started. Every setting has a sensible default.

The daemon hot-reloads the config file every 30 seconds. Environment variables (`CLAUDE_SPEAK_VOICE`, `CLAUDE_SPEAK_SPEED`, `CLAUDE_SPEAK_DEVICE`) override TOML values.

### `[tts]`

| Key | Default | Description |
|-----|---------|-------------|
| `engine` | `"kokoro"` | TTS engine: `"kokoro"`, `"piper"`, or `"elevenlabs"` |
| `voice` | `"af_sarah"` | Voice ID, or a blend like `"bm_george:60+bm_fable:40"` |
| `speed` | `1.0` | Playback speed multiplier (1.0 = natural, 1.3 = conversational) |
| `lang` | `"en-us"` | Language code for Kokoro (`en-us`, `en-gb`, etc.) |
| `device` | `"auto"` | Audio output: `"auto"`, a substring like `"AirPods"`, or a device index |
| `max_chunk_chars` | `400` | Max characters per TTS chunk (smaller = faster first audio) |
| `volume` | `1.0` | TTS speech volume (0.0--1.0) |
| `elevenlabs_api_key` | `""` | ElevenLabs API key (env var `ELEVENLABS_API_KEY` takes priority) |

### `[wakeword]`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable wake word detection |
| `engine` | `"openwakeword"` | Detection engine |
| `model` | `"hey_jarvis"` | Built-in model name or path to a custom `.onnx` file |
| `stop_model` | `""` | Path to a trained stop-word `.onnx` model |
| `sensitivity` | `0.5` | Detection threshold (0.0--1.0; lower = fewer false positives) |
| `stop_sensitivity` | `0.5` | Separate threshold for stop model |
| `stop_phrases` | `["stop", "quiet", "shut up"]` | Phrases that immediately halt TTS |

### `[input]`

| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `"builtin"` | STT backend: `"builtin"`, `"superwhisper"`, or `"auto"` |
| `auto_submit` | `true` | Automatically press Enter after transcription |
| `stt_backend` | `"auto"` | STT engine: `"auto"`, `"mlx"`, or `"whisper_cpp"` |
| `stt_model` | `"base"` | Whisper model size: `"tiny"`, `"base"`, `"small"`, `"medium"` |
| `vad_threshold` | `0.5` | Silero VAD speech probability threshold |

### `[audio]`

| Key | Default | Description |
|-----|---------|-------------|
| `chimes` | `true` | Play chime sounds on state changes |
| `greeting` | `"Ready."` | Text spoken on daemon start (empty string to disable) |
| `volume` | `0.3` | Chime volume (0.0--1.0; does not affect TTS voice volume) |
| `bt_mic_workaround` | `true` | Use built-in mic when output is Bluetooth |
| `media_keys_enabled` | `true` | Intercept hardware media keys for TTS control |

### `[normalization]`

| Key | Default | Description |
|-----|---------|-------------|
| `skip_code` | `true` | Replace code blocks with brief spoken descriptions |
| `expand_units` | `true` | Expand unit abbreviations (`"5ms"` becomes `"5 milliseconds"`) |
| `expand_abbreviations` | `true` | Expand tech abbreviations (`"API"` becomes `"A P I"`) |
| `shorten_paths` | `true` | Shorten long file paths to just the filename |
| `custom_pronunciations` | `""` | Path to a custom `pronunciations.toml` file |
| `context_aware` | `true` | Insert SSML tags based on detected content type |

### `[hotkeys]`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable global keyboard shortcuts |
| `toggle_tts` | `"cmd+shift+s"` | Toggle TTS on/off |
| `stop_playback` | `"cmd+shift+x"` | Stop current playback |
| `voice_input` | `"cmd+shift+v"` | Trigger voice input |

### `[voice_commands]`

| Key | Default | Description |
|-----|---------|-------------|
| `pause` | `"pause"` | Pause TTS playback |
| `resume` | `"resume"` | Resume TTS playback |
| `repeat` | `"repeat"` | Repeat last spoken text |
| `louder` | `"louder"` | Increase volume |
| `quieter` | `"quieter"` | Decrease volume |
| `faster` | `"faster"` | Increase speed |
| `slower` | `"slower"` | Decrease speed |
| `stop` | `"stop"` | Stop playback and clear queue |

---

## Available Voices

### Kokoro (default engine)

Kokoro ships with a set of built-in voices. Run `claude-speak voices` to see the full list. Common options:

| Voice ID | Description |
|----------|-------------|
| `af_sarah` | American female (default) |
| `af_bella` | American female |
| `af_nicole` | American female |
| `am_adam` | American male |
| `am_michael` | American male |
| `bf_emma` | British female |
| `bm_george` | British male |
| `bm_fable` | British male |

You can blend any two voices:

```toml
[tts]
voice = "am_adam:60+bm_george:40"   # 60% Adam, 40% George
```

Preview voices before choosing:

```bash
claude-speak preview af_sarah
claude-speak preview "am_adam:60+bm_george:40"
claude-speak preview --all
```

### Piper (alternative local engine)

Piper voices are downloaded on first use from HuggingFace:

| Voice | Description |
|-------|-------------|
| `en_US-lessac-medium` | US English, female, general purpose |
| `en_US-ryan-medium` | US English, male |
| `en_GB-alan-medium` | British English, male |

### ElevenLabs (cloud engine)

Use any ElevenLabs voice ID. Requires an API key set via `elevenlabs_api_key` in config or the `ELEVENLABS_API_KEY` environment variable.

---

## Architecture Overview

```
Claude Code --> Hook (PostToolUse/Stop) --> IPC Socket --> Daemon --> Normalizer --> TTS Engine --> Audio
                                                            ^
Wake Word (openwakeword) --> Voice Controller --> STT (MLX Whisper) --> Auto-submit to Claude
```

**Hook layer**: Claude Code fires hooks after every tool use and at response completion. The hook sends new text to the daemon via a Unix domain socket (with file-based queue fallback).

**Daemon**: A persistent background process that loads the TTS model once, watches for incoming text, normalizes it, chunks it, and streams audio with parallel generation for gapless playback.

**Normalizer**: A 24-step regex pipeline that transforms markdown, code blocks, tables, URLs, currency, math, dates, units, abbreviations, and more into natural spoken prose.

**TTS Engine**: Backend-agnostic synthesis with streaming playback, device routing, and gapless chunk transitions via a persistent audio stream.

**Voice Controller**: Orchestrates wake word detection, voice activity detection (Silero VAD), speech-to-text (MLX Whisper), and auto-submission to Claude Code.

For a detailed breakdown of every module, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## CLI Commands

All commands are available via the `claude-speak` entry point after installation.

### Daemon

| Command | Description |
|---------|-------------|
| `claude-speak start` | Start the daemon in the background |
| `claude-speak stop` | Stop the daemon gracefully |
| `claude-speak restart` | Kill all instances and start fresh |
| `claude-speak status` | Show PID, uptime, queue depth, voice, speed, device |
| `claude-speak kill-all` | Force-kill all claude-speak processes |

### Speech

| Command | Description |
|---------|-------------|
| `claude-speak say "text"` | Speak text immediately (loads model inline, no daemon needed) |
| `claude-speak speak "text"` | Send text to the running daemon via IPC socket |
| `claude-speak queue "text"` | Normalize, chunk, and enqueue text for the daemon |
| `claude-speak clear` | Clear the TTS queue |

### Playback Control

| Command | Description |
|---------|-------------|
| `claude-speak enable` | Enable voice output |
| `claude-speak disable` | Disable voice output (daemon stays loaded but silent) |
| `claude-speak pause` | Mute TTS playback |
| `claude-speak resume` | Unmute TTS playback |
| `claude-speak volume 0.8` | Set TTS volume (0.1--1.0) |
| `claude-speak speed 1.3` | Set TTS speed |

### Voice & Models

| Command | Description |
|---------|-------------|
| `claude-speak voices` | List all available TTS voices |
| `claude-speak preview af_sarah` | Hear a sample in a specific voice |
| `claude-speak preview --all` | Hear a sample in every voice |
| `claude-speak stt-models` | List available Whisper STT models and sizes |

### Voice Input

| Command | Description |
|---------|-------------|
| `claude-speak listen` | Start the voice controller (wake word + auto-submit) |
| `claude-speak voice-input` | Trigger a single voice input cycle |
| `claude-speak train-wakeword "hey claude"` | Train a custom wake word from recorded samples |

### Setup & Config

| Command | Description |
|---------|-------------|
| `claude-speak setup` | First-time setup (download models, install hooks, create config) |
| `claude-speak config` | Print all current configuration values |
| `claude-speak log` | Show the last 20 lines of the daemon log |
| `claude-speak uninstall` | Remove runtime artifacts and hooks |
| `claude-speak uninstall --all` | Also remove `~/.claude-speak/` (models and config) |

---

## Optional Dependencies

The base install includes only what is needed for Kokoro TTS output. Additional features require optional extras:

```bash
# Wake word detection ("Hey Jarvis" and custom wake words)
pip install claude-speak[wakeword]

# Piper TTS engine (lightweight local alternative)
pip install claude-speak[piper]

# ElevenLabs cloud TTS
pip install claude-speak[elevenlabs]

# Built-in speech-to-text via MLX Whisper (Apple Silicon)
pip install claude-speak[stt]

# Global hotkeys and media key support (requires macOS Accessibility permissions)
pip install claude-speak[macos-extras]

# Custom wake word training
pip install claude-speak[train]

# Development tools (ruff, pytest, mypy, pre-commit)
pip install claude-speak[dev]

# Install everything
pip install claude-speak[wakeword,stt,macos-extras]
```

---

## FAQ

**Q: Does this work on Linux or Windows?**
A: Not yet. claude-speak currently requires macOS for audio routing, Accessibility APIs (hotkeys, media keys), and Apple Silicon acceleration (MLX Whisper). Linux support is planned -- track progress at [github.com/vnicivanov/claude-speak/issues](https://github.com/vnicivanov/claude-speak/issues).

**Q: Do I need a paid Superwhisper subscription for voice input?**
A: No. The default STT backend is built-in MLX Whisper, which is fully local and free. Superwhisper is still supported as an alternative backend (`backend = "superwhisper"` in `[input]`), but it is no longer required.

**Q: How large are the model downloads?**
A: The Kokoro TTS model is ~350 MB total (ONNX model + voice data). Whisper STT models range from ~39 MB (tiny) to ~769 MB (medium). The `base` model (~74 MB) is the default and works well for most use cases. Models are downloaded once to `~/.claude-speak/models/` and cached.

**Q: Can I use a custom wake word instead of "Hey Jarvis"?**
A: Yes. Run `claude-speak train-wakeword "hey claude"` to record samples and train a custom model. The trained `.onnx` file is saved automatically, and you can point to it in config: `model = "/path/to/custom.onnx"`.

**Q: How do I reduce latency to first audio?**
A: Lower `max_chunk_chars` in `[tts]` (try 100--200). Smaller chunks generate faster, and parallel generation ensures subsequent chunks are ready before the current one finishes playing. Check `claude-speak log` for `[perf]` lines to identify bottlenecks.

**Q: Can I use ElevenLabs instead of local TTS?**
A: Yes. Install the extra (`pip install claude-speak[elevenlabs]`), set `engine = "elevenlabs"` in `[tts]`, and provide your API key via `elevenlabs_api_key` in config or the `ELEVENLABS_API_KEY` environment variable.

**Q: Why does the daemon not respond after I restart Claude Code?**
A: The SessionStart hook automatically starts the daemon, but if it was killed abnormally, a stale lock file might block startup. Run `claude-speak kill-all` then `claude-speak start`, or check `claude-speak log` for details.

**Q: How do I route audio to AirPods or another device?**
A: Set `device = "AirPods"` in `[tts]` (substring match against device names). Use `device = "auto"` for the system default. The daemon re-checks the device every 30 seconds, so it picks up newly connected devices automatically.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on development setup, running tests, and submitting pull requests.

---

## License

[MIT](LICENSE)
